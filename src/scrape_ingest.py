"""
scrape_ingest.py — Fetch web articles and ingest them into Qdrant in one pass.

Supports single URLs or a file with one URL per line.
Skips PDFs, already-processed URLs (tracked in data/.processed_urls.txt),
and applies a polite delay between requests.

Usage examples:
  # Single article
  python src/scrape_ingest.py https://raypeat.com/articles/articles/thyroid.shtml \
      --author "Ray Peat" --paradigm bioenergetic --domain endocrinology

  # Batch from file
  python src/scrape_ingest.py --url-file data/urls_ray_peat.txt \
      --author "Ray Peat" --paradigm bioenergetic --domain nutrition \
      --delay 2.0 --dry-run
"""

import argparse
import ipaddress
import re
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Reuse existing ingest pipeline
sys.path.insert(0, str(Path(__file__).parent))
from ingest import (
    _connect,
    ensure_collection,
    build_points,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

load_dotenv()

EMBEDDING_MODEL = "BAAI/bge-m3"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
COLLECTION_NAME = "health_rag"
PROCESSED_LOG = Path("data/.processed_urls.txt")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; health-rag-bot/1.0; "
        "+https://github.com/health-rag)"
    )
}

_MAX_FETCH_REDIRECTS = 10


def _ip_is_public_routable(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """True only if *ip* is global unicast / usable on the public Internet (SSRF-safe)."""
    if (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_multicast
        or ip.is_reserved
        or ip.is_unspecified
    ):
        return False
    if ip.version == 6 and ip.is_site_local:
        return False
    return True


def _resolved_hosts_public(hostname: str) -> None:
    """Raise ValueError if *hostname* does not resolve only to public routable addresses."""
    addrs: list[str] = []
    try:
        for fam, _, _, _, sockaddr in socket.getaddrinfo(
            hostname, None, type=socket.SOCK_STREAM
        ):
            if fam in (socket.AF_INET, socket.AF_INET6):
                addrs.append(sockaddr[0])
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve host {hostname!r}: {exc}") from exc

    if not addrs:
        raise ValueError(f"No addresses resolved for host {hostname!r}")

    seen: set[str] = set()
    for addr in addrs:
        if addr in seen:
            continue
        seen.add(addr)
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError as exc:
            raise ValueError(f"Invalid resolved address {addr!r} for host {hostname!r}") from exc
        if not _ip_is_public_routable(ip):
            raise ValueError(
                f"Host {hostname!r} resolves to a non-public address {addr!r} "
                "(blocked for SSRF protection)."
            )


def _assert_ssrf_safe_url(url: str) -> None:
    """
    Allow only http(s) URLs whose host resolves exclusively to public routable IPs.
    Blocks loopback, RFC1918, link-local (incl. cloud metadata), and literal unsafe IPs.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Only http(s) URLs are allowed, got scheme {parsed.scheme!r}")

    host = parsed.hostname
    if not host:
        raise ValueError("URL has no hostname")

    ip_literal: ipaddress.IPv4Address | ipaddress.IPv6Address | None = None
    try:
        ip_literal = ipaddress.ip_address(host)
    except ValueError:
        if host.isdigit():
            try:
                val = int(host)
                if 0 <= val <= 0xFFFFFFFF:
                    ip_literal = ipaddress.ip_address(val)
            except ValueError:
                pass

    if ip_literal is not None:
        if not _ip_is_public_routable(ip_literal):
            raise ValueError(
                f"Address {host!r} is not a public routable target (SSRF protection)."
            )
        return

    _resolved_hosts_public(host)


def _http_get_ssrf_safe(url: str, *, max_redirects: int = _MAX_FETCH_REDIRECTS) -> requests.Response:
    """
    GET *url* following redirects manually so each hop is validated (open redirect SSRF).
    """
    current = url
    session = requests.Session()
    for _ in range(max_redirects + 1):
        _assert_ssrf_safe_url(current)
        resp = session.get(
            current,
            headers=HEADERS,
            timeout=20,
            allow_redirects=False,
        )
        if resp.status_code in (301, 302, 303, 307, 308):
            loc = resp.headers.get("Location")
            if not loc:
                resp.raise_for_status()
            current = urljoin(current, loc)
            continue
        resp.raise_for_status()
        return resp
    raise ValueError(
        f"Too many redirects (>{max_redirects}) while fetching — aborting for safety."
    )

# Map URL path keywords → domain tag (first match wins)
DOMAIN_HINTS: list[tuple[str, str]] = [
    ("thyroid", "endocrinology"),
    ("estrogen", "endocrinology"),
    ("progesterone", "endocrinology"),
    ("hormone", "endocrinology"),
    ("cancer", "oncology"),
    ("diabetes", "metabolic"),
    ("glucose", "metabolic"),
    ("sugar", "nutrition"),
    ("salt", "nutrition"),
    ("fat", "nutrition"),
    ("oil", "nutrition"),
    ("milk", "nutrition"),
    ("vitamin", "nutrition"),
    ("aging", "aging"),
    ("alzheimer", "aging"),
    ("osteoporosis", "aging"),
    ("brain", "neuroscience"),
    ("serotonin", "neuroscience"),
    ("stress", "neuroscience"),
]


# ─── helpers ────────────────────────────────────────────────────────────────

def load_processed() -> set[str]:
    if PROCESSED_LOG.exists():
        return set(PROCESSED_LOG.read_text(encoding="utf-8").splitlines())
    return set()


def mark_processed(url: str) -> None:
    PROCESSED_LOG.parent.mkdir(parents=True, exist_ok=True)
    with PROCESSED_LOG.open("a", encoding="utf-8") as f:
        f.write(url + "\n")


def guess_domain(url: str, fallback: str) -> str:
    slug = urlparse(url).path.lower()
    for keyword, domain in DOMAIN_HINTS:
        if keyword in slug:
            return domain
    return fallback


def guess_keywords(url: str) -> list[str]:
    slug = Path(urlparse(url).path).stem.replace("-", " ")
    words = [w for w in slug.split() if len(w) > 3]
    return words[:8]


def fetch_article(url: str) -> tuple[str, str]:
    """Return (title, body_text) from a URL. Raises on error."""
    resp = _http_get_ssrf_safe(url)
    final_url = resp.url

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noisy tags
    for tag in soup(["script", "style", "nav", "header", "footer", "noscript", "iframe"]):
        tag.decompose()

    # Title
    title_tag = soup.find("title")
    h1_tag = soup.find("h1")
    if h1_tag:
        title = h1_tag.get_text(strip=True)
    elif title_tag:
        title = title_tag.get_text(strip=True)
    else:
        title = Path(urlparse(final_url).path).stem.replace("-", " ").title()

    # Body text — prefer <article> or <main>, fallback to <body>
    body = soup.find("article") or soup.find("main") or soup.find("body")
    if body is None:
        raise ValueError("No body content found")

    text = body.get_text(separator="\n")
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 200:
        raise ValueError(f"Content too short ({len(text)} chars) — likely a nav page")

    return title, text


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    docs = splitter.create_documents([text])
    return [d.page_content for d in docs]


# ─── core ───────────────────────────────────────────────────────────────────

def process_url(
    url: str,
    *,
    model: SentenceTransformer,
    client,
    author: str,
    paradigm: str,
    domain: str,
    auto_domain: bool,
    dry_run: bool,
) -> int:
    """Scrape + ingest one URL. Returns number of points upserted (0 on dry-run)."""
    if url.lower().endswith(".pdf"):
        print(f"  [SKIP] PDF not supported: {url}")
        return 0

    resolved_domain = guess_domain(url, domain) if auto_domain else domain
    keywords = guess_keywords(url)

    print(f"\n>> {url}")
    try:
        title, text = fetch_article(url)
    except Exception as exc:
        print(f"  [ERROR] fetch failed: {exc}")
        return 0

    chunks = chunk_text(text)
    print(f"  Title   : {title}")
    print(f"  Domain  : {resolved_domain}")
    print(f"  Chunks  : {len(chunks)}")

    if dry_run:
        print("  [DRY-RUN] skipping embed + upsert")
        return 0

    vectors = model.encode(
        chunks,
        show_progress_bar=False,
        normalize_embeddings=True,
    ).tolist()

    points = build_points(
        chunks,
        vectors,
        author=author,
        paradigm=paradigm,
        domain=resolved_domain,
        source_type="article",
        source_title=title,
        keywords=keywords,
    )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"  [OK] {len(points)} chunks upserted")
    return len(points)


# ─── CLI ────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Scrape web articles and ingest into health-rag Qdrant collection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "urls",
        nargs="*",
        help="One or more article URLs to scrape.",
    )
    p.add_argument(
        "--url-file",
        type=Path,
        help="Text file with one URL per line (lines starting with # are skipped).",
    )
    p.add_argument("--author", required=True, help='e.g. "Ray Peat"')
    p.add_argument("--paradigm", required=True, help='e.g. bioenergetic')
    p.add_argument(
        "--domain",
        default="health",
        help="Default domain tag; auto-detected from URL slug unless --no-auto-domain.",
    )
    p.add_argument(
        "--no-auto-domain",
        dest="auto_domain",
        action="store_false",
        default=True,
        help="Disable auto-detection of domain from URL; use --domain for all.",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Seconds to wait between requests (default: 1.5).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Fetch and chunk but do NOT embed or upsert.",
    )
    p.add_argument(
        "--skip-processed",
        action="store_true",
        default=True,
        help="Skip URLs already in data/.processed_urls.txt (default: on).",
    )
    p.add_argument(
        "--no-skip-processed",
        dest="skip_processed",
        action="store_false",
        help="Re-ingest even if URL was already processed.",
    )
    return p.parse_args()


def collect_urls(args: argparse.Namespace) -> list[str]:
    urls: list[str] = list(args.urls or [])
    if args.url_file:
        if not args.url_file.exists():
            sys.exit(f"URL file not found: {args.url_file}")
        for line in args.url_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    if not urls:
        sys.exit("No URLs provided. Pass URLs directly or use --url-file.")
    return urls


def main() -> None:
    args = parse_args()
    urls = collect_urls(args)
    processed = load_processed() if args.skip_processed else set()

    print(f"URLs to process : {len(urls)}")
    print(f"Already done    : {len(processed & set(urls))}")
    print(f"Dry-run         : {args.dry_run}")
    print(f"Auto-domain     : {args.auto_domain}")
    print(f"Delay           : {args.delay}s")

    if not args.dry_run:
        try:
            client = _connect()
            ensure_collection(client)
        except Exception as exc:
            sys.exit(f"Qdrant connection failed: {exc}")

        print(f"\nLoading embedding model '{EMBEDDING_MODEL}' ...")
        model = SentenceTransformer(EMBEDDING_MODEL)
    else:
        client = None
        model = None

    total_points = 0
    skipped = 0
    errors = 0

    for i, url in enumerate(urls):
        if url in processed:
            print(f"\n>> [ALREADY DONE] {url}")
            skipped += 1
            continue

        try:
            n = process_url(
                url,
                model=model,
                client=client,
                author=args.author,
                paradigm=args.paradigm,
                domain=args.domain,
                auto_domain=args.auto_domain,
                dry_run=args.dry_run,
            )
            total_points += n
            if not args.dry_run and n > 0:
                mark_processed(url)
        except Exception as exc:
            print(f"  [FATAL] {exc}")
            errors += 1

        if i < len(urls) - 1:
            time.sleep(args.delay)

    print(f"\n{'='*60}")
    print(f"Done. Processed {len(urls) - skipped - errors} URLs")
    print(f"  Skipped (already done) : {skipped}")
    print(f"  Errors                 : {errors}")
    print(f"  Total chunks upserted  : {total_points}")


if __name__ == "__main__":
    main()
