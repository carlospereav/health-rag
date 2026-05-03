# 🧬 Comparative Lifestyle RAG MVP

Un sistema RAG (Retrieval-Augmented Generation) avanzado diseñado para comparar diferentes paradigmas de salud y estilo de vida (ej. Bioenergética de Ray Peat vs. Biología Circadiana de Andrew Huberman vs. Ciencia Convencional).

Este proyecto implementa una arquitectura de **Enrutamiento de Consultas (Query Routing)** y **Búsqueda Híbrida (Hybrid Search)** para evitar el sesgo de dominancia de datos, asegurando que el LLM reciba contexto equilibrado de diferentes corrientes de pensamiento antes de emitir una síntesis.

## 🏗️ Arquitectura y Stack Tecnológico (Stack 0€ Local)

Este proyecto está diseñado para funcionar de manera local y gratuita, manteniendo la privacidad de los datos:

* **Base de Datos Vectorial:** [Qdrant](https://qdrant.tech/) (Local via Docker).
* **Embeddings (Locales):** `BAAI/bge-m3` (HuggingFace). Elegido específicamente por su soporte nativo para generar vectores densos (semántica) y dispersos (palabras clave exactas) simultáneamente.
* **LLM / Inferencia:** [Groq API](https://groq.com/) (Modelos Llama-3 / Mixtral) para síntesis ultrarrápida a coste cero.
* **Orquestación:** Python (LlamaIndex / LangChain).

## ✨ Características Principales

* **Stratified Retrieval:** Divide una consulta de usuario en sub-consultas dirigidas a metadatos específicos (`paradigm`).
* **Búsqueda Híbrida:** Combina búsqueda semántica profunda con coincidencias exactas de términos médicos o bioquímicos específicos (BM25/Sparse).
* **Fail-Fast & Seguridad:** Validaciones de entorno estrictas y configuración de Qdrant sin exposición de credenciales.

## 📋 Requisitos Previos

* **Python 3.11+**
* **Docker y Docker Compose** instalados y corriendo en la máquina.
* Una API Key gratuita de [Groq Cloud](https://console.groq.com/keys).

## 🚀 Instalación y Configuración

**1. Clonar el repositorio**

```bash
git clone https://github.com/TU_USUARIO/TU_REPOSITORIO.git
cd TU_REPOSITORIO
```

**2. Entorno Python y dependencias**

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

**3. Variables de entorno**

Copia `.env.example` a `.env` y rellena al menos:

| Variable | Descripción |
|----------|-------------|
| `QDRANT_URL` | URL del servidor Qdrant (p. ej. `http://localhost:6333` en local). |
| `QDRANT_API_KEY` | Clave de API de Qdrant. |
| `GROQ_API_KEY` | Clave de [Groq Cloud](https://console.groq.com/keys) (necesaria para `ask.py`). |
| `GROQ_MODEL` | *(Opcional)* ID del modelo en Groq; si no se define, se usa `llama-3.3-70b-versatile` (también puedes pasar `--model` en la CLI). |

**4. Qdrant con Docker**

```bash
docker compose up -d
```

Con la colección `health_rag` indexada (vía `src/ingest.py`), ya puedes buscar y preguntar.

## 🔎 Búsqueda semántica (solo recuperación)

Recupera los *chunks* más similares a la consulta (embeddings `BAAI/bge-m3` + Qdrant):

```bash
python src/search.py "tu consulta en lenguaje natural" -k 5
```

## 💬 Respuesta RAG con Groq (evidencia citada)

Cierra el bucle RAG: recupera los mismos top-*k* fragmentos que `search.py`, los formatea como contexto numerado `[1]`…`[k]` (con metadatos de fuente) y llama a la API de Groq para generar una respuesta **solo a partir de ese contexto**, con instrucciones de citar pasajes y no inventar fuera de ellos.

```bash
python src/ask.py "tu pregunta" -k 5
```

Opciones útiles:

* `--temperature` — temperatura de muestreo (por defecto `0.2`).
* `--model` — modelo Groq; si no se indica, se usa `GROQ_MODEL` del `.env` o el valor por defecto anterior.

Si no hay resultados en Qdrant, el script avisa y **no** llama al LLM. Si hay fragmentos pero falta `GROQ_API_KEY`, el proceso termina con un mensaje claro.