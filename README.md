# Insurance Claims Agent

> Agente de IA empresarial para procesamiento automatizado de reclamaciones de seguros,
> construido con **NVIDIA NeMo Agent Toolkit v1.5**.

[![NVIDIA NeMo](https://img.shields.io/badge/NVIDIA_NeMo_Agent_Toolkit-v1.5-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/nemo)
[![LangChain](https://img.shields.io/badge/LangChain-ReAct_Agent-1C3C3C?style=flat-square)](https://python.langchain.com)
[![Milvus](https://img.shields.io/badge/Milvus-Vector_DB-00A1EA?style=flat-square)](https://milvus.io)
[![Phoenix](https://img.shields.io/badge/Arize_Phoenix-Observability-7C3AED?style=flat-square)](https://phoenix.arize.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

---

<!-- Replace this comment with: ![Demo](docs/demo.gif) after recording -->

---

## Qué hace este agente

El agente recibe una **descripción de siniestro en lenguaje natural** y en un solo paso de razonamiento multi-etapa:

1. **Clasifica** el tipo de reclamación, prioridad y complejidad
2. **Busca** en las pólizas relevantes via RAG semántico (Milvus)
3. **Determina** cobertura, deducible aplicable y exclusiones
4. **Lista** los documentos requeridos para el trámite

```
Usuario → "Tormenta rompió mis ventanas e inundó el sótano. Póliza de hogar."

Agente →
  [Thought]      Necesito clasificar esta reclamación primero...
  [Action]       classify_claim(claim_description="...")
  [Observation]  { claim_type: "home", priority: "high", complexity: "moderate" }

  [Thought]      Debo buscar la cobertura aplicable en la póliza...
  [Action]       policy_search(query="daños por tormenta cobertura hogar")
  [Observation]  [Excerpt — Home policy, page 4] Coverage B — Water Damage...

  [Thought]      Con el contexto de la póliza, determino la cobertura...
  [Action]       check_coverage(claim_scenario="...")
  [Observation]  { coverage_status: "fully_covered", deductible: "MXN $8,500" }

  [Thought]      Necesito los documentos requeridos...
  [Action]       required_docs(claim_type="home")

  [Final Answer] La reclamación está cubierta al 100% bajo Cobertura B...
```

---

## Stack Técnico

| Capa | Tecnología | Rol |
|---|---|---|
| **Agent Runtime** | NVIDIA NeMo Agent Toolkit v1.5 | Orquestación ReAct, API REST, config declarativa |
| **LLM** | Llama 3.3 70B via NVIDIA NIM | Clasificación, razonamiento, decisiones de cobertura |
| **Embeddings** | `nv-embedqa-e5-v5` via NVIDIA NIM | Vectorización de documentos de póliza |
| **Vector DB** | Milvus (Docker) | Almacenamiento persistente de embeddings, búsqueda semántica |
| **Framework** | LangChain | Herramientas RAG, cadenas de razonamiento |
| **Observabilidad** | Arize Phoenix | Trazas OTEL end-to-end de cada tool call |
| **API** | FastAPI (via NAT) | REST endpoint OpenAI-compatible en `/v1/chat/completions` |
| **Demo UI** | Streamlit | Interfaz visual para demostración |
| **Infra** | Docker Compose | Milvus + etcd + MinIO |

---

## Arquitectura

```
┌─────────────────────────────────────────────────────┐
│                  NAT ReAct Orchestrator              │
│              (meta/llama-3.3-70b-instruct)          │
└───────────┬─────────┬──────────────┬───────────────┘
            │         │              │
     ┌──────▼──┐ ┌────▼──────┐ ┌────▼─────────────┐
     │classify │ │  policy   │ │  check_coverage  │
     │ _claim  │ │  _search  │ │  (RAG + LLM)     │
     │  (LLM)  │ │  (RAG)    │ └──────────────────┘
     └─────────┘ └────┬──────┘
                      │
             ┌────────▼────────┐
             │   Milvus DB     │  ← 3 pólizas indexadas
             │  (Docker)       │     Auto / Hogar / Vida
             └─────────────────┘

         Observabilidad: Arize Phoenix (OTEL traces)
         API: POST /v1/chat/completions (OpenAI-compatible)
```

### Herramientas del Agente

| Herramienta | Input | Output | Tecnología |
|---|---|---|---|
| `classify_claim` | Descripción en texto libre | Tipo, prioridad, complejidad, manejador recomendado | Llama 3.3 70B + Pydantic schema |
| `policy_search` | Pregunta sobre cobertura | Fragmentos relevantes de póliza (top-k=4) | Milvus RAG + NVIDIA embeddings |
| `check_coverage` | Escenario del siniestro | Cobertura, deducible, límite, exclusiones | RAG + Llama 3.3 70B |
| `required_docs` | Tipo de reclamación | Lista de documentos requeridos | Lookup table determinista |

---

## Demo Rápida

### Prerequisitos

- Docker Desktop en ejecución
- Python 3.12+
- NVIDIA API Key — [gratis en build.nvidia.com](https://build.nvidia.com)

### Setup (3 comandos)

```bash
# 1. Clonar e instalar dependencias
git clone <repo-url> && cd insurance-claims-agent
pip install -e .         # o: uv sync

# 2. Configurar API key
cp .env.example .env
# Editar .env y agregar: NVIDIA_API_KEY=nvapi-...

# 3. Levantar infraestructura e indexar pólizas
make infra-up
make ingest
```

### Demo UI (Streamlit)

```bash
# Terminal 1: NAT REST API
make serve

# Terminal 2: Streamlit UI
make demo
# → http://localhost:8501
```

### CLI

```bash
make run INPUT="Mi auto fue chocado en el estacionamiento por un conductor que huyó"
```

### Observabilidad con Phoenix

```bash
# Terminal adicional — ver trazas ReAct en tiempo real
make phoenix
# → http://localhost:6006
```

---

## Estructura del Proyecto

```
insurance-claims-agent/
├── configs/
│   └── config.yml              # Configuración declarativa del agente NAT
├── demo/
│   └── app.py                  # Demo UI (Streamlit)
├── docs/
│   └── DECISIONS.md            # 9 decisiones de arquitectura documentadas
├── eval/
│   └── datasets/claims_eval.json  # 16 casos de prueba
├── insurance_claims/
│   ├── register.py             # Entry point NAT (descubrimiento de herramientas)
│   ├── tools/
│   │   ├── classify_claim.py   # Clasificación estructurada con schema Pydantic
│   │   ├── policy_search.py    # RAG tool: Milvus + NVIDIA embeddings
│   │   ├── check_coverage.py   # Cobertura: RAG + razonamiento LLM
│   │   └── required_docs.py    # Lookup de documentos requeridos
│   └── data_prep/
│       ├── generate_policies.py  # Generación de PDFs sintéticos (fpdf2)
│       └── ingest_policies.py    # Parsing + embedding + ingesta a Milvus
├── data/policies/              # 3 pólizas sintéticas (Auto, Hogar, Vida)
├── docker-compose.yml          # Milvus + etcd + MinIO
└── Makefile                    # Comandos de conveniencia
```

---

## Comandos de Referencia

```bash
make serve        # API REST en :8000
make demo         # Demo UI en :8501
make phoenix      # Dashboard de trazas en :6006
make trace        # 3 queries de demo con Phoenix habilitado
make ingest       # Indexar PDFs en Milvus
make infra-up     # Levantar Docker stack
make infra-down   # Apagar containers (datos preservados)
make validate     # Validar config.yml
```

---

## Casos de Prueba Incluidos

El dataset `eval/datasets/claims_eval.json` contiene 16 casos organizados por herramienta:

- **Auto** — colisión, robo, daños por tercero no asegurado
- **Hogar** — tormenta, inundación, robo
- **Vida** — fallecimiento, beneficiarios
- **Multi-tool** — casos que ejercen clasificación + cobertura + documentos en secuencia

---

## Decisiones de Arquitectura

Ver [`docs/DECISIONS.md`](docs/DECISIONS.md) para el razonamiento detrás de cada decisión técnica:

- Por qué **Milvus** sobre FAISS (persistencia, producción, concurrencia)
- Por qué **Llama 3.3 70B** para razonamiento de cobertura multi-paso
- Por qué **un agente ReAct** con herramientas especializadas vs. arquitectura multi-agente
- Cómo se integra **Phoenix tracing** con el pipeline de LangChain
- Decisiones de layout de paquete Python para paths de iCloud

---

## Autor

**Richard Figueroa** — Developer & AI/Automation Consultant

Especializado en sistemas de IA empresarial con n8n, NVIDIA NeMo, LangChain y FastAPI
para automatización de procesos en sectores de seguros, distribución y servicios financieros.
