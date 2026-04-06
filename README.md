# Insurance Claims Agent

> Agente de IA empresarial para procesamiento automatizado de reclamaciones de seguros,
> construido con **NVIDIA NeMo Agent Toolkit v1.5**.

[![NVIDIA NeMo](https://img.shields.io/badge/NVIDIA_NeMo_Agent_Toolkit-v1.5-76b900?style=flat-square&logo=nvidia&logoColor=white)](https://developer.nvidia.com/nemo)
[![LangChain](https://img.shields.io/badge/LangChain-ReAct_Agent-1C3C3C?style=flat-square)](https://python.langchain.com)
[![Milvus](https://img.shields.io/badge/Milvus-Vector_DB-00A1EA?style=flat-square)](https://milvus.io)
[![Phoenix](https://img.shields.io/badge/Arize_Phoenix-Observability-7C3AED?style=flat-square)](https://phoenix.arize.com)
[![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)

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

### Stack técnico

| Capa | Tecnología | Rol |
|---|---|---|
| **Agent Runtime** | NVIDIA NeMo Agent Toolkit v1.5 | Orquestación ReAct, API REST, config declarativa |
| **LLM** | Llama 3.3 70B via NVIDIA NIM | Clasificación, razonamiento, decisiones de cobertura |
| **Embeddings** | `nv-embedqa-e5-v5` via NVIDIA NIM | Vectorización de documentos de póliza |
| **Vector DB** | Milvus (Docker) | Almacenamiento persistente, búsqueda semántica |
| **Framework** | LangChain | Herramientas RAG, cadenas de razonamiento |
| **Observabilidad** | Arize Phoenix | Trazas OTEL end-to-end de cada tool call |
| **API** | FastAPI (via NAT) | REST endpoint OpenAI-compatible |
| **Demo UI** | Streamlit | Interfaz visual para demostración |
| **Infra** | Docker Compose | Milvus + etcd + MinIO |

### Herramientas del agente

| Herramienta | Input | Output | Tecnología |
|---|---|---|---|
| `classify_claim` | Descripción en texto libre | Tipo, prioridad, complejidad, handler | Llama 3.3 70B + Pydantic schema |
| `policy_search` | Pregunta sobre cobertura | Fragmentos relevantes de póliza (top-k=4) | Milvus RAG + NVIDIA embeddings |
| `check_coverage` | Escenario del siniestro | Cobertura, deducible, límite, exclusiones | RAG + Llama 3.3 70B |
| `required_docs` | Tipo de reclamación | Lista de documentos requeridos | Lookup table determinista |

---

## Ejemplos de uso

### Auto — Conductor dio fuga

```
Input: "Mi auto fue chocado en el estacionamiento. El culpable huyó sin dejar nota.
        Hay cámara CCTV que registró la placa. Póliza de auto con daños propios."

  classify_claim   → { claim_type: "auto", priority: "high",
                       complexity: "moderate", handler: "junior_adjuster" }

  policy_search    → [Auto policy, p.3] Coverage A — Collision: covers damage
                     to insured vehicle regardless of fault.
                     Deductible: 3% of agreed value (MXN $420,000 → $12,600).

  check_coverage   → { coverage_status: "fully_covered",
                       deductible: "MXN $12,600",
                       coverage_limit: "MXN $420,000 (valor acordado)" }

  required_docs    → Formulario de reclamación, reporte policial, fotos del daño,
                     licencia, tarjeta de circulación, estimado de taller autorizado
```

### Hogar — Tormenta e inundación

```
Input: "Tormenta severa rompió dos ventanas. El agua entró e inundó el sótano.
        Perdí electrodomésticos por $80,000 MXN. Póliza de hogar."

  classify_claim   → { claim_type: "home", priority: "high",
                       complexity: "moderate", handler: "senior_adjuster" }

  policy_search    → [Home policy, p.4] Coverage B — Water Damage: covers
                     sudden and accidental water damage from storms.
                     Contents sublimit: MXN $50,000.

  check_coverage   → { coverage_status: "partially_covered",
                       deductible: "MXN $8,500",
                       coverage_limit: "MXN $50,000 (contents sublimit)" }

  required_docs    → Formulario, fotos de daños, facturas de electrodomésticos,
                     reporte meteorológico, estimados de reparación
```

### Vida — Fallecimiento del asegurado

```
Input: "Solicito pago de suma asegurada por fallecimiento de mi esposo.
        Póliza de vida entera, suma $5,000,000 MXN. Soy beneficiaria al 100%."

  classify_claim   → { claim_type: "life", priority: "urgent",
                       complexity: "simple", handler: "junior_adjuster" }

  policy_search    → [Life policy, p.2] Death Benefit: MXN $5,000,000 payable
                     to designated beneficiary within 20 business days.

  check_coverage   → { coverage_status: "fully_covered",
                       coverage_limit: "MXN $5,000,000",
                       deductible: "N/A" }

  required_docs    → Acta de defunción (original), póliza original,
                     identificación del beneficiario, acta de matrimonio,
                     formulario de reclamación de beneficiario
```

---

## Demo rápida

### Prerequisitos

- Docker Desktop en ejecución
- Python 3.12+
- NVIDIA API Key — [gratis en build.nvidia.com](https://build.nvidia.com)

### Setup

```bash
# Clonar e instalar
git clone https://github.com/Richard7856/AgentNeMo.git
cd AgentNeMo
pip install -e .

# Configurar API key
cp .env.example .env
# Editar .env → NVIDIA_API_KEY=nvapi-...

# Levantar infraestructura e indexar pólizas
make infra-up
make ingest
```

### Ejecutar

```bash
# Demo UI (Streamlit)
make serve          # Terminal 1: NAT REST API en :8000
make demo           # Terminal 2: UI en :8501

# CLI
make run INPUT="Mi auto fue chocado en el estacionamiento por un conductor que huyó"

# Observabilidad (Phoenix)
make phoenix        # Dashboard de trazas en :6006
```

---

## Estructura del proyecto

```
insurance-claims-agent/
├── configs/
│   └── config.yml                 # Config declarativa del agente NAT
├── demo/
│   └── app.py                     # Demo UI (Streamlit)
├── docs/
│   └── DECISIONS.md               # 9 decisiones de arquitectura documentadas
├── eval/
│   └── datasets/claims_eval.json  # 16 casos de prueba
├── insurance_claims/
│   ├── register.py                # Entry point NAT
│   ├── tools/
│   │   ├── classify_claim.py      # Clasificación con Pydantic schema
│   │   ├── policy_search.py       # RAG: Milvus + NVIDIA embeddings
│   │   ├── check_coverage.py      # Cobertura: RAG + razonamiento LLM
│   │   └── required_docs.py       # Lookup de documentos requeridos
│   └── data_prep/
│       ├── generate_policies.py   # Generación de PDFs sintéticos
│       └── ingest_policies.py     # Parsing + embedding + ingesta
├── data/policies/                 # 3 pólizas sintéticas (Auto, Hogar, Vida)
├── docker-compose.yml             # Milvus + etcd + MinIO
└── Makefile                       # Comandos de conveniencia
```

---

## Decisiones de arquitectura

Ver [`docs/DECISIONS.md`](docs/DECISIONS.md) para el razonamiento documentado:

- Por qué **Milvus** sobre FAISS (persistencia, producción, concurrencia)
- Por qué **Llama 3.3 70B** para razonamiento multi-paso
- Por qué **un agente ReAct** vs. arquitectura multi-agente
- Integración de **Phoenix tracing** con LangChain

---

## Casos de prueba

16 casos en `eval/datasets/claims_eval.json`:

- **Auto** — colisión, robo, daños por tercero no asegurado
- **Hogar** — tormenta, inundación, robo
- **Vida** — fallecimiento, beneficiarios
- **Multi-tool** — clasificación + cobertura + documentos en secuencia

---

## Autor

**Richard Figueroa** — AI Implementation Engineer & Automation Architect

[LinkedIn](https://linkedin.com/in/richard-figueroaluna) · [GitHub](https://github.com/Richard7856) · rfigue97@gmail.com
