"""
Insurance Claims Agent — Demo UI
Powered by NVIDIA NeMo Agent Toolkit v1.5

Setup:
    make infra-up          # Start Milvus (Docker)
    make serve             # NAT REST API on :8000
    streamlit run demo/app.py

The UI calls the NAT FastAPI endpoint POST /v1/chat/completions
and parses the ReAct agent output into structured cards.
"""

import json
import re

import requests
import streamlit as st

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="Insurance Claims Agent · NVIDIA NeMo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand CSS — NVIDIA green accents ─────────────────────────────────────────
st.markdown(
    """
    <style>
    /* NVIDIA green accent */
    .nvidia-badge {
        background: #76b900; color: #000;
        padding: 2px 10px; border-radius: 12px;
        font-size: 0.75rem; font-weight: 700;
        display: inline-block; margin: 2px;
    }
    .tech-badge {
        background: #1e1e1e; color: #ccc;
        padding: 2px 8px; border-radius: 12px;
        font-size: 0.72rem; border: 1px solid #444;
        display: inline-block; margin: 2px;
    }
    .status-dot-green { color: #76b900; font-size: 1.1rem; }
    .status-dot-red   { color: #e84040; font-size: 1.1rem; }
    .step-thought     { border-left: 3px solid #4a90d9; padding-left: 10px; }
    .step-action      { border-left: 3px solid #76b900; padding-left: 10px; }
    .step-observation { border-left: 3px solid #f5a623; padding-left: 10px; }
    .step-answer      { border-left: 3px solid #9b59b6; padding-left: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Pre-built demo scenarios ──────────────────────────────────────────────────
# Each maps to a real eval case from eval/datasets/claims_eval.json
SCENARIOS = {
    "Auto — Conductor dio fuga": {
        "claim_type": "auto",
        "policy_number": "AUTO-2024-8821",
        "description": (
            "Estaba estacionado en el centro comercial y al regresar encontré "
            "mi auto con el parachoques trasero abollado y la defensa rayada. "
            "El conductor responsable no dejó nota. El estacionamiento tiene "
            "cámara CCTV que registró la placa. La póliza de auto incluye "
            "cobertura de daños propios con deducible del 3% sobre el valor acordado."
        ),
    },
    "Hogar — Tormenta e inundación": {
        "claim_type": "home",
        "policy_number": "HOGAR-2024-3347",
        "description": (
            "Una tormenta severa del pasado fin de semana rompió dos ventanas "
            "y el agua entró inundando el sótano. Perdí una lavadora, secadora "
            "y electrodomésticos valuados en aproximadamente $80,000 MXN. "
            "Tengo fotografías del daño y facturas de los equipos. Mi póliza "
            "de hogar cubre fenómenos hidrometeorológicos."
        ),
    },
    "Vida — Fallecimiento del asegurado": {
        "claim_type": "life",
        "policy_number": "VIDA-2024-1102",
        "description": (
            "Solicito el pago de la suma asegurada por el fallecimiento de mi "
            "esposo el 15 de marzo de 2026. Él era el asegurado principal con "
            "póliza de vida entera, suma asegurada de $5,000,000 MXN. "
            "Soy la beneficiaria designada al 100%. Tengo el acta de defunción "
            "y el documento original de la póliza."
        ),
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def check_api_health() -> bool:
    """Ping the NAT health endpoint — fast timeout so UI doesn't hang."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def call_agent(user_message: str) -> str:
    """
    POST to NAT's OpenAI-compatible endpoint and return the agent's full response.
    Uses non-streaming for simplicity — the agent is deterministic (temp=0.0).
    """
    payload = {
        "model": "insurance-claims-agent",
        "messages": [{"role": "user", "content": user_message}],
        "stream": False,
    }
    resp = requests.post(
        f"{API_BASE}/v1/chat/completions",
        json=payload,
        timeout=180,  # ReAct multi-step can take 30-60s
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def parse_react_trace(raw: str) -> list[dict]:
    """
    Extract Thought / Action / Action Input / Observation / Final Answer
    blocks from the ReAct agent's verbose output, preserving order.
    """
    # Each block runs until the next keyword or end of string
    pattern = re.compile(
        r"(Thought|Action Input|Action|Observation|Final Answer)\s*:[ \t]*(.*?)(?="
        r"\nThought:|\nAction:|\nObservation:|\nFinal Answer:|$)",
        re.DOTALL | re.IGNORECASE,
    )
    steps = []
    for m in pattern.finditer(raw):
        kind = m.group(1).strip().lower().replace(" ", "_")
        content = m.group(2).strip()
        steps.append({"kind": kind, "content": content})
    return steps


def try_parse_json(text: str) -> dict | None:
    """Try to extract the first JSON object from a string."""
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])
    except Exception:
        return None


def find_json_blocks(raw: str) -> tuple[dict | None, dict | None]:
    """
    Scan the full agent output for CoverageDecision and ClaimClassification
    JSON blocks. Returns (classification, coverage) or (None, None).
    """
    classification = None
    coverage = None
    # Find all JSON-like blocks (handles nested objects via balanced braces)
    for match in re.finditer(r"\{", raw):
        start = match.start()
        depth = 0
        end = start
        for i, ch in enumerate(raw[start:], start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        candidate = raw[start:end]
        obj = try_parse_json(candidate)
        if obj:
            if "coverage_status" in obj and coverage is None:
                coverage = obj
            elif "claim_type" in obj and classification is None:
                classification = obj
    return classification, coverage


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<span class="nvidia-badge">NVIDIA NeMo Agent Toolkit v1.5</span>',
        unsafe_allow_html=True,
    )
    st.markdown("## Insurance Claims Agent")
    st.caption("Agente de IA empresarial para reclamaciones de seguros")

    st.divider()

    is_healthy = check_api_health()
    if is_healthy:
        st.markdown(
            '<span class="status-dot-green">●</span> API activa — puerto 8000',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-dot-red">●</span> API offline — corre `make serve`',
            unsafe_allow_html=True,
        )
        st.code("make infra-up\nmake serve", language="bash")

    st.divider()
    st.markdown("**Stack técnico**")
    st.markdown(
        """
        <span class="nvidia-badge">NVIDIA NIM</span>
        <span class="nvidia-badge">NeMo Toolkit</span>
        <span class="tech-badge">LangChain</span>
        <span class="tech-badge">Milvus</span>
        <span class="tech-badge">Arize Phoenix</span>
        <span class="tech-badge">FastAPI</span>
        <span class="tech-badge">Docker</span>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("**Escenario de demo**")
    scenario_name = st.radio(
        "Selecciona:",
        list(SCENARIOS.keys()),
        label_visibility="collapsed",
    )

    st.divider()
    st.caption(
        "Trazabilidad completa en Arize Phoenix  \n"
        "http://localhost:6006"
    )


# ── Main ──────────────────────────────────────────────────────────────────────
st.markdown("# Insurance Claims Agent")
st.markdown(
    "Procesamiento automatizado de reclamaciones con razonamiento multi-paso · "
    "Modelo: `meta/llama-3.3-70b-instruct` via NVIDIA NIM"
)
st.divider()

scenario = SCENARIOS[scenario_name]

# ── Form + Results side by side ───────────────────────────────────────────────
left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    st.markdown("### Nueva reclamación")

    claim_type = st.selectbox(
        "Tipo de póliza",
        ["auto", "home", "life", "health", "liability"],
        index=["auto", "home", "life", "health", "liability"].index(
            scenario["claim_type"]
        ),
    )
    policy_number = st.text_input(
        "Número de póliza", value=scenario["policy_number"]
    )
    description = st.text_area(
        "Descripción del siniestro",
        value=scenario["description"],
        height=200,
    )

    user_prompt = (
        f"Reclamación de seguro\n"
        f"Tipo de póliza: {claim_type}\n"
        f"Número de póliza: {policy_number}\n\n"
        f"{description}\n\n"
        "Instrucciones: "
        "1) Clasifica la reclamación con classify_claim. "
        "2) Busca la cobertura aplicable en la póliza con policy_search. "
        "3) Determina cobertura, deducible y límite con check_coverage. "
        "4) Lista los documentos requeridos con required_docs. "
        "En tu respuesta final incluye obligatoriamente: "
        "primero el JSON completo de classify_claim, "
        "luego el JSON completo de check_coverage, "
        "y al final el resumen en texto."
    )

    submitted = st.button(
        "Procesar reclamación →",
        type="primary",
        disabled=not is_healthy,
        use_container_width=True,
    )

    if not is_healthy:
        st.warning("El API no está disponible. Corre `make serve` primero.")

with right_col:
    # Initialize session state for results
    if "result_raw" not in st.session_state:
        st.session_state.result_raw = ""
    if "result_scenario" not in st.session_state:
        st.session_state.result_scenario = ""

    # Process on submit
    if submitted:
        with st.spinner("Procesando reclamación con el agente de IA..."):
            try:
                raw = call_agent(user_prompt)
                st.session_state.result_raw = raw
                st.session_state.result_scenario = scenario_name
            except requests.exceptions.ConnectionError:
                st.error("No se pudo conectar al API. ¿Está corriendo `make serve`?")
                st.session_state.result_raw = ""
            except Exception as e:
                st.error(f"Error al procesar la reclamación: {e}")
                st.session_state.result_raw = ""

    raw = st.session_state.result_raw

    if not raw:
        st.markdown("### Resultado")
        st.info(
            "Selecciona un escenario y presiona **Procesar reclamación →** para ver "
            "el análisis del agente.",
            icon="🛡️",
        )
    else:
        classification, coverage = find_json_blocks(raw)

        # ── Classification card ──────────────────────────────────────────────
        st.markdown("### Clasificación")
        if classification:
            priority_icon = {
                "urgent": "🔴",
                "high": "🟠",
                "medium": "🟡",
                "low": "🟢",
            }.get(classification.get("priority", ""), "⚪")

            c1, c2 = st.columns(2)
            c1.metric(
                "Tipo de siniestro",
                classification.get("claim_type", "—").upper(),
            )
            c2.metric(
                "Prioridad",
                f"{priority_icon} {classification.get('priority', '—').title()}",
            )
            c3, c4 = st.columns(2)
            c3.metric(
                "Complejidad",
                classification.get("estimated_complexity", "—").title(),
            )
            c4.metric(
                "Manejador",
                classification.get("recommended_handler", "—")
                .replace("_", " ")
                .title(),
            )
            if classification.get("reasoning"):
                st.caption(f"_{classification['reasoning']}_")
        else:
            st.caption("Sin clasificación estructurada en la respuesta.")

        st.divider()

        # ── Coverage decision card ────────────────────────────────────────────
        st.markdown("### Decisión de Cobertura")
        if coverage:
            status = coverage.get("coverage_status", "")
            badge = {
                "fully_covered": "✅ Cubierto al 100%",
                "partially_covered": "⚠️ Cobertura Parcial",
                "not_covered": "❌ No Cubierto",
                "coverage_uncertain": "❓ Cobertura Incierta",
            }.get(status, f"• {status}")
            st.markdown(f"#### {badge}")

            d1, d2 = st.columns(2)
            d1.metric("Deducible", coverage.get("deductible", "N/A"))
            d2.metric("Límite de Cobertura", coverage.get("coverage_limit", "N/A"))

            if coverage.get("applicable_coverage"):
                st.markdown(
                    f"**Sección aplicable:** {coverage['applicable_coverage']}"
                )

            exclusions = coverage.get("exclusions_triggered", [])
            if exclusions:
                st.markdown("**Exclusiones:**")
                for excl in exclusions:
                    st.markdown(f"- {excl}")

            if coverage.get("explanation"):
                st.info(coverage["explanation"])

            if coverage.get("recommended_next_step"):
                st.success(
                    f"**Siguiente paso:** {coverage['recommended_next_step']}"
                )
        else:
            st.caption("Sin decisión de cobertura estructurada en la respuesta.")

# ── ReAct Trace (full width below) ───────────────────────────────────────────
if st.session_state.result_raw:
    st.divider()
    st.markdown("### Razonamiento del Agente (ReAct Trace)")
    st.caption(
        "Cada paso muestra cómo el agente razona y selecciona herramientas. "
        "Trazas completas disponibles en Arize Phoenix → http://localhost:6006"
    )

    steps = parse_react_trace(st.session_state.result_raw)

    STEP_CONFIG = {
        "thought": ("💭", "Razonamiento", "step-thought", True),
        "action": ("🔧", "Herramienta seleccionada", "step-action", True),
        "action_input": ("📥", "Entrada de la herramienta", "step-action", False),
        "observation": ("📤", "Resultado de la herramienta", "step-observation", False),
        "final_answer": ("🎯", "Respuesta Final", "step-answer", True),
    }

    if steps:
        for step in steps:
            icon, label, css_class, expanded = STEP_CONFIG.get(
                step["kind"], ("•", step["kind"], "", False)
            )
            with st.expander(f"{icon} **{label}**", expanded=expanded):
                parsed = try_parse_json(step["content"])
                if parsed:
                    st.json(parsed)
                else:
                    st.markdown(
                        f'<div class="{css_class}">{step["content"]}</div>',
                        unsafe_allow_html=True,
                    )
    else:
        # Fallback: raw output in a code block
        st.text_area("Respuesta del agente (raw)", st.session_state.result_raw, height=300)
