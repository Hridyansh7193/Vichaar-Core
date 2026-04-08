"""
Vichaar-Core — Multi-Agent Decision Intelligence Dashboard
Gradio UI connecting to the FastAPI backend on Hugging Face Spaces.
"""

import gradio as gr
import requests
import json

import os

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
BASE_URL = API_BASE_URL

# ─────────────────────────────────────────────
# Custom CSS — Linear / Stripe inspired
# ─────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Import Font ─────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Global Tokens ───────────────────────── */
:root {
    --bg: #f7f8fa;
    --card: #ffffff;
    --border: #e5e7eb;
    --border-hover: #d1d5db;
    --text-1: #111827;
    --text-2: #4b5563;
    --text-3: #9ca3af;
    --accent: #5046e5;
    --accent-hover: #4338ca;
    --accent-soft: #eef2ff;
    --green: #059669;
    --green-bg: #ecfdf5;
    --amber: #d97706;
    --amber-bg: #fffbeb;
    --red: #dc2626;
    --red-bg: #fef2f2;
    --r: 12px;
    --r-sm: 8px;
    --shadow-xs: 0 1px 2px rgba(0,0,0,0.03);
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.05), 0 1px 2px rgba(0,0,0,0.03);
    --shadow-md: 0 4px 6px -1px rgba(0,0,0,0.06), 0 2px 4px -2px rgba(0,0,0,0.04);
}

/* ── Force light everywhere ──────────────── */
.dark { color-scheme: light !important; }

body, .gradio-container, .dark, .dark .gradio-container {
    background: var(--bg) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    color: var(--text-1) !important;
}

.gradio-container {
    max-width: 880px !important;
    margin: 0 auto !important;
}

/* ── Nuke ALL default Gradio block chrome ── */
.gradio-container .block,
.gradio-container .form,
.gradio-container .gr-group,
.gradio-container .gr-box,
.gradio-container .gr-panel {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* ── Card panels ─────────────────────────── */
.card {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r) !important;
    padding: 28px 28px 24px !important;
    box-shadow: var(--shadow-xs) !important;
    transition: box-shadow 0.2s ease, border-color 0.2s ease !important;
}

.card:hover {
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Inner elements must be white ────────── */
.card .block,
.card .form,
.card .gr-group,
.card .gr-box,
.card .wrap,
.card .gr-padded,
.card > div,
.card > div > div,
.card .svelte-1mwvhlq,
.card [class*="group"],
.card [class*="form"],
.card [class*="wrap"],
.card [class*="panel"] {
    background: transparent !important;
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* Force Gradio CSS variables to white inside cards */
.card, .card * {
    --background-fill-secondary: transparent !important;
    --block-background-fill: transparent !important;
    --panel-background-fill: transparent !important;
    --input-background-fill: #ffffff !important;
}

/* ── Section labels ──────────────────────── */
.section-title {
    font-size: 11px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.8px !important;
    color: var(--text-3) !important;
    margin: 36px 0 10px 0 !important;
    padding: 0 !important;
}

.first-section {
    margin-top: 8px !important;
}

/* ── Header ──────────────────────────────── */
#header-area {
    text-align: center;
    padding: 44px 0 6px 0;
}
#header-area h1 {
    font-size: 26px !important;
    font-weight: 700 !important;
    color: var(--text-1) !important;
    letter-spacing: -0.4px;
    margin: 0 0 4px 0 !important;
}
#header-area p {
    font-size: 13.5px !important;
    color: var(--text-2) !important;
    margin: 0 !important;
    font-weight: 400;
}

/* ── Inputs ──────────────────────────────── */
.gradio-container textarea,
.gradio-container input[type="text"],
.gradio-container input[type="number"] {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    color: var(--text-1) !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
}
.gradio-container textarea:focus,
.gradio-container input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px var(--accent-soft) !important;
    outline: none !important;
}

/* ── Labels ──────────────────────────────── */
.gradio-container label span,
.gradio-container label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
}

/* ── Sliders ─────────────────────────────── */
input[type="range"] {
    accent-color: var(--accent) !important;
}

/* ── Buttons ─────────────────────────────── */
.btn-primary {
    background: var(--accent) !important;
    color: #fff !important;
    border: none !important;
    border-radius: var(--r-sm) !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 24px !important;
    box-shadow: 0 1px 3px rgba(80,70,229,0.22) !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    min-height: 42px !important;
}
.btn-primary:hover {
    background: var(--accent-hover) !important;
    box-shadow: 0 3px 10px rgba(80,70,229,0.28) !important;
    transform: translateY(-1px) !important;
}

.btn-secondary {
    background: var(--card) !important;
    color: var(--text-1) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    font-weight: 500 !important;
    font-size: 13.5px !important;
    padding: 10px 24px !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
    min-height: 42px !important;
}
.btn-secondary:hover {
    background: var(--bg) !important;
    border-color: var(--border-hover) !important;
    box-shadow: var(--shadow-xs) !important;
}

/* ── Result panel ────────────────────────── */
.result-panel {
    background: var(--bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--r-sm) !important;
    padding: 20px 24px !important;
}

/* ── Markdown inside results ─────────────── */
.gradio-container .prose {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-1) !important;
}
.gradio-container .prose h3 {
    font-size: 16px !important;
    font-weight: 600 !important;
    color: var(--text-1) !important;
    margin: 0 0 12px 0 !important;
    padding-bottom: 8px !important;
    border-bottom: 1px solid var(--border) !important;
}
.gradio-container .prose h4 {
    font-size: 12px !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
    color: var(--text-3) !important;
    margin: 20px 0 6px 0 !important;
}
.gradio-container .prose p,
.gradio-container .prose li {
    font-size: 13.5px !important;
    line-height: 1.65 !important;
    color: var(--text-1) !important;
}
.gradio-container .prose code {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
    padding: 2px 7px !important;
    border-radius: 5px !important;
    font-size: 12.5px !important;
    font-weight: 500 !important;
    font-family: 'JetBrains Mono', 'SF Mono', monospace !important;
}
.gradio-container .prose hr {
    border: none !important;
    border-top: 1px solid var(--border) !important;
    margin: 16px 0 !important;
}
.gradio-container .prose strong {
    font-weight: 600 !important;
    color: var(--text-1) !important;
}
.gradio-container .prose ul {
    padding-left: 0 !important;
    list-style: none !important;
}
.gradio-container .prose ul li {
    padding: 4px 0 !important;
}

/* ── Hide Gradio footer ──────────────────── */
footer { display: none !important; }

/* ── Notification area ───────────────────── */
.toast-wrap {
    font-family: 'Inter', sans-serif !important;
}

/* ── Connection badge ────────────────────── */
.connection-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--green-bg);
    color: var(--green);
    font-size: 11px;
    font-weight: 600;
    padding: 4px 10px;
    border-radius: 20px;
    margin-top: 10px;
}
.connection-badge::before {
    content: '';
    width: 6px;
    height: 6px;
    background: var(--green);
    border-radius: 50%;
}
"""

# ─────────────────────────────────────────────
# API Helpers
# ─────────────────────────────────────────────

def api_reset(scenario, profit, legal, env_imp, sentiment, cost):
    try:
        resp = requests.post(f"{BASE_URL}/reset", timeout=30)
        resp.raise_for_status()

        step_resp = requests.post(f"{BASE_URL}/step", json={"action": ""}, timeout=60)
        step_resp.raise_for_status()
        data = step_resp.json()

        return _format_step_result(data, header="Scenario Initialized")

    except requests.exceptions.Timeout:
        return _error_md("Request timed out. The backend may be waking up — try again in ~30s.")
    except requests.exceptions.ConnectionError:
        return _error_md("Could not connect to the backend.")
    except Exception as e:
        return _error_md(str(e))


def api_step():
    try:
        resp = requests.post(f"{BASE_URL}/step", json={"action": ""}, timeout=60)
        resp.raise_for_status()
        return _format_step_result(resp.json(), header="Step Complete")
    except requests.exceptions.Timeout:
        return _error_md("Step timed out — the agents may need more time.")
    except requests.exceptions.ConnectionError:
        return _error_md("Connection lost — backend may have restarted.")
    except Exception as e:
        return _error_md(str(e))


def api_state():
    try:
        resp = requests.get(f"{BASE_URL}/state", timeout=15)
        resp.raise_for_status()
        return _format_state_result(resp.json())
    except requests.exceptions.Timeout:
        return _error_md("State request timed out.")
    except requests.exceptions.ConnectionError:
        return _error_md("Could not reach the backend.")
    except Exception as e:
        return _error_md(str(e))


# ─────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────

def _metric_line(label, value, warn_high=False):
    pct = round(value * 100, 1)
    if warn_high:
        ind = "🔴" if pct >= 70 else ("🟡" if pct >= 45 else "🟢")
    else:
        ind = "🟢" if pct >= 60 else ("🟡" if pct >= 35 else "🔴")
    return f"{ind}  **{label}** — `{pct}%`"


def _format_metrics(metrics):
    if not metrics:
        return "*No metrics available*"
    warn = {"legal_risk", "env_impact", "cost"}
    lines = []
    for k, v in metrics.items():
        lines.append(_metric_line(k.replace("_", " ").title(), float(v), k in warn))
    return "\n\n".join(lines)


def _format_messages(messages):
    if not messages:
        return "*No agent messages yet*"
    icons = {"PROFIT": "📊", "ETHICS": "⚖️", "PR": "📢", "LEGAL": "🛡️", "RISK": "⚠️", "CEO": "👔"}
    out = []
    for m in messages:
        m = str(m).strip()
        if not m:
            continue
        if m.startswith("[") and "]" in m:
            i = m.index("]")
            agent = m[1:i].strip().upper()
            content = m[i+1:].strip()
            icon = icons.get(agent, "💬")
            out.append(f"- {icon} **{agent}** — {content}")
        else:
            out.append(f"- 💬 {m}")
    return "\n".join(out) if out else "*No agent messages yet*"


def _format_events(events):
    if not events:
        return "*No events triggered*"
    labels = {
        "regulatory_crisis": "⚠️ Regulatory Crisis",
        "market_opportunity": "📈 Market Opportunity",
        "competitor_move": "🏁 Competitor Move",
        "media_scandal": "📰 Media Scandal",
        "supply_disruption": "🔗 Supply Disruption",
    }
    return "\n".join(f"- {labels.get(e, e.replace('_',' ').title())}" for e in events)


def _format_history(history):
    if not history:
        return "*No actions taken yet*"
    parts = [f"`{i}.` {a.replace('_',' ').title()}" for i, a in enumerate(history, 1)]
    return "  →  ".join(parts)


def _format_step_result(data, header="Result"):
    obs = data.get("observation", {})
    reward = data.get("reward", 0)
    done = data.get("done", False)
    info = data.get("info", {})

    scenario = obs.get("scenario", "—")
    phase = obs.get("phase", "—").title()
    step_n = obs.get("step_count", 0)
    metrics = obs.get("metrics", {})
    messages = obs.get("agent_messages", [])
    events = info.get("events", obs.get("events", []))
    history = obs.get("history", [])
    action = history[-1].replace("_", " ").title() if history else "—"
    status = "✅ Episode Complete" if done else "▶️ Running"
    r_pct = round(float(reward) * 100, 1)
    r_ind = "🟢" if r_pct >= 60 else ("🟡" if r_pct >= 30 else "🔴")

    return f"""### {header}

---

#### Scenario
{scenario}

#### Status
**Phase** `{phase}` · **Step** `{step_n}` · {status}

---

#### Selected Action
`{action}`

#### Reward
{r_ind} **{r_pct}%**

---

#### Metrics

{_format_metrics(metrics)}

---

#### Agent Deliberation

{_format_messages(messages)}

---

#### Events

{_format_events(events)}

---

#### Action History

{_format_history(history)}""".strip()


def _format_state_result(data):
    scenario = data.get("scenario", "—")
    phase = data.get("phase", "—").title()
    step_n = data.get("step_count", 0)
    metrics = data.get("metrics", {})
    messages = data.get("agent_messages", [])
    events = data.get("events", [])
    history = data.get("history", [])

    return f"""### Current State

---

#### Scenario
{scenario}

#### Status
**Phase** `{phase}` · **Step** `{step_n}`

---

#### Metrics

{_format_metrics(metrics)}

---

#### Agent Messages

{_format_messages(messages)}

---

#### Events

{_format_events(events)}

---

#### Action History

{_format_history(history)}""".strip()


def _error_md(msg):
    return f"""### Error

---

⚠️ {msg}

*Check that the backend is reachable at `{BASE_URL}`*"""


# ─────────────────────────────────────────────
# Build UI
# ─────────────────────────────────────────────

with gr.Blocks(
    css=CUSTOM_CSS,
    title="Vichaar-Core — Decision Intelligence",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
        font_mono=gr.themes.GoogleFont("JetBrains Mono"),
    ),
) as demo:

    # Header
    gr.HTML("""
        <div id="header-area">
            <h1>Vichaar-Core</h1>
            <p>Multi-Agent Decision Intelligence Dashboard</p>
            <div class="connection-badge">Connected to HF Spaces</div>
        </div>
    """)

    # ── Scenario Setup ───────────────────────
    gr.HTML('<p class="section-title first-section">Scenario Setup</p>')

    with gr.Group(elem_classes="card"):
        scenario_input = gr.Textbox(
            label="Scenario Description",
            placeholder="Describe a business scenario for the agents to evaluate…",
            lines=2,
            max_lines=3,
            value="A tech company deciding whether to launch a new AI product with potential privacy concerns.",
        )

        gr.HTML('<div style="height:4px"></div>')

        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                sl_profit = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Expected Profit")
            with gr.Column(scale=1, min_width=180):
                sl_legal = gr.Slider(0.0, 1.0, value=0.1, step=0.05, label="Legal Risk")

        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                sl_env = gr.Slider(0.0, 1.0, value=0.05, step=0.05, label="Environmental Impact")
            with gr.Column(scale=1, min_width=180):
                sl_sentiment = gr.Slider(0.0, 1.0, value=0.5, step=0.05, label="Public Sentiment")

        with gr.Row():
            with gr.Column(scale=1, min_width=180):
                sl_cost = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Cost")
            with gr.Column(scale=1, min_width=180):
                pass  # empty for balance

        gr.HTML('<div style="height:8px"></div>')

        btn_init = gr.Button("Initialize Scenario", elem_classes="btn-primary", size="lg")

    # ── Simulation Controls ──────────────────
    gr.HTML('<p class="section-title">Simulation Controls</p>')

    with gr.Group(elem_classes="card"):
        with gr.Row():
            btn_step = gr.Button("Run Step", elem_classes="btn-primary", size="lg", scale=2)
            btn_state = gr.Button("Get Current State", elem_classes="btn-secondary", size="lg", scale=1)

    # ── Results ──────────────────────────────
    gr.HTML('<p class="section-title">Results</p>')

    with gr.Group(elem_classes="card"):
        result_output = gr.Markdown(
            value="*Initialize a scenario to begin the simulation.*",
            elem_classes="result-panel",
        )

    # ── Wiring ───────────────────────────────
    btn_init.click(fn=api_reset, inputs=[scenario_input, sl_profit, sl_legal, sl_env, sl_sentiment, sl_cost], outputs=result_output)
    btn_step.click(fn=api_step, inputs=[], outputs=result_output)
    btn_state.click(fn=api_state, inputs=[], outputs=result_output)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
