# EDA Agent

A multi-agent Exploratory Data Analysis system built with LangGraph and Claude. Upload a CSV/Excel file, describe your goal, and a team of specialist agents profiles your data, runs statistical analysis, generates visualizations, and produces a written narrative — all in one automated pipeline.

## How it works

```
START → supervisor → [profiler | stat_analyst | viz_agent] → insight_critic → narrator → output_router → END
```

- **supervisor** — generates an analysis plan and waits for your approval (human-in-the-loop)
- **profiler** — schema, nulls, distributions, cardinality
- **stat_analyst** — correlations, outliers, normality tests, feature importance proxy
- **viz_agent** — Plotly chart specs (histograms, scatter, heatmap, boxplots)
- **insight_critic** — scores findings, flags weak conclusions, optionally re-dispatches an agent
- **narrator** — synthesises everything into a written narrative with key insights and caveats
- **output_router** — writes requested output files (report, JSON summary, dashboard, email draft)

## Quickstart

```bash
git clone <repo-url>
cd eda-agent

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Gradio UI (default)

```bash
python run.py
# Opens at http://localhost:7860
```

Upload a CSV or Excel file, type your analysis goal, review the generated plan, then type `approve` to run the full pipeline.

### CLI

```bash
python run.py --mode cli \
  --file data.csv \
  --goal "find key drivers of revenue" \
  --output report,json
```

The plan is printed to stdout. Type `approve` (or feedback) to continue.

## Output formats

| Flag | Output file |
|------|-------------|
| `report` | `output/<session_id>/report_<session_id>.html` |
| `json` | `output/<session_id>/summary_<session_id>.json` |
| `dashboard` | `output/<session_id>/dashboard_<session_id>.html` |
| `email` | `output/<session_id>/email_<session_id>.txt` |

## Running tests

```bash
pytest tests/ -v
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes | Your Anthropic API key |
| `DB_PATH` | No | SQLite checkpoint DB path (default: `./eda_agent.db`) |
| `GRADIO_PORT` | No | Port for Gradio UI (default: `7860`) |
| `SMTP_HOST` / `SMTP_USER` / `SMTP_PASSWORD` | No | Enable email sending in `email_drafter` |
