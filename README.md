# SupportPilot API

AI-powered customer support automation — rebuilt in Python from my n8n SupportPilot workflow.

## What it does (so far)
- Classifies customer emails into category, urgency, and whether a human is needed
- Validates AI output with Pydantic so bad data never reaches downstream systems
- Handles cut-off replies and invalid AI output safely

## Tech
Python · OpenAI API · Pydantic

## Run it
1. `pip install -r requirements.txt`
2. Create `.env` with `OPENAI_API_KEY=...` and `MODEL=...`
3. `python classify.py`

## Roadmap
- [x] AI email classifier with validation
- [ ] FastAPI endpoint called from n8n
- [ ] RAG over help-centre docs (Qdrant)
- [ ] LangGraph agent with human handoff
- [ ] Docker + CI
- [ ] Cloud deployment