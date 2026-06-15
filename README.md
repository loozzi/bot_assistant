# Telegram Second Brain Bot

A multi-functional Telegram bot with long-term memory, powered by specialized AI agents.

## Features

- **Journal Agent** — journaling with emotion analysis
- **Finance Agent** — expense tracking and behavioral pattern recognition
- **Search Agent** — web search with summarization
- **Insight Agent** — long-term personal trend analysis

## Tech Stack

- **Bot**: aiogram 3.x
- **Agent framework**: LangGraph
- **Vector DB**: Qdrant
- **Relational DB**: PostgreSQL 16
- **Cache / Session**: Redis 7
- **ORM**: SQLAlchemy 2.x async

## Quick Start

```bash
# 1. Copy and fill in environment variables
cp .env.example .env

# 2. Start infrastructure
docker compose up -d postgres redis qdrant

# 3. Seed the database
python scripts/seed_db.py

# 4. Start the bot
docker compose up app
```

## Project Structure

```
app/
├── core/          # Abstract interfaces (BaseAgent, BaseMemory, BaseTool)
├── modules/       # Feature modules (journal, finance, search, insight)
├── orchestrator/  # LangGraph routing graph
├── infra/         # Shared infrastructure (memory, DB, providers)
├── bot/           # Telegram handlers and middlewares
├── config/        # Settings and logging
└── utils/         # Pure utility functions
```

## Memory Layers

| Layer | Backend | Purpose |
|---|---|---|
| Working memory | Redis | Session context, short TTL |
| Episodic memory | Qdrant | Vector search, permanent |
| Semantic memory | PostgreSQL | Structured facts and relations |

## Environment Variables

See [.env.example](.env.example) for all required variables.
