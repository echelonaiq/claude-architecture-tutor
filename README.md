# Claude Architecture Tutor — Full-Stack Web App

A streaming chat app powered by the Anthropic API. Claude acts as an expert tutor on LLM internals: transformers, tokenization, RLHF, Constitutional AI, context windows, and more.

## Setup

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Set your Anthropic API key**
```bash
# macOS / Linux
export ANTHROPIC_API_KEY=sk-ant-...

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**3. Run the server**
```bash
uvicorn server:app --reload
```

**4. Open the app**

Navigate to [http://localhost:8000](http://localhost:8000) in your browser.

## Features

- **Streaming responses** — tokens appear as Claude generates them
- **Prompt caching** — system prompt is cached after the first turn (~90% cost reduction on subsequent turns)
- **Multi-turn conversation** — full message history sent with every request
- **5 quick-question buttons** — sidebar shortcuts for common architecture topics
- **Reset** — clears conversation history and starts fresh
- **Markdown rendering** — code blocks, lists, and bold text rendered correctly

## Stack

| Layer    | Technology                        |
|----------|-----------------------------------|
| Backend  | FastAPI + Anthropic Python SDK    |
| Streaming | Server-Sent Events (SSE) via `StreamingResponse` |
| Model    | `claude-opus-4-5`                 |
| Frontend | Vanilla HTML/CSS/JS + marked.js   |

## Project structure

```
building_with_claude_api/
├── server.py        # FastAPI backend
├── index.html       # Single-file frontend
├── requirements.txt
└── README.md
```
