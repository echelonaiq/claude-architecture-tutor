from dotenv import load_dotenv
load_dotenv()
import json
import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Claude Architecture Tutor")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

SYSTEM_PROMPT = """\
You are an expert Claude Architecture Tutor. Your mission is to help students
deeply understand how large language models like Claude are designed and trained.

Topics you teach:
- Transformer architecture: multi-head self-attention, feed-forward layers,
  residual connections, layer normalisation, positional encodings
- Tokenization: BPE / SentencePiece, vocabulary size, why "strawberry" = 3 tokens
- Pre-training (next-token prediction), fine-tuning, and RLHF
- Constitutional AI (CAI): supervised learning from AI feedback (SL-CAI) +
  reinforcement learning from AI feedback (RL-CAI)
- Context windows: KV cache mechanics, sliding-window attention, long-context
  trade-offs between memory, compute, and attention quality
- Claude model families: how Haiku, Sonnet, and Opus differ in capability,
  speed, context window, and cost — and when to choose each
- Safety techniques: harmlessness training, red-teaming, refusal calibration,
  honesty / calibration objectives
- Prompt engineering: system prompts, few-shot examples, chain-of-thought,
  prompt caching, structured outputs

Teaching style:
- Lead with an intuitive analogy before diving into technical detail
- Use concrete examples and short code snippets when they clarify a point
- After each explanation, pose a short check-for-understanding question
- Keep answers focused — one concept at a time, no unnecessary padding
"""

client = anthropic.AsyncAnthropic()


class ChatRequest(BaseModel):
    messages: list[dict]
    user_input: str


async def sse_stream(messages: list[dict], user_input: str):
    """Async generator that yields SSE-formatted tokens from the Anthropic stream."""
    all_messages = messages + [{"role": "user", "content": user_input}]

    try:
        async with client.messages.stream(
            model="claude-opus-4-5",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": SYSTEM_PROMPT,
                    # Cache the system prompt — saves ~90 % on turns 2+ once warm
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=all_messages,
        ) as stream:
            async for text in stream.text_stream:
                payload = json.dumps({"type": "token", "text": text})
                yield f"data: {payload}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    except anthropic.APIStatusError as exc:
        error_payload = json.dumps({"type": "error", "message": exc.message})
        yield f"data: {error_payload}\n\n"
    except Exception as exc:
        error_payload = json.dumps({"type": "error", "message": str(exc)})
        yield f"data: {error_payload}\n\n"


@app.post("/chat")
async def chat(req: ChatRequest):
    return StreamingResponse(
        sse_stream(req.messages, req.user_input),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # disable Nginx proxy buffering
        },
    )


@app.get("/")
async def root():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path)
