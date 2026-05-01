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

QUESTION_GEN_SYSTEM = """\
You are an exam question generator for the Claude Certified Architect \
Foundations (CCA-F) exam. Generate hard, scenario-based questions that \
mirror the real exam. The exam has 5 domains:
1. Agentic Architecture & Orchestration (27%) - agentic loops, \
   multi-agent coordinator-subagent patterns, task decomposition, \
   session state management
2. Claude Code Configuration & Workflows (20%) - CLAUDE.md files, \
   Agent Skills, MCP server integrations, plan mode
3. Prompt Engineering & Structured Output (20%) - few-shot techniques, \
   JSON schema enforcement, validation loops, explicit criteria design
4. Tool Design & MCP Integration (18%) - clean tool interfaces, \
   structured error handling, MCP server distribution
5. Context Management & Reliability (15%) - long context preservation, \
   escalation logic, error propagation in pipelines

Questions must be grounded in one of these 6 real production scenarios:
- Customer support resolution agent
- Multi-agent research pipeline with coverage gaps
- CI/CD automated code review system
- Developer productivity assistant
- Structured data extraction from documents
- Multi-turn conversational system

Return ONLY a JSON array of exactly 5 strings. Each string is one \
question. No numbering, no labels, no extra text. Hard difficulty only.\
"""

FALLBACK_QUESTIONS = [
    "A customer support agent built on Claude intermittently fails to escalate high-priority tickets. The orchestration loop uses a single claude-haiku-4-5 call with a 2,000-token context window. What architectural changes would you make to reliably detect and escalate urgent cases without reprocessing entire conversation histories?",
    "Your CLAUDE.md file defines a code review skill, but agents in a CI/CD pipeline ignore it and generate free-form comments instead. What is the most likely root cause, and how would you enforce structured output conformance across all pipeline agents?",
    "A multi-agent research pipeline has three subagents fetching data from different sources. Sometimes the final synthesizer agent receives incomplete results because one subagent silently times out. How would you redesign error propagation so the orchestrator can detect partial failures and decide whether to retry, skip, or abort?",
    "You need to extract structured invoice data (vendor, line items, totals) from scanned PDFs using Claude. Extraction accuracy is 94% but the remaining 6% produce malformed JSON that breaks downstream systems. Design a validation loop using JSON schema enforcement and describe your fallback strategy.",
    "A developer productivity assistant maintains context across a 40-turn coding session. By turn 30, response quality degrades because the context window is saturated with low-value tool outputs. What context management strategy would you implement, and how would you decide what to summarize versus preserve verbatim?",
]

EVALUATOR_SYSTEM = """\
You are an expert AI response quality evaluator. Given a system prompt, a user \
input, the model's response, and a list of evaluation criteria, score each \
criterion from 1–10 and provide a brief reason.

Return ONLY valid JSON — no markdown fences, no preamble. Exactly this shape:
{
  "scores": {
    "<criterion>": {"score": <1-10>, "reason": "<one sentence>"}
  },
  "improvement_suggestion": "<one concrete suggestion>"
}
"""

client = anthropic.AsyncAnthropic()


class ChatRequest(BaseModel):
    messages: list[dict]
    user_input: str


class EvaluateRequest(BaseModel):
    prompt: str
    test_input: str
    criteria: list[str]


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


@app.get("/questions")
async def get_questions():
    try:
        response = await client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            system=QUESTION_GEN_SYSTEM,
            messages=[{"role": "user", "content": "Generate 5 exam questions now."}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        questions = json.loads(text)
        if (
            isinstance(questions, list)
            and len(questions) == 5
            and all(isinstance(q, str) for q in questions)
        ):
            return {"questions": questions}
    except Exception:
        pass
    return {"questions": FALLBACK_QUESTIONS}


@app.post("/evaluate")
async def evaluate(req: EvaluateRequest):
    # Call 1: run the prompt against the test input
    run_resp = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=req.prompt,
        messages=[{"role": "user", "content": req.test_input}],
    )
    response_text = run_resp.content[0].text.strip()

    # Call 2: evaluate the response against criteria (cached evaluator system)
    criteria_list = "\n".join(f"- {c}" for c in req.criteria)
    eval_user_msg = (
        f"System prompt:\n{req.prompt}\n\n"
        f"User input:\n{req.test_input}\n\n"
        f"Model response:\n{response_text}\n\n"
        f"Criteria to score:\n{criteria_list}"
    )
    eval_resp = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": EVALUATOR_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": eval_user_msg}],
    )

    raw = eval_resp.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=502, detail="Evaluator returned invalid JSON")

    scores: dict = parsed.get("scores", {})
    score_values = [v["score"] for v in scores.values() if isinstance(v.get("score"), (int, float))]
    overall = round(sum(score_values) / len(score_values), 1) if score_values else 0

    return {
        "response": response_text,
        "scores": scores,
        "overall_score": overall,
        "improvement_suggestion": parsed.get("improvement_suggestion", ""),
    }


@app.get("/eval")
async def eval_page():
    html_path = BASE_DIR / "eval.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="eval.html not found")
    return FileResponse(html_path)


@app.get("/")
async def root():
    html_path = BASE_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="index.html not found")
    return FileResponse(html_path)
