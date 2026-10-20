"""Fake OpenAI-compatible upstream for the live LiteLLM test.

Echoes the last user message back. Streams in 3-character slices so that
sentinels (``<PERSON_1>``) are split across chunks, and records the exact
request bodies it received at ``GET /_received`` so tests can assert that no
PII ever left the proxy.

Magic words in the user message:
  ``[slow]``  — 150 ms between chunks (to measure incremental delivery)
  ``[trunc]`` — stream ends in the middle of a possible sentinel prefix
Request with ``tools`` -> answers with a tool call whose JSON arguments embed
the user message.
"""
import asyncio
import json
import os
import sys
import time

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

app = FastAPI()
received: list[dict] = []


def _chunk(delta: dict, finish: str | None = None, **extra) -> str:
    body = {
        "id": "chatcmpl-fake",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "fake",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        **extra,
    }
    return f"data: {json.dumps(body)}\n\n"


@app.get("/_received")
async def get_received():
    return received


@app.post("/_reset")
async def reset():
    received.clear()
    return {}


@app.post("/v1/chat/completions")
async def chat(req: Request):
    body = await req.json()
    received.append(body)
    user = next(m["content"] for m in reversed(body["messages"]) if m["role"] == "user")
    if not isinstance(user, str):
        user = " ".join(p.get("text", "") for p in user)
    tools = body.get("tools")
    text = "Echo: " + user
    if "[trunc]" in user:
        text = "Echo: a < b and then <PERS"
    delay = 0.15 if "[slow]" in user else 0.0

    if not body.get("stream"):
        if tools:
            msg = {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1", "type": "function",
                    "function": {"name": tools[0]["function"]["name"],
                                 "arguments": json.dumps({"q": user})},
                }],
            }
            finish = "tool_calls"
        else:
            msg, finish = {"role": "assistant", "content": text}, "stop"
        return JSONResponse({
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": "fake",
            "choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        })

    async def gen():
        yield _chunk({"role": "assistant", "content": ""})
        if tools:
            args = json.dumps({"q": user})
            first = True
            for i in range(0, len(args), 3):
                tc = {"index": 0, "function": {"arguments": args[i:i + 3]}}
                if first:
                    tc.update(id="call_1", type="function")
                    tc["function"]["name"] = tools[0]["function"]["name"]
                    first = False
                yield _chunk({"tool_calls": [tc]})
                await asyncio.sleep(delay)
            yield _chunk({}, "tool_calls")
        else:
            for i in range(0, len(text), 3):
                yield _chunk({"content": text[i:i + 3]})
                await asyncio.sleep(delay)
            yield _chunk({}, "stop")
        if (body.get("stream_options") or {}).get("include_usage"):
            yield _chunk({}, None, usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}) \
                .replace('"choices": [{"index": 0, "delta": {}, "finish_reason": null}]', '"choices": []')
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host=os.getenv("HOST", "127.0.0.1"), port=int(sys.argv[1]) if len(sys.argv) > 1 else 9100,
                log_level="warning")
