"""
Batwan's predictive text on a Hugging Face Space: a search box, and its API.

    GET /api/suggest?q=maayong%20a&lang=any     what to suggest for what is typed
    GET /health                                 the web service is up
    GET /health/model                           and the model: up, loading or down
    GET /                                       a search box to try it in

llama-server (start.sh) serves the GGUF on 127.0.0.1:8080; predict.py asks
it what comes next. This file is what a type-ahead needs around that on two
CPU cores:

- **Only the latest keystroke counts.** A visitor's request that is still
  waiting for the model when a newer one arrives is answered at once, empty
  (`"stale": true`), rather than queued: by then nobody wants it.
- **A cache.** "maa" is typed often; it is answered once.
- **A rate limit**, per visitor, generous for typing (PREDICT_RATE_LIMIT).
- **Nothing typed is logged or kept**, except in the cache, in memory,
  under the text itself, for at most PREDICT_CACHE_SECONDS.

The answer is shaped like batwan.org's own `/api/suggest`, with
`"type": "predict"` items, so its search box can put the two side by side.
"""

import asyncio
import collections
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import predict

HERE = Path(__file__).resolve().parent
LLM_URL = os.getenv("PREDICT_LLM_URL", "http://127.0.0.1:8080")
MODEL_NAME = os.getenv("MODEL_SOURCE", "")
RATE_LIMIT = int(os.getenv("PREDICT_RATE_LIMIT", "120"))
CACHE_SIZE = int(os.getenv("PREDICT_CACHE_SIZE", "4096"))
CACHE_SECONDS = int(os.getenv("PREDICT_CACHE_SECONDS", "21600"))
CONCURRENCY = int(os.getenv("PREDICT_CONCURRENCY", "2"))
TIMEOUT = float(os.getenv("PREDICT_TIMEOUT", "30"))
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "PREDICT_ALLOWED_ORIGINS",
    "https://batwan.org,https://www.batwan.org,https://batwan-app.alser.ai").split(",") if o.strip()]
FRAME_ANCESTORS = os.getenv("SPACE_FRAME_ANCESTORS", "https://huggingface.co")

SETTINGS = predict.Settings(
    beams=int(os.getenv("PREDICT_BEAMS", "3")),
    candidates=int(os.getenv("PREDICT_CANDIDATES", "16")),
    branch=int(os.getenv("PREDICT_BRANCH", "3")),
    max_tokens=int(os.getenv("PREDICT_MAX_TOKENS", "4")),
    end_probs=int(os.getenv("PREDICT_END_PROBS", "12")),
    words=int(os.getenv("PREDICT_WORDS", "0")),
    limit=int(os.getenv("PREDICT_LIMIT", "3")),
    min_share=float(os.getenv("PREDICT_MIN_SHARE", "0.02")),
)

STARTED = time.time()


class State:
    client: httpx.AsyncClient = None
    predictor: predict.Predictor = None
    gate: asyncio.Semaphore = None
    model: str = "loading"
    checked: float = 0.0

    def __init__(self):
        self.cache = collections.OrderedDict()
        self.hits = collections.defaultdict(collections.deque)
        self.latest = {}


state = State()


@asynccontextmanager
async def lifespan(_app):
    state.client = httpx.AsyncClient(timeout=TIMEOUT)
    fmt = predict.load_format(os.getenv("PREDICT_FORMAT", ""))
    state.predictor = predict.Predictor(state.client, LLM_URL, fmt, SETTINGS)
    state.gate = asyncio.Semaphore(CONCURRENCY)
    yield
    await state.client.aclose()


app = FastAPI(title="Batwan predictive text", docs_url="/docs", redoc_url=None, lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET"],
                   allow_headers=["*"], max_age=3600)


@app.middleware("http")
async def headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = f"frame-ancestors 'self' {FRAME_ANCESTORS}"
    return response


async def model_status() -> str:
    """llama-server's /health, asked at most every two seconds."""
    if time.time() - state.checked < 2:
        return state.model
    try:
        response = await state.client.get(LLM_URL + "/health", timeout=2)
        state.model = "up" if response.status_code == 200 else "loading"
    except httpx.HTTPError:
        state.model = "loading" if time.time() - STARTED < 900 else "down"
    state.checked = time.time()
    return state.model


def visitor(request: Request) -> str:
    # Behind the Space's proxy (uvicorn --proxy-headers), the visitor's address.
    return request.client.host if request.client else "?"


def allowed(who: str) -> bool:
    now, window = time.time(), state.hits[who]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= RATE_LIMIT:
        return False
    window.append(now)
    if len(state.hits) > 10000:
        for key in [k for k, v in state.hits.items() if not v][:5000]:
            del state.hits[key]
    return True


@app.get("/api/suggest")
async def suggest(request: Request,
                  q: str = Query(..., min_length=1, max_length=100),
                  lang: str = Query("any", pattern="^(any|hil|en)$"),
                  limit: int = Query(0, ge=0, le=8),
                  words: int = Query(-1, ge=-1, le=3)):
    started = time.time()
    text = predict.normalise(q)
    limit = limit or SETTINGS.limit
    words = SETTINGS.words if words < 0 else words
    base = {"query": text, "lang": lang, "model": MODEL_NAME}
    who = visitor(request)
    if not allowed(who):
        return JSONResponse({**base, "items": [], "error": "Too many requests"}, status_code=429)
    if not text.strip():
        return {**base, "items": [], "ms": 0}

    key = (lang, text, limit, words)
    hit = state.cache.get(key)
    if hit and time.time() - hit[0] < CACHE_SECONDS:
        state.cache.move_to_end(key)
        return {**base, "items": hit[1], "ms": int((time.time() - started) * 1000), "cached": True}

    if await model_status() != "up":
        return {**base, "items": [], "ms": 0, "status": state.model}

    ticket = object()
    state.latest[who] = ticket
    async with state.gate:
        if state.latest.get(who) is not ticket:
            return {**base, "items": [], "ms": int((time.time() - started) * 1000), "stale": True}
        try:
            result = await state.predictor.suggest(text, lang, limit, words)
        except (httpx.HTTPError, ValueError, KeyError) as problem:
            state.checked = 0
            return JSONResponse({**base, "items": [], "error": type(problem).__name__},
                                status_code=503)
        finally:
            if state.latest.get(who) is ticket:
                del state.latest[who]
    items = result["items"]
    state.cache[key] = (time.time(), items)
    while len(state.cache) > CACHE_SIZE:
        state.cache.popitem(last=False)
    ms = int((time.time() - started) * 1000)
    print(f"suggest lang={lang} words={len(text.split())} items={len(items)} ms={ms}", flush=True)
    return {**base, "items": items, "ms": ms}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/model")
async def health_model():
    return {"model": await model_status(), "name": MODEL_NAME}


app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")
