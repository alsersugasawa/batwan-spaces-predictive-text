"""
What to suggest for what has been typed into batwan.org's search box.

The model is batwan-gemma-4-e2b-predictive-text, a LoRA of
google/gemma-4-E2B-it trained on search queries written as

    <bos>Batwan search: maayong aga<eos>

(the prefix for each direction is in predictive_text.json, which comes from
the model repository, so this and the training agree by construction). It
suggests as a phone keyboard does: the rest of the word being typed, or,
after a space, the next word. For "maayong a" it is asked how
"Batwan search: maayong" goes on, in two rounds through llama-server:

1. **Which word.** The prompt stops before the word being typed, and a
   grammar says the continuation must start with " a". This is token
   healing: the word is not cut where the typing stopped, so the model sees
   it as it saw every word in training, whole and with its space. One step
   gives the likeliest first tokens that agree with the typing - " aga",
   " adlaw", " a", " " - and their probabilities among those that do.
2. **Branch.** Words that share a first token would otherwise share a
   beam: Gemma writes both "maayong" and "maayo" as " ma" and more. The
   `branch` likeliest paths are taken one token further, all their
   continuations at once; one that starts a new word, or ends the query,
   says the word ended there.
3. **The rest of the word.** The best `beams` open paths are carried on
   greedily, together (one request, several prompts, as many slots), until
   the word ends. A path that covers only part of the typing (" ma" for
   "maa") is held to the rest of it by a grammar.

A suggestion is ranked by the probability of its word's tokens. One path
of tokens counts once however it ended; two tokenisations of one word add
up. `words` (0 by default) lets a suggestion run on for that many more
words when the model ends the query with them.

Why so few tokens: Gemma 4 E2B writes 6 to 9 tokens a second on two CPU
cores. A suggestion is a step, a step for each branch, and one to four for
each beam.
"""

import asyncio
import json
import math
import re
import unicodedata
from dataclasses import dataclass

DEFAULT_FORMAT = {
    "format": "batwan-predictive-text/1",
    "prefix": {
        "any": "Batwan search:",
        "hil": "Batwan search (Hiligaynon):",
        "en": "Batwan search (English):",
    },
    "lowercase": True,
    "end_of_query": "<eos>",
}
LANGS = ("any", "hil", "en")
MAX_QUERY = 100
# What a word may continue with: what the model was trained on.
WORD_CHARS = "a-zñáéíóúàèìòùâêîôûäëïöüç'-"
WORD_RUN = re.compile(f"[{WORD_CHARS}]*")
# A suggestion never ends on one of these (the training data trims them
# from every phrase): "maayong aga sa" is not a query.
LINKERS = frozenset("ang nga sa kag sang mga ka si sanday ni kay sing ukon kon para pero may "
                    "kundi agud bangud tungod samtang apang kun o ug the a an of to in on at "
                    "for with by from and or but is are was were be as".split())


def load_format(path: str = "") -> dict:
    """The model's own format file, or the default it was trained with."""
    if path:
        try:
            with open(path, encoding="utf-8") as handle:
                fmt = json.load(handle)
            if set(fmt.get("prefix", {})) == set(LANGS):
                return fmt
        except (OSError, ValueError):
            pass
    return DEFAULT_FORMAT


def normalise(typed: str) -> str:
    """As the training data was written: lower case, single spaces, straight quotes."""
    text = unicodedata.normalize("NFC", typed or "").lower()
    text = text.replace("’", "'").replace("‘", "'").replace("ʼ", "'")
    text = re.sub(r"[\"“”«»?!.,;:()\[\]{}]", " ", text)
    trailing = text[-1:].isspace()
    text = " ".join(text.split())
    if trailing and text:
        text += " "
    return text[:MAX_QUERY]


def split_typed(text: str) -> tuple:
    """
    (context, fragment): the words before the one being typed, and that one
    with its leading space - every word in training starts with a space.

        "maayong a"  -> (" maayong", " a")
        "maa"        -> ("", " maa")
        "maayong "   -> (" maayong", " ")    the next word, whatever it is
    """
    if text.endswith(" "):
        return " " + text.rstrip(), " "
    if " " in text:
        head, last = text.rsplit(" ", 1)
        return " " + head, " " + last
    return "", " " + text


def gbnf_literal(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def grammar(rest: str, words: int, word_started: bool) -> str:
    """
    `rest` (what of the typed word the tokens so far have not covered), the
    rest of that word, then up to `words` more. A word not started yet - the
    typing ended in a space - must have a letter.
    """
    word = f"[{WORD_CHARS}]"
    first = f"{word}*" if word_started else f"{word}+"
    more = f' (" " {word}+){{0,{words}}}' if words else ""
    lead = f"{gbnf_literal(rest)} " if rest else ""
    return f"root ::= {lead}{first}{more}"


@dataclass
class Settings:
    beams: int = 3        # open paths carried on to the end of their word
    candidates: int = 16  # tokens asked for at each branching step
    branch: int = 2       # paths taken one token further before the beams are chosen
    max_tokens: int = 4   # per beam in round 2, before the word must have ended
    words: int = 0        # more words a suggestion may run on for
    limit: int = 3


@dataclass
class _Path:
    tokens: list          # token ids after the prompt
    text: str             # what they say, from the leading space
    logprob: float        # of the word's tokens
    ended: str = ""       # "" open; "eos" the query ended; "word" another word began


@dataclass
class _Found:
    text: str
    logprob: float
    complete: bool


class Predictor:
    """Suggestions from a llama-server, through an httpx.AsyncClient (or anything with `post`)."""

    def __init__(self, client, base_url: str, fmt: dict = None, settings: Settings = None):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.fmt = fmt or DEFAULT_FORMAT
        self.settings = settings or Settings()

    async def _post(self, path: str, body: dict):
        response = await self.client.post(self.base_url + path, json=body)
        response.raise_for_status()
        return response.json()

    async def _step(self, ids: list, paths: list, fragment: str, words: int) -> list:
        """
        For each path, the tokens that may come next and their probabilities
        among those the typing allows: one token each, one request per grammar.
        """
        s = self.settings
        groups = {}
        for i, path in enumerate(paths):
            groups.setdefault(_grammar_after(path.text, fragment, words, step=True), []).append(i)
        out = [[] for _ in paths]

        async def run(gram, members):
            body = {"prompt": [ids + paths[i].tokens for i in members], "n_predict": 1,
                    "n_probs": s.candidates, "post_sampling_probs": True, "grammar": gram,
                    "samplers": ["top_k", "temperature"], "top_k": s.candidates,
                    "temperature": 1.0, "cache_prompt": True}
            result = await self._post("/completion", body)
            for i, r in zip(members, result if isinstance(result, list) else [result]):
                out[i] = (r.get("completion_probabilities") or [{}])[0].get("top_probs") or []
        await asyncio.gather(*(run(g, m) for g, m in groups.items()))
        return out

    async def suggest(self, typed: str, lang: str = "any", limit: int = 0,
                      words: int = None) -> dict:
        s = self.settings
        limit = max(1, min(limit or s.limit, 8))
        words = s.words if words is None else max(0, min(words, 3))
        text = normalise(typed)
        if not text.strip():
            return {"query": text, "items": []}
        lang = lang if lang in LANGS else "any"
        context, fragment = split_typed(text)
        ids = (await self._post("/tokenize", {
            "content": self.fmt["prefix"][lang] + context, "add_special": True}))["tokens"]
        root = _Path([], "", 0.0)

        # 1. The first tokens that agree with the typing.
        paths = []
        for cand in (await self._step(ids, [root], fragment, 0))[0]:
            if cand.get("prob", 0) > 0 and _agrees(cand.get("token", ""), fragment):
                paths.append(_Path([cand["id"]], cand["token"], math.log(cand["prob"])))
        paths = paths[:s.candidates]
        if not paths:
            return {"query": text, "items": []}

        # 2. The likeliest paths one token further: a word that shares its
        # first token with a likelier one gets a beam of its own.
        if s.branch:
            parents, others = paths[:s.branch], paths[s.branch:]
            children = []
            for parent, nexts in zip(parents, await self._step(ids, parents, fragment, 1)):
                # The parent's probability is shared between its word ending
                # here - the query ends, or another word begins - and each
                # way of going on with it.
                ends, eos = 0.0, 0.0
                for cand in nexts:
                    if cand.get("prob", 0) <= 0:
                        continue
                    token = cand.get("token", "")
                    if token == "" or (token.startswith(" ") and len(parent.text) >= len(fragment)):
                        ends += cand["prob"]
                        eos += cand["prob"] if token == "" else 0.0
                    elif not token.startswith(" "):
                        children.append(_Path(parent.tokens + [cand["id"]], parent.text + token,
                                              parent.logprob + math.log(cand["prob"])))
                if ends > 0:
                    children.append(_Path(parent.tokens, parent.text,
                                          parent.logprob + math.log(ends),
                                          "eos" if eos >= ends / 2 else "word"))
            paths = children + others
        paths = _dedupe(paths)
        paths.sort(key=lambda p: -p.logprob)

        # 3. Each open path to the end of its word, greedily. A path that has
        # covered the typing needs no grammar: a stop at the next space ends
        # the word, and the model was taught to write words only.
        open_paths = [p for p in paths if not p.ended][:max(s.beams, limit)]
        groups = {}
        for i, path in enumerate(open_paths):
            groups.setdefault(_grammar_after(path.text, fragment, words, step=False), []).append(i)
        results = [None] * len(open_paths)

        async def run(gram, members):
            body = {"prompt": [ids + open_paths[i].tokens for i in members],
                    "n_predict": s.max_tokens + 3 * words, "n_probs": 1,
                    "post_sampling_probs": False, "temperature": 0.0, "cache_prompt": True}
            if gram:
                body["grammar"] = gram
            if not words:
                body["stop"] = [" "]
            out = await self._post("/completion", body)
            for i, r in zip(members, out if isinstance(out, list) else [out]):
                results[i] = r
        await asyncio.gather(*(run(g, m) for g, m in groups.items()))

        found = {}

        def add(suggestion, logprob, complete, key):
            # One path of tokens counts once, however it ended; two
            # tokenisations of the same word add up.
            entry = found.setdefault(suggestion, {"paths": {}, "complete": False})
            entry["paths"][key] = max(entry["paths"].get(key, -math.inf), logprob)
            entry["complete"] |= complete

        for path in paths:
            if path.ended:
                word = path.text.strip()
                if word and WORD_RUN.fullmatch(word) and path.text.startswith(fragment):
                    add(" ".join(x for x in (context.strip(), word) if x), path.logprob,
                        path.ended == "eos", tuple(path.tokens))
        for path, r in zip(open_paths, results):
            if not r:
                continue
            whole = path.text + r.get("content", "")
            if not whole.startswith(fragment):
                continue
            # The word: what was typed, and the letters after it - up to the
            # first character no query has (the stock model, never taught,
            # writes punctuation and capitals).
            end = len(fragment) + len(WORD_RUN.match(whole, len(fragment)).group(0))
            word = whole[:end].strip()
            stop, after = r.get("stop_type"), whole[end:]
            # It ended if the model ended the query, began another word, or
            # wrote something that is not a letter. Out of tokens mid-word,
            # it did not: half a word is not a suggestion.
            if not word or not word[0].isalpha() or (stop not in ("eos", "word") and not after):
                continue
            steps = r.get("completion_probabilities") or []
            lp, seen, used = path.logprob, len(path.text), []
            for step in steps:
                if seen >= end:
                    break
                lp += step.get("logprob", 0.0)
                seen += len(step.get("token", ""))
                used.append(step.get("id", step.get("token")))
            extra = []
            if words and stop == "eos":
                extra = [w for w in after.split() if WORD_RUN.fullmatch(w)][:words]
                while extra and extra[-1] in LINKERS:
                    extra.pop()
            complete = stop == "eos" and (not after.strip() or bool(extra))
            add(" ".join(x for x in (context.strip(), word, *extra) if x), lp, complete,
                tuple(path.tokens) + tuple(map(str, used)))

        ranked = []
        for suggestion, entry in found.items():
            logprobs = list(entry["paths"].values())
            total = logprobs[0]
            for lp in logprobs[1:]:
                total = _logaddexp(total, lp)
            ranked.append(_Found(suggestion, total, entry["complete"]))
        ranked.sort(key=lambda x: -x.logprob)
        return {"query": text, "items": [
            {"type": "predict", "text": x.text,
             "completion": x.text[len(text):] if x.text.startswith(text) else "",
             "score": round(math.exp(x.logprob), 4), "complete": x.complete}
            for x in ranked[:limit]]}


def _grammar_after(text: str, fragment: str, words: int, step: bool):
    """
    The grammar for what may follow a path's `text`, or None when nothing
    needs holding (the typing is covered and the next space will stop it).
    A branching step (`step`) may also begin the next word, which is how it
    sees that the word ended.
    """
    rest = fragment[len(text):] if fragment.startswith(text) else ""
    started = bool(text.strip()) or bool(rest.strip())
    if step:
        return grammar(rest, max(words, 1) if text else 0, started)
    if rest or not started or words:
        return grammar(rest, words, started)
    return None


def _dedupe(paths: list) -> list:
    """The same tokens reached twice (a branch, and the path it came from) count once."""
    best = {}
    for path in paths:
        key = (tuple(path.tokens), path.ended)
        if key not in best or path.logprob > best[key].logprob:
            best[key] = path
    return list(best.values())


def _agrees(token: str, fragment: str) -> bool:
    """A first token agrees if it covers the typed fragment, or is the start of it."""
    return bool(token) and (token.startswith(fragment) or fragment.startswith(token))


def _logprob_through(steps: list, chars: int) -> float:
    """The log-probability of the generated tokens that make up the next `chars` characters."""
    total, seen = 0.0, 0
    for step in steps:
        if seen >= chars:
            break
        total += step.get("logprob", 0.0)
        seen += len(step.get("token", ""))
    return total


def _logaddexp(a: float, b: float) -> float:
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))
