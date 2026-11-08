"""
What to suggest for what has been typed into batwan.org's search box.

The model is batwan-gemma-4-e2b-predictive-text, a LoRA of
google/gemma-4-E2B-it trained on search queries written as

    <bos>Batwan search: maayong aga<eos>

(the prefix for each direction is in predictive_text.json, which comes from
the model repository, so this and the training agree by construction). It
suggests as a phone keyboard does: the rest of the word being typed, or,
after a space, the next word. For "maayong a" it is asked how
"Batwan search: maayong" goes on, in three steps through llama-server:

1. **Which word.** The prompt stops before the word being typed. This is
   token healing: the word is not cut where the typing stopped, so the
   model sees it as it saw every word in training, whole and with its
   space. One step gives the model's top tokens there, of which those that
   agree with the typing - " aga", " adlaw", " a", " " - are kept. (A word
   so unlikely that none is in the top is found with a grammar instead.)
2. **Branch.** Words that share a first token would otherwise share a
   beam: Gemma writes both "maayong" and "maayo" as " ma" and more. The
   `branch` likeliest paths are taken one token further, all their
   continuations at once; one that starts a new word, or ends the query,
   says the word ended there.
3. **The rest of the word.** The best `beams` open paths are carried on
   greedily, together (one request, several prompts, as many slots), until
   the word ends. A path that covers only part of the typing (" ma" for
   "maa") is held to the rest of it by a grammar.

A suggestion is ranked by the probability of its word's tokens - the
model's own, unconstrained probabilities, read from llama-server's raw
top-k and filtered here against what is typed. Not the probabilities a
grammar leaves: llama.cpp applies a grammar lazily (it samples freely and
resamples only if that was not allowed), so a grammar's `top_probs` are
sometimes renormalised over what it allows and sometimes not, and a token
that is a likely word on its own (" sa", for "sala...") would be scored as
if its being a prefix were as likely. One path of tokens counts once
however it ended; two tokenisations of one word add up. A score is the
suggestion's share of all those found. `words` (0 by default) lets a
suggestion run on for that many more words when the model ends the query
with them.

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
    candidates: int = 16  # raw top tokens asked for at each step, before those that agree are kept
    branch: int = 3       # paths taken one token further before the beams are chosen
    max_tokens: int = 4   # per beam in round 2, before the word must have ended
    end_probs: int = 12   # raw top tokens per greedy step, read for where the word could end
    words: int = 0        # more words a suggestion may run on for
    limit: int = 3
    min_share: float = 0.02  # suggestions with less of the share are not shown (one always is)


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

    async def _step(self, ids: list, paths: list) -> list:
        """
        For each path, the model's raw distribution over its next token: the
        top `candidates` with their unconstrained log-probabilities. One request.
        """
        body = {"prompt": [ids + path.tokens for path in paths], "n_predict": 1,
                "n_probs": self.settings.candidates, "post_sampling_probs": False,
                "temperature": 0.0, "cache_prompt": True}
        result = await self._post("/completion", body)
        return [((r.get("completion_probabilities") or [{}])[0].get("top_logprobs") or [])
                for r in (result if isinstance(result, list) else [result])]

    async def _constrained(self, ids: list, fragment: str) -> list:
        """
        The first tokens a grammar allows, for a word so unlikely that none
        of them is in the raw top: probabilities renormalised over what the
        grammar allows, so they rank these among themselves only.
        """
        result = await self._post("/completion", {
            "prompt": ids, "n_predict": 1, "n_probs": 16, "post_sampling_probs": True,
            "grammar": grammar(fragment, 0, fragment != " "), "samplers": ["top_k", "temperature"],
            "top_k": 16, "temperature": 1.0, "cache_prompt": True})
        tops = (result.get("completion_probabilities") or [{}])[0].get("top_probs") or []
        return [{"id": c["id"], "token": c["token"], "logprob": math.log(c["prob"])}
                for c in tops if c.get("prob", 0) > 0]

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

        # 1. The first tokens that agree with the typing: those in the raw
        # top, and those a grammar finds - the word's own first token is
        # often not among the likeliest after the prompt (" sal" is not;
        # " sa", a word of its own, is). The grammar's probabilities are put
        # on the raw scale by the tokens both lists have.
        def agrees(c):
            token = c.get("token", "")
            # A bare space is never how a word starts in training - Gemma
            # writes the space onto the word (" kam", not " " "kam") - and a
            # path from one spells its word a letter or two at a time.
            if not token.strip():
                return False
            return _agrees(token, fragment) and WORD_RUN.fullmatch(
                token[len(fragment):] if token.startswith(fragment) else "")
        raw, constrained = await asyncio.gather(self._step(ids, [root]),
                                                self._constrained(ids, fragment))
        firsts = {c["id"]: (c["token"], c["logprob"]) for c in raw[0] if agrees(c)}
        constrained = [c for c in constrained if agrees(c)]
        shifts = sorted(firsts[c["id"]][1] - c["logprob"] for c in constrained if c["id"] in firsts)
        shift = shifts[len(shifts) // 2] if shifts else 0.0
        for c in constrained:
            firsts.setdefault(c["id"], (c["token"], c["logprob"] + shift))
        paths = [_Path([tid], token, lp) for tid, (token, lp) in firsts.items()]
        if not paths:
            return {"query": text, "items": []}

        # 2. The likeliest paths one token further: a word that shares its
        # first token with a likelier one gets a beam of its own, and a word
        # that ends there is seen to end.
        if s.branch:
            paths.sort(key=lambda p: -p.logprob)
            parents, others = paths[:s.branch], paths[s.branch:]
            children = []
            for parent, nexts in zip(parents, await self._step(ids, parents)):
                owed = fragment[len(parent.text):] if fragment.startswith(parent.text) else ""
                started = bool(parent.text.strip())
                ends = []
                for cand in nexts:
                    token, lp = cand.get("token", ""), cand.get("logprob", -math.inf)
                    if owed:
                        # Still owes typed letters: only tokens that write them.
                        if token and (token.startswith(owed) or owed.startswith(token)) and \
                                WORD_RUN.fullmatch(token[len(owed):] if token.startswith(owed) else ""):
                            children.append(_Path(parent.tokens + [cand["id"]], parent.text + token,
                                                  parent.logprob + lp))
                    elif token == "" or token.startswith(" "):
                        # The query ends, or another word begins.
                        if started:
                            ends.append((lp, "eos" if token == "" else "word"))
                    elif WORD_RUN.fullmatch(token):
                        children.append(_Path(parent.tokens + [cand["id"]], parent.text + token,
                                              parent.logprob + lp))
                if ends:
                    total = ends[0][0]
                    for lp, _ in ends[1:]:
                        total = _logaddexp(total, lp)
                    eos = [lp for lp, kind in ends if kind == "eos"]
                    eos_total = eos[0] if eos else -math.inf
                    for lp in eos[1:]:
                        eos_total = _logaddexp(eos_total, lp)
                    children.append(_Path(parent.tokens, parent.text, parent.logprob + total,
                                          "eos" if eos_total >= total - math.log(2) else "word"))
            paths = children + others
        paths = _dedupe(paths)
        paths.sort(key=lambda p: -p.logprob)

        # 3. Each open path to the end of its word, greedily. A path that has
        # covered the typing needs no grammar: a stop at the next space ends
        # the word, and the model was taught to write words only.
        open_paths = [p for p in paths if not p.ended][:max(s.beams, limit)]
        groups = {}
        for i, path in enumerate(open_paths):
            groups.setdefault(_grammar_after(path.text, fragment, words), []).append(i)
        results = [None] * len(open_paths)

        async def run(gram, members):
            body = {"prompt": [ids + open_paths[i].tokens for i in members],
                    "n_predict": s.max_tokens + 3 * words, "n_probs": s.end_probs,
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
            if not words:
                # Every point on the way where the word could end is a
                # suggestion: each step's raw top says how likely the query
                # was to end, or another word to begin, there. ("kamatis" is
                # offered even if the greedy path goes on to "kamatison".)
                so_far, lp, used, offered = path.text, path.logprob, [], set()
                for step in r.get("completion_probabilities") or []:
                    if _is_word(so_far, fragment):
                        ends = [c for c in step.get("top_logprobs") or []
                                if c.get("token", "") == "" or c["token"].startswith(" ")]
                        if ends:
                            total = _logsumexp([c["logprob"] for c in ends])
                            eos = _logsumexp([c["logprob"] for c in ends if c.get("token", "") == ""])
                            add(" ".join(x for x in (context.strip(), so_far.strip()) if x),
                                lp + total, eos >= total - math.log(2),
                                tuple(path.tokens) + tuple(used))
                            offered.add(tuple(used))
                    token = step.get("token", "")
                    if token == "" or token.startswith(" ") or not WORD_RUN.fullmatch(token):
                        break
                    lp += step.get("logprob", 0.0)
                    so_far += token
                    used.append(str(step.get("id", token)))
                else:
                    # Out of tokens: whatever word it was in the middle of is
                    # not offered, half a word not being a suggestion.
                    if r.get("stop_type") not in ("eos", "word"):
                        continue
                # llama-server does not report the step a stop string fired
                # on: the word it stopped at is offered without the end's
                # probability (which, a word having ended, is most of it).
                if (tuple(used) not in offered and _is_word(so_far, fragment)
                        and r.get("stop_type") in ("eos", "word")):
                    add(" ".join(x for x in (context.strip(), so_far.strip()) if x), lp,
                        r.get("stop_type") == "eos", tuple(path.tokens) + tuple(used))
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
        if not ranked:
            return {"query": text, "items": []}
        total = ranked[0].logprob
        for x in ranked[1:]:
            total = _logaddexp(total, x.logprob)
        # A long tail of near-zero shares is mostly words that are not words
        # ("palangetcwd"): on the assessment, 2% hid 150 of 1,166
        # suggestions and one right answer.
        shown = [x for x in ranked if math.exp(x.logprob - total) >= s.min_share] or ranked[:1]
        return {"query": text, "items": [
            {"type": "predict", "text": x.text,
             "completion": x.text[len(text):] if x.text.startswith(text) else "",
             "score": round(math.exp(x.logprob - total), 4), "complete": x.complete}
            for x in shown[:limit]]}


def _grammar_after(text: str, fragment: str, words: int):
    """
    The grammar for what may follow a path's `text` when it is carried on
    greedily, or None when nothing needs holding (the typing is covered and
    the next space will stop it).
    """
    rest = fragment[len(text):] if fragment.startswith(text) else ""
    started = bool(text.strip()) or bool(rest.strip())
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


def _is_word(text: str, fragment: str) -> bool:
    """A path's text is a whole word that agrees with the typing: " kamatis" for " kam"."""
    return (text.startswith(fragment) and len(text) > 1 and text[1].isalpha()
            and bool(WORD_RUN.fullmatch(text[1:])))


def _logsumexp(values: list) -> float:
    if not values:
        return -math.inf
    hi = max(values)
    return hi + math.log(sum(math.exp(v - hi) for v in values))


def _logaddexp(a: float, b: float) -> float:
    hi = max(a, b)
    return hi + math.log(math.exp(a - hi) + math.exp(b - hi))
