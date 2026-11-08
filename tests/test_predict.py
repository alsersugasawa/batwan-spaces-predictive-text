"""
predict.py against a pretend llama-server.

The pretend server is a language model over a handful of queries: each has
a probability and a tokenisation, and the probability of a next token is
that of the queries that go on with it - as a trained model's would be,
were it this simple. It answers /tokenize and /completion in llama-server's
own shapes (tools/server/README.md at b12182): `top_probs` with
`post_sampling_probs`, `completion_probabilities` with `logprob`, an end of
query as a token whose text is "", `stop_type`. It honours the grammars
predict.py writes and the stop at a space, as llama.cpp does.
"""

import asyncio
import json
import math
import re

import httpx
import pytest

import predict

# Gemma writes "maayong" as " ma" "ay" "ong", and "maayo" as " ma" "ayo":
# the two share a first token, which is what the branching step is for.
PIECES = {
    "maayong": [" ma", "ay", "ong"], "maayo": [" ma", "ayo"], "aga": [" aga"],
    "adlaw": [" ad", "law"], "ad": [" ad"], "hapon": [" hap", "on"], "salamat": [" salamat"], "gid": [" gid"],
    "palangga": [" pal", "angga"], "sa": [" sa"], "imo": [" imo"],
}
QUERIES = {
    "maayong aga": 0.20, "maayong adlaw": 0.10, "maayong hapon": 0.06, "maayong": 0.02,
    "maayong ad": 0.005,
    "maayo": 0.18, "salamat gid": 0.16, "salamat": 0.06, "salamat sa imo": 0.04,
    "palangga": 0.18,
}
EOS = "<eos>"


def tokens_of(words):
    return [t for w in words for t in PIECES[w]]


class FakeServer:
    def __init__(self, queries=QUERIES):
        self.queries = {q: (p, tokens_of(q.split()) + [EOS]) for q, p in queries.items()}
        self.calls = []

    # -- the model ---------------------------------------------------------
    def next_tokens(self, prefix):
        """{token: probability} after a prefix of query tokens."""
        out, total = {}, 0.0
        for p, toks in self.queries.values():
            if toks[:len(prefix)] == prefix and len(toks) > len(prefix):
                out[toks[len(prefix)]] = out.get(toks[len(prefix)], 0.0) + p
                total += p
        return {t: p / total for t, p in out.items()} if total else {}

    @staticmethod
    def allowed(token, gram, generated):
        """Would llama.cpp's grammar let `token` follow `generated`?"""
        if gram is None:
            return True
        lit = re.match(r'root ::= "((?:[^"\\]|\\.)*)"', gram)
        rest = lit.group(1).replace('\\"', '"') if lit else ""
        more = re.search(r"\{0,(\d+)\}", gram)
        words = int(more.group(1)) if more else 0
        needs_letter = "]+" in gram.split("(")[0] and not rest
        owed = rest[len(generated):] if rest.startswith(generated) else ""
        text = "" if token == EOS else token
        if owed:
            return bool(text) and (text.startswith(owed) or owed.startswith(text))
        if token == EOS:
            return not (needs_letter and not generated.strip())
        if text.startswith(" "):
            spaces = (generated + text).count(" ") - rest.count(" ")
            return words > 0 and spaces <= words and (generated.strip() or rest.strip())
        return True

    # -- the server --------------------------------------------------------
    def prompt_tokens(self, prompt):
        prefix_ok = isinstance(prompt[0], str) and prompt[0].startswith("P:")
        assert prefix_ok, prompt
        return [t for t in prompt[1:]]

    def handle(self, request):
        body = json.loads(request.content)
        self.calls.append((request.url.path, body))
        if request.url.path == "/tokenize":
            content = body["content"]
            prefix = next(p for p in sorted(predict.DEFAULT_FORMAT["prefix"].values(),
                                            key=len, reverse=True) if content.startswith(p))
            context = content[len(prefix):].split()
            return httpx.Response(200, json={"tokens": ["P:" + prefix] + tokens_of(context)})
        prompts = body["prompt"]
        prompts = prompts if isinstance(prompts[0], list) else [prompts]
        gram = body.get("grammar")
        out = []
        for prompt in prompts:
            prefix = self.prompt_tokens(prompt)
            if body["n_predict"] == 1:
                dist = {t: p for t, p in self.next_tokens(prefix).items()
                        if self.allowed(t, gram, "")}
                total = sum(dist.values()) or 1.0
                top = sorted(dist.items(), key=lambda kv: -kv[1])[:body["n_probs"]]
                out.append({"content": "", "completion_probabilities": [{"top_probs": [
                    {"id": t, "token": "" if t == EOS else t, "prob": p / total} for t, p in top]}]})
                continue
            generated, steps, stop = "", [], "limit"
            for _ in range(body["n_predict"]):
                dist = {t: p for t, p in self.next_tokens(prefix).items()
                        if self.allowed(t, gram, generated)}
                if not dist:
                    break
                t, p = max(dist.items(), key=lambda kv: kv[1])
                steps.append({"id": t, "token": "" if t == EOS else t, "logprob": math.log(p)})
                if t == EOS:
                    stop = "eos"
                    break
                if body.get("stop") == [" "] and " " in t:
                    stop = "word"
                    break
                generated += t
                prefix = prefix + [t]
            out.append({"content": generated, "stop_type": stop, "completion_probabilities": steps})
        return httpx.Response(200, json=out if isinstance(body["prompt"][0], list) else out[0])


def run(coro):
    return asyncio.run(coro)


def predictor(server=None, **settings):
    server = server or FakeServer()
    client = httpx.AsyncClient(transport=httpx.MockTransport(server.handle))
    return predict.Predictor(client, "http://llm", settings=predict.Settings(**settings)), server


def texts(out):
    return [i["text"] for i in out["items"]]


# -- the typing --------------------------------------------------------------

@pytest.mark.parametrize("typed, context, fragment", [
    ("maayong a", " maayong", " a"),
    ("maa", "", " maa"),
    ("maayong ", " maayong", " "),
    ("  Maayong   Aga  ", " maayong aga", " "),
    ("salamat?", " salamat", " "),
])
def test_typing_is_split_into_context_and_the_word_being_typed(typed, context, fragment):
    assert predict.split_typed(predict.normalise(typed)) == (context, fragment)


def test_normalise_writes_queries_as_training_did():
    assert predict.normalise("Ang “Palangga”, Bala?") == "ang palangga bala "
    assert predict.normalise("D’yos") == "d'yos"
    assert len(predict.normalise("a" * 300)) == predict.MAX_QUERY


def test_grammars():
    w = f"[{predict.WORD_CHARS}]"
    assert predict.grammar(" maa", 0, True) == f'root ::= " maa" {w}*'
    assert predict.grammar("", 0, False) == f"root ::= {w}+"
    assert predict.grammar("a", 1, True) == f'root ::= "a" {w}* (" " {w}+){{0,1}}'
    assert predict.grammar('x"y', 0, True).startswith('root ::= "x\\"y"')


# -- the suggestions ---------------------------------------------------------

def test_two_words_that_share_a_first_token_are_both_found():
    p, server = predictor(limit=3)
    out = run(p.suggest("maa"))
    assert texts(out)[:2] == ["maayong", "maayo"]
    assert out["items"][0]["completion"] == "yong"
    first = [b for path, b in server.calls if path == "/completion"][0]
    assert first["grammar"].startswith('root ::= " maa"') and first["post_sampling_probs"]


def test_scores_are_the_probability_of_the_word():
    p, _ = predictor(limit=3)
    items = run(p.suggest("maa"))["items"]
    # maayong (any query that starts with it) 0.38 of the 0.56 that start " ma".
    assert items[0]["score"] == pytest.approx(0.38 / 0.56, abs=0.01)
    assert items[1]["score"] == pytest.approx(0.18 / 0.56, abs=0.01)


def test_next_word_after_a_space():
    p, _ = predictor(limit=3)
    out = run(p.suggest("maayong "))
    assert texts(out) == ["maayong aga", "maayong adlaw", "maayong hapon"]
    assert out["items"][0]["completion"] == "aga"
    assert out["items"][0]["complete"] is True      # the model ends the query there


def test_a_word_that_rarely_ends_is_not_ranked_above_what_it_goes_on_to():
    # " ad" is the first token of "adlaw" and, rarely, a query's last word:
    # it gets only the share of " ad" that ends there.
    p, _ = predictor(limit=4)
    items = run(p.suggest("maayong a"))["items"]
    assert [i["text"] for i in items][:2] == ["maayong aga", "maayong adlaw"]
    ad = [i for i in items if i["text"] == "maayong ad"]
    assert not ad or ad[0]["score"] < 0.05


def test_the_rest_of_a_word_after_others():
    p, _ = predictor(limit=3)
    assert texts(run(p.suggest("maayong h"))) == ["maayong hapon"]


def test_a_word_already_whole_is_offered_as_it_is():
    p, _ = predictor(limit=3)
    out = run(p.suggest("salamat"))
    assert texts(out) == ["salamat"]
    assert out["items"][0]["completion"] == ""


def test_extra_words_only_when_the_model_ends_the_query_with_them():
    p, _ = predictor(limit=3, words=1)
    assert texts(run(p.suggest("salamat ")))[0] == "salamat gid"
    p, _ = predictor(limit=3, words=0)
    assert texts(run(p.suggest("salamat ")))[0] == "salamat gid"


def test_half_a_word_is_never_suggested():
    p, _ = predictor(limit=5, max_tokens=1, branch=0)
    # "palangga" needs " pal" "angga": with one token after the first it
    # does not end, and is left out rather than offered as "palangg".
    out = run(p.suggest("pa"))
    assert all(t in ("palangga",) for t in texts(out)) and "pal" not in texts(out)


def test_nothing_the_model_does_not_know():
    p, _ = predictor()
    assert run(p.suggest("xyz"))["items"] == []


def test_the_direction_chooses_the_prefix():
    p, server = predictor()
    run(p.suggest("maa", lang="en"))
    run(p.suggest("maa", lang="nonsense"))
    asked = [b["content"] for path, b in server.calls if path == "/tokenize"]
    assert asked == ["Batwan search (English):", "Batwan search:"]


def test_nothing_typed_asks_nothing():
    p, server = predictor()
    assert run(p.suggest("   "))["items"] == []
    assert server.calls == []


def test_format_file_falls_back_to_the_default(tmp_path):
    assert predict.load_format(str(tmp_path / "missing.json")) is predict.DEFAULT_FORMAT
    bad = tmp_path / "bad.json"
    bad.write_text('{"prefix": {"any": "x"}}')
    assert predict.load_format(str(bad)) is predict.DEFAULT_FORMAT
    good = tmp_path / "good.json"
    good.write_text('{"prefix": {"any": "A:", "hil": "H:", "en": "E:"}}')
    assert predict.load_format(str(good))["prefix"]["hil"] == "H:"
