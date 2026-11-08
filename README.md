---
title: Batwan Predictive Text
emoji: ⌨️
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: apache-2.0
models:
  - alsersugasawa/batwan-gemma-4-e2b-predictive-text
  - google/gemma-4-E2B-it
datasets:
  - alsersugasawa/batwan-hiligaynon
short_description: Predictive text for batwan.org's search box (Hiligaynon)
---

# Batwan predictive text on Hugging Face Spaces

> **This is a demo to showcase Batwan's predictive text and is not intended
> for commercial use.**

A search box that guesses what you are typing, for
[batwan.org](https://batwan.org)'s Hiligaynon-English dictionary: the rest
of the word, or, after a space, the next one. The best guess is shown as
grey text after the cursor (<kbd>Tab</kbd> takes it), and the others under
the box, as batwan.org shows its own suggestions.

The model is
[`alsersugasawa/batwan-gemma-4-e2b-predictive-text`](https://huggingface.co/alsersugasawa/batwan-gemma-4-e2b-predictive-text),
a LoRA fine-tune of
[`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it),
3 GB at Q4_K_M. It was trained on search queries cut from
[`alsersugasawa/batwan-hiligaynon`](https://huggingface.co/datasets/alsersugasawa/batwan-hiligaynon):
the dictionary's Hiligaynon words, their English, phrases from the prose,
and the questions AI mode is asked. How, and how well it does, is in
[batwan-ai-gemma-4-e2b-it-predictive-text](https://github.com/alsersugasawa/batwan-ai-gemma-4-e2b-it-predictive-text).

One container, laid out as [batwan-spaces](https://github.com/alsersugasawa/batwan-spaces)'s is:

- **llama-server** (llama.cpp, build b12182) on `127.0.0.1:8080`, serving
  the GGUF. Only the web service can reach it.
- **[app.py](app.py)**, on the Space's port 7860: the search page and
  `/api/suggest`. [predict.py](predict.py) is how a suggestion is asked for.

The weights are not in the image: [start.sh](start.sh) fetches the model,
and the `predictive_text.json` it was trained with, when the Space starts.

## How a suggestion is made

The model was trained on queries written as
`<bos>Batwan search: maayong aga<eos>`. Asked about `maayong a`, the
service asks llama-server how `Batwan search: maayong` goes on, in three
small steps ([predict.py](predict.py)):

1. **Which word.** The prompt stops *before* the word being typed: token
   healing, so the model sees the word whole, as in training, not cut where
   the typing stopped. Of its likeliest next tokens, those that agree with
   ` a…` are kept, and a grammar finds the ones that are not among the
   likeliest (Gemma writes *salamat* as ` sal` `amat`, and ` sal` is not
   among the likeliest first tokens; ` sa`, a word of its own, is).
2. **Branch.** The three likeliest paths go one token further, so words
   that share a first token (*maayong* and *maayo* both start ` ma`) each
   get a chance, and a word that ends there is seen to end.
3. **The rest of the word**, greedily, for the best three, all at once;
   every point on the way where the word could end is offered too
   (*kamatis* as well as *kamatison*).

Every path is scored with the model's own probabilities, read from
llama-server's raw top-k, not with what a grammar leaves: llama.cpp applies
a grammar lazily, so its probabilities are sometimes renormalised and
sometimes not. A suggestion's score is its share of all those found.

Gemma 4 E2B writes about 6 to 9 tokens a second on two CPU cores; a
suggestion takes a dozen or so, so it arrives in about two seconds on the
free tier. Each visitor's latest keystroke wins: an older request still
waiting is dropped.

## The API

```sh
curl 'https://alsersugasawa-batwan-predictive-text.hf.space/api/suggest?q=maayong%20a&lang=any&limit=3'
```

```json
{"query": "maayong a", "lang": "any", "model": "batwan-gemma-4-e2b-predictive-text-1.0.0-Q4_K_M.gguf",
 "items": [{"type": "predict", "text": "maayong aga", "completion": "ga", "score": 0.61, "complete": true}],
 "ms": 1180}
```

| Parameter | |
|---|---|
| `q` | what is typed, 1 to 100 characters. A trailing space asks for the next word. |
| `lang` | `any` (default), `hil` or `en`: batwan.org's search directions, as the model was trained with them |
| `limit` | suggestions, 1 to 8 (default 3) |
| `words` | 0 to 3: a suggestion may run on for this many words past the current one, when the model ends the query there (default 0) |

| Field | |
|---|---|
| `items[].text` | the whole query suggested |
| `items[].completion` | what to add to what was typed: the grey text |
| `items[].score` | the probability of the word, among the words the typing allows |
| `items[].complete` | the model ended the query there |
| `status` | `loading` or `down` instead of items while the model is not up |
| `stale` | `true`: a newer request from the same visitor came in; ignore this one |

The shape is batwan.org's own `/api/suggest`'s (`items`, with `text`), with
`"type": "predict"`, so its search box can show both: its dictionary's
headwords, which are instant, and these, a moment later. Browsers may call
it from batwan.org, www.batwan.org, batwan-app.alser.ai and the Space's own
page (`PREDICT_ALLOWED_ORIGINS`). `/docs` has the OpenAPI page.

### Things to know before relying on it

- **It is not instant.** A second or two on the free CPU tier; a cached
  query is answered at once. Call it after the typing pauses (the page
  waits 140 ms) and cancel what is outdated.
- **It suggests what is likely to be typed, not what is right.** It has
  learned which words follow which in the dataset, including its mistakes.
  No Hiligaynon speaker has reviewed it.
- **A sleeping Space wakes slowly.** A free Space sleeps after two days
  without visitors; the first call after that waits for it to start and the
  model to download and load (`status: loading` until then).

## Setting up the Space

1. Create a **Docker** Space, `alsersugasawa/batwan-predictive-text`, and
   push this repository to it.
2. The Space downloads the GGUF from the model repository when it starts.
   Make that repository public, or add `HF_TOKEN` as a Space secret: a
   fine-grained token with read access to it alone. Without either, the
   page says the model could not be loaded.
3. Hardware: the free **CPU basic** tier (2 vCPU, 16 GB) holds the 3 GB
   model with room to spare. A larger CPU tier makes suggestions faster;
   llama-server uses every core (`LLM_THREADS`).

## Settings

| Variable | Default | |
|---|---|---|
| `MODEL_REPO` | `alsersugasawa/batwan-gemma-4-e2b-predictive-text` | where the GGUF is |
| `MODEL_SOURCE` | `batwan-gemma-4-e2b-predictive-text-1.0.0-Q4_K_M.gguf` | which file |
| `LLM_CTX` | `2048` | llama-server's context, shared by its slots |
| `LLM_PARALLEL` | `4` | slots: the beams of one suggestion run side by side |
| `LLM_THREADS` | all cores | |
| `PREDICT_CANDIDATES` | `16` | the model's top tokens read at each step |
| `PREDICT_BRANCH` | `3` | paths taken one token further before the beams are chosen |
| `PREDICT_BEAMS` | `3` | paths carried to the end of their word |
| `PREDICT_MAX_TOKENS` | `4` | tokens a beam may take to end its word |
| `PREDICT_END_PROBS` | `12` | top tokens read at each step of a beam, for where its word could end |
| `PREDICT_LIMIT` | `3` | suggestions by default |
| `PREDICT_MIN_SHARE` | `0.02` | suggestions with a smaller share are not shown (one always is) |
| `PREDICT_RATE_LIMIT` | `120` | requests per visitor per minute |
| `PREDICT_CACHE_SECONDS` | `21600` | how long an answer is kept, in memory |
| `PREDICT_ALLOWED_ORIGINS` | the Space, batwan.org, batwan-app.alser.ai | pages that may call the API from a browser |
| `SPACE_FRAME_ANCESTORS` | `https://huggingface.co` | pages that may frame the Space |

The stock model can be served for comparison: `MODEL_REPO=ggml-org/gemma-4-E2B-it-GGUF`,
`MODEL_SOURCE=gemma-4-E2B-it-Q4_0.gguf`. It has no `predictive_text.json`,
and is asked with the same prefixes; it has never seen them, and it shows.

## Privacy

Nothing typed is logged or stored. The log has, per suggestion, its
direction, how many words were typed, how many suggestions there were and
how long it took. Answers are cached in memory, under the text typed, for
six hours, so that a common prefix is answered at once.

## Testing locally

```sh
docker build -t batwan-predictive-text .
docker run --rm -p 7860:7860 batwan-predictive-text           # http://localhost:7860
python -m pytest -q tests                                       # predict.py against a pretend llama-server
```

## Licence

This repository: Apache-2.0. The model is a fine-tune of Gemma 4 (Apache
2.0) on data whose sources include non-commercial and share-alike terms,
and two that state no licence, so it is released for non-commercial use
(CC-BY-NC-4.0); see its card.
