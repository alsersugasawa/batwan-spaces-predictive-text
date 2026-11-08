# Changelog

All notable changes to this Space are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). These are the versions of this
Space's packaging; the model's are in
batwan-ai-gemma-4-e2b-it-predictive-text's changelog. A release of each that
go together share a version number.

## [1.0.0] - 2026-11-07

### Added

- **The Space**: llama-server (llama.cpp b12182) serving
  `alsersugasawa/batwan-gemma-4-e2b-predictive-text` →
  `batwan-gemma-4-e2b-predictive-text-1.0.0-Q4_K_M.gguf`, and a web
  service in front of it, in one container, as batwan-spaces is laid out.
  The weights and their `predictive_text.json` are fetched when the Space
  starts; the web service is up at once and says the model is loading until
  it is.
- **`GET /api/suggest`**, shaped like batwan.org's own (`items` with `text`,
  and `"type": "predict"`): the rest of the word being typed, or after a
  space the next word, in `any`, `hil` or `en`, with each suggestion's
  share and whether the model ended the query there (predict.py). Token
  healing: the prompt stops before the word being typed, and the first
  tokens that agree with it come from the model's raw top and from a
  grammar, put on one scale. Three paths branch one token further, so
  words that share a first token are told apart; three beams finish their
  words together, and every point where a word could end is offered.
  Paths are scored with the model's raw probabilities throughout, not with
  a grammar's, which llama.cpp renormalises only sometimes.
- A visitor's newest keystroke wins: an older request still waiting for the
  model is answered empty (`"stale": true`). Answers are cached in memory
  for six hours. 120 requests a minute per visitor. CORS for batwan.org,
  www.batwan.org and batwan-app.alser.ai.
- **The search page**: batwan.org's colours, the best suggestion as grey
  text after the cursor (<kbd>Tab</kbd> or <kbd>→</kbd> takes it and asks for
  the next word), the others below, the three search directions, and
  <kbd>Enter</kbd> to search batwan.org itself. Light and dark.
- Tests: predict.py against a pretend llama-server that is a small language
  model over a few queries.
