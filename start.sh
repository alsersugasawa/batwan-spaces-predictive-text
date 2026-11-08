#!/usr/bin/env bash
# Start the Space: fetch the GGUF and its format file, start llama-server on
# localhost, then the web service on the Space's port.
#
# The web service starts at once, whether or not the model is there yet, so
# the Space is up while 3 GB downloads; until the model is loaded it answers
# with no suggestions and says the model is loading, as Batwan's Space does.
set -euo pipefail

MODEL_REPO="${MODEL_REPO:-alsersugasawa/batwan-gemma-4-e2b-predictive-text}"
MODEL_SOURCE="${MODEL_SOURCE:-batwan-gemma-4-e2b-predictive-text-1.0.0-Q4_K_M.gguf}"
PORT="${PORT:-7860}"
export MODEL_SOURCE

# /data is the Space's persistent storage, when it has some: the model then
# survives a restart. Otherwise it is downloaded again each time.
if [ -d /data ] && [ -w /data ]; then STORE=/data; else STORE="$HOME/store"; fi
mkdir -p "$STORE/models"
MODEL="$STORE/models/$MODEL_SOURCE"
FORMAT="$STORE/models/${MODEL_SOURCE%.gguf}.predictive_text.json"
export PREDICT_FORMAT="$FORMAT"
export PREDICT_LLM_URL="${PREDICT_LLM_URL:-http://127.0.0.1:8080}"

# The Space's own address, https://<owner>-<name>.hf.space, set by Hugging
# Face, beside batwan.org's: pages whose browsers may call /api/suggest.
if [ -n "${SPACE_HOST:-}" ]; then SPACE_ORIGIN="https://$SPACE_HOST,"; fi
export PREDICT_ALLOWED_ORIGINS="${PREDICT_ALLOWED_ORIGINS:-${SPACE_ORIGIN:-}https://batwan.org,https://www.batwan.org,https://batwan-app.alser.ai}"

fetch() {   # fetch <file in MODEL_REPO> <destination>
    local auth=()
    # The token goes to curl on standard input, not on the command line.
    if [ -n "${HF_TOKEN:-}" ]; then auth=(-H @-); fi
    printf 'Authorization: Bearer %s\n' "${HF_TOKEN:-}" | curl -fsSL --retry 5 --retry-delay 10 -C - \
        "${auth[@]}" -o "$2.part" "https://huggingface.co/$MODEL_REPO/resolve/main/$1" \
        && mv "$2.part" "$2"
}

fetch_and_serve() {
    if [ ! -s "$MODEL" ] || [ "$(head -c 4 "$MODEL")" != "GGUF" ]; then
        rm -f "$MODEL"
        echo "fetching $MODEL_SOURCE from $MODEL_REPO"
        if ! fetch "$MODEL_SOURCE" "$MODEL"; then
            echo "could not fetch the model: if $MODEL_REPO is private, is HF_TOKEN set as a Space secret with read access to it?" >&2
            return 1
        fi
        if [ "$(head -c 4 "$MODEL")" != "GGUF" ]; then
            echo "$MODEL_SOURCE is not a GGUF file - check MODEL_REPO and MODEL_SOURCE" >&2
            rm -f "$MODEL"
            return 1
        fi
    fi
    # How the model was trained to read a query. A stock Gemma GGUF has none,
    # and is served with the same prefixes (predict.py's DEFAULT_FORMAT).
    if [ ! -s "$FORMAT" ]; then
        fetch predictive_text.json "$FORMAT" 2>/dev/null || echo "no predictive_text.json in $MODEL_REPO: using the default format"
    fi
    echo "starting llama-server on $MODEL_SOURCE"
    # Small slots: a query is a few dozen tokens. --parallel beyond the beams
    # lets two visitors' suggestions share a batch.
    exec /app/llama-server --model "$MODEL" --host 127.0.0.1 --port 8080 \
        --ctx-size "${LLM_CTX:-2048}" --threads "${LLM_THREADS:-$(nproc)}" \
        --parallel "${LLM_PARALLEL:-4}" --alias predictive
}

fetch_and_serve &

cd /opt/space
exec uvicorn app:app --host 0.0.0.0 --port "$PORT" \
    --proxy-headers --forwarded-allow-ips "*" --no-server-header
