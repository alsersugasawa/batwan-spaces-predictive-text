# syntax=docker/dockerfile:1
# Batwan's predictive text on a Hugging Face Space, in one container:
#
#   llama-server   127.0.0.1:8080, the GGUF, never reachable from outside
#   app.py         0.0.0.0:7860, the search box and /api/suggest
#
# As batwan-spaces does it: llama.cpp's own server image, Python beside it,
# and the weights fetched by start.sh when the Space starts, not baked in.
# The image is pinned to the llama.cpp build the GGUF was converted with
# (scripts/ci/gguf_from_adapter.sh in batwan-ai-gemma-4-e2b-it-predictive-text):
# Gemma 4 needs a recent one, and the two should not drift apart. Change
# both together.
ARG LLAMA_IMAGE=ghcr.io/ggml-org/llama.cpp:server-b12182
FROM ${LLAMA_IMAGE}

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000. Ubuntu's images already have one
# (ubuntu); make one only where there is none.
RUN id -u 1000 >/dev/null 2>&1 || useradd --uid 1000 --create-home --shell /bin/bash user \
    && mkdir -p /home/user && chown 1000:1000 /home/user

COPY requirements.txt /opt/space/requirements.txt
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --disable-pip-version-check -r /opt/space/requirements.txt

COPY start.sh /opt/start.sh
COPY app.py predict.py /opt/space/
COPY static /opt/space/static
RUN chmod 755 /opt/start.sh

# The base image tells llama-server to listen on 0.0.0.0; it must not.
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/home/user \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/opt/space \
    LD_LIBRARY_PATH=/app \
    LLAMA_ARG_HOST=127.0.0.1

USER 1000
EXPOSE 7860
ENTRYPOINT []
CMD ["/opt/start.sh"]
