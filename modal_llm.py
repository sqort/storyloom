# Storyloom model server on Modal.
# Serves a Hugging Face model through an OpenAI-compatible API, protected by your own API key.
#   - Normal models (safetensors) run on vLLM.
#   - GGUF models (repo name contains "GGUF") run on llama.cpp. Pick the quant with a colon:
#       DavidAU/Some-Model-GGUF:Q4_K_M
# The "Deploy model to Modal" GitHub workflow can override the three settings below.

import os
import shutil
import subprocess

import modal

MODEL = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NEO-CODER-MAX-MTP-GGUF"
GPU = "L40S"
MAX_LEN = 16384

IDLE_MINUTES = 5  # shut the GPU off after this many minutes with no requests
MINUTES = 60
IS_GGUF = "gguf" in MODEL.lower()

if IS_GGUF:
    image = (
        modal.Image.from_registry("ghcr.io/ggml-org/llama.cpp:server-cuda", add_python="3.12")
        .entrypoint([])
        .env({"LLAMA_CACHE": "/root/.cache/llama.cpp"})
    )
else:
    image = (
        modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
        .entrypoint([])
        .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
        .env({"HF_XET_HIGH_PERFORMANCE": "1"})
    )

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)
llama_cache = modal.Volume.from_name("llama-cache", create_if_missing=True)

app = modal.App("storyloom-llm")


@app.server(
    image=image,
    gpu=GPU,
    port=8000,
    scaledown_window=IDLE_MINUTES * MINUTES,
    startup_timeout=20 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
        "/root/.cache/llama.cpp": llama_cache,
    },
    # Open to the internet, but the server rejects any request without your key (VLLM_API_KEY).
    unauthenticated=True,
    secrets=[
        modal.Secret.from_name("storyloom-llm"),  # VLLM_API_KEY
        modal.Secret.from_name("huggingface"),  # HF_TOKEN, needed for gated models
    ],
)
class Server:
    @modal.enter()
    def start(self):
        if IS_GGUF:
            binary = shutil.which("llama-server") or "/app/llama-server"
            cmd = [
                binary,
                "-hf", MODEL,  # downloads the GGUF file (repo:QUANT) from Hugging Face
                "--alias", "storyloom",
                "--host", "0.0.0.0",
                "--port", "8000",
                "-c", str(MAX_LEN),
                "-np", "1",  # one chat at a time gets the full context
                "-ngl", "999",  # put every layer on the GPU
                "--jinja",  # use the model's own chat template
                "--api-key", os.environ["VLLM_API_KEY"],
            ]
        else:
            cmd = [
                "vllm", "serve", MODEL,
                "--served-model-name", MODEL, "storyloom",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--max-model-len", str(MAX_LEN),
                "--enforce-eager",  # faster cold starts
            ]
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
        
