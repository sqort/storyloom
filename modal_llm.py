# Storyloom model server on Modal.
# Serves a Hugging Face model through vLLM's OpenAI-compatible API, protected by your own API key.
# The "Deploy model to Modal" GitHub workflow can override the three settings below.

import subprocess

import modal

MODEL = "mistralai/Mistral-Nemo-Instruct-2407"
GPU = "L40S"
MAX_LEN = 16384

IDLE_MINUTES = 5  # shut the GPU off after this many minutes with no requests
MINUTES = 60

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0", "transformers==5.8.1")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("storyloom-llm")


@app.server(
    image=image,
    gpu=GPU,
    port=8000,
    scaledown_window=IDLE_MINUTES * MINUTES,
    startup_timeout=20 * MINUTES,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    # Open to the internet, but vLLM rejects any request without your key (VLLM_API_KEY).
    unauthenticated=True,
    secrets=[
        modal.Secret.from_name("storyloom-llm"),  # VLLM_API_KEY
        modal.Secret.from_name("huggingface"),  # HF_TOKEN, needed for gated models
    ],
)
class Server:
    @modal.enter()
    def start(self):
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
