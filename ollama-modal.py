import subprocess

from modal import Image, App, Volume, web_server

MODELS_DOWNLOAD_PATH = "/root/models"
volume = Volume.from_name("ollama", create_if_missing=True)

image = (
    Image.debian_slim()
    .apt_install("curl")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .env({"OLLAMA_MODELS": MODELS_DOWNLOAD_PATH, "OLLAMA_HOST": "0.0.0.0"})
)

app = App(name="ollama", image=image, volumes={MODELS_DOWNLOAD_PATH: volume})

@app.function(gpu="T4",container_idle_timeout=60,)
@web_server(port=11434)
def serve():
    subprocess.run("ollama serve &", shell=True)