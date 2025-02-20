import subprocess
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

from modal import Image, App, Volume, enter, asgi_app
import httpx

MODELS_DOWNLOAD_PATH = "/root/models"
volume = Volume.from_name("ollama", create_if_missing=True)

image = (
    Image.debian_slim()
    .apt_install("curl", "systemctl")
    .run_commands(  # from https://github.com/ollama/ollama/blob/main/docs/linux.md
        "curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama-linux-amd64.tgz",
        "tar -C /usr -xzf ollama-linux-amd64.tgz",
        "useradd -r -s /bin/false -U -m -d /usr/share/ollama ollama",
        "usermod -a -G ollama $(whoami)",
    )
    .add_local_file("ollama.service", "/etc/systemd/system/ollama.service", copy=True)
    .pip_install("fastapi[standard]","ollama", "httpx", "starlette")
    .env({"OLLAMA_MODELS": MODELS_DOWNLOAD_PATH})
    .run_commands(["systemctl daemon-reload", "systemctl enable ollama"])
)

app = App(name="ollama", image=image, volumes={MODELS_DOWNLOAD_PATH: volume})
api = FastAPI()
client = httpx.AsyncClient(base_url="http://localhost:11434/api", timeout=httpx.Timeout(10.0))

@api.get("/api/{path:path}")
async def proxy_request(path: str):
    req = client.build_request("GET", path)
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        background=BackgroundTask(r.aclose),
        headers=r.headers
   )
   
@api.post("/api/{path:path}")
async def proxy_request(path: str, request: Request):
    req = client.build_request("POST", path, data=await request.body())
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        background=BackgroundTask(r.aclose),
        headers=r.headers
    )

@app.cls(gpu="T4",container_idle_timeout=60,)
class Ollama:
    @enter()
    def load(self):
        subprocess.run(["systemctl", "start", "ollama"])

    @asgi_app()
    def serve(self):
        return api