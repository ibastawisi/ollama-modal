import os
import subprocess
import time
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.background import BackgroundTask

from modal import Image, App, enter, asgi_app
import httpx

MODEL = os.environ.get("MODEL", "deepscaler")


def pull(model: str = MODEL):
    subprocess.run(["systemctl", "daemon-reload"])
    subprocess.run(["systemctl", "enable", "ollama"])
    subprocess.run(["systemctl", "start", "ollama"])
    time.sleep(2)  # 2s, wait for the service to start
    subprocess.run(["ollama", "pull", model], stdout=subprocess.PIPE)


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
    .run_function(pull)
)

app = App(name="ollama", image=image)
api = FastAPI()
client = httpx.AsyncClient(base_url="http://localhost:11434/api", timeout=httpx.Timeout(10.0))

@api.get("/api/{path:path}")
async def tile_request(path: str):
    req = client.build_request("GET", path)
    r = await client.send(req, stream=True)
    return StreamingResponse(
        r.aiter_raw(),
        background=BackgroundTask(r.aclose),
        headers=r.headers
   )
   
@api.post("/api/{path:path}")
async def tile_request(path: str, request: Request):
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