# Load the reviewed, pinned GitHub release and run its Gradio-compatible app.
from pathlib import Path
import io, os, runpy, sys, urllib.request, zipfile
REVISION = "2d4ac82344d90e74b9c0afd9c3eeab4309622bc9"
ROOT = Path("/tmp/speechsae-release")
ROOT.mkdir(exist_ok=True)
url = f"https://github.com/kehinde-elelu/SpeechSAE/archive/{REVISION}.zip"
with urllib.request.urlopen(url, timeout=90) as response:
    archive = zipfile.ZipFile(io.BytesIO(response.read()))
    archive.extractall(ROOT)
source = ROOT / f"SpeechSAE-{REVISION}"
os.environ["SAE_BASE"] = str(source)
os.environ["SAE_DEVICE"] = "cpu"
os.environ["PORT"] = "7860"
os.environ["HF_HOME"] = "/tmp/speechsae-hf-cache"
sys.path.insert(0, str(source))
os.chdir(source)
import spaces
import gradio as gr
import threading
from starlette.routing import Route
from starlette.responses import JSONResponse
core = runpy.run_path(str(source / "app.py"), run_name="speech_core")
impl = core["server"]

@spaces.GPU(duration=60)
def gpu_inspect(raw):
    if impl.MODEL is None:
        raise gr.Error("Models are warming up. Please try again shortly.")
    h, sae, _ = impl.MODEL
    impl.MODEL = (h.to("cuda"), sae.to("cuda"), "cuda")
    try:
        result = impl.analyze(raw, "Uploaded recording")
        return {k: v for k, v in result.items() if k not in ["audio", "mel", "waveform", "times", "features"]}
    finally:
        impl.MODEL = (h.to("cpu"), sae.to("cpu"), "cpu")

native = gr.Interface(gpu_inspect, inputs=gr.File(type="binary"), outputs=gr.JSON(), title="SpeechSAE GPU inspection", flagging_mode="never")
server_app, _, _ = native.launch(server_name="0.0.0.0", prevent_thread_lock=True, show_error=True, ssr_mode=False)

async def frontend(request):
    result = await core["proxy"](request.url.path.lstrip("/"), request)
    return JSONResponse(result) if isinstance(result, (dict, list)) else result

# Keep Gradio config/queue routes for the supported ZeroGPU lifecycle.
# The main website uses CPU inference; the native API provides optional GPU inspection.
for pattern in ["/", "/results", "/app.js", "/style.css", "/api/{tail:path}"]:
    server_app.router.routes.insert(0, Route(pattern, endpoint=frontend, methods=["GET", "POST"]))
print("SpeechSAE website mounted on the supported Gradio server", flush=True)
threading.Event().wait()
