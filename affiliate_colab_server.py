# server.py - Khanun affiliate inference (RunPod) - v3
# fixes: persist on /workspace (survives Stop) + /warmup endpoint (preload model inside Pod, avoids 100s proxy 524)
import base64, io, os, time, uuid, gc
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, ImageFilter

PERSIST = "/workspace" if os.path.isdir("/workspace") else os.getcwd()   # /workspace = ถาวร (ไม่โดน Stop ล้าง)
OUT = os.environ.get("OUT_DIR", os.path.join(PERSIST, "aff_outputs"))
MODELS_ROOT = os.environ.get("MODELS_ROOT", os.path.join(PERSIST, "aff_models"))
os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(MODELS_ROOT, "_hf"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
GPU_PROFILE = os.environ.get("GPU_PROFILE", "a100")

app = FastAPI(title="Khanun Affiliate Inference v3")

def _safe(name):
    return "".join(c for c in (name or "") if c.isalnum() or c in "-_ ")[:40].strip().replace(" ", "_") or "job"
def _decode(s):
    if s.startswith("data:"):
        s = s.split(",", 1)[1]; return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    import urllib.request
    with urllib.request.urlopen(s, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")
def _job_dir(ip):
    d = os.path.join(OUT, _safe(ip.get("job") or ip.get("product"))); os.makedirs(d, exist_ok=True); return d
def _save_img(img, ip, p="img"):
    d = _job_dir(ip); fn = "%s_%s.png" % (p, uuid.uuid4().hex[:8]); img.save(os.path.join(d, fn))
    return os.path.relpath(os.path.join(d, fn), OUT)
def _save_bytes(data, ext, ip, p="f"):
    d = _job_dir(ip); fn = "%s_%s.%s" % (p, uuid.uuid4().hex[:8], ext); open(os.path.join(d, fn), "wb").write(data)
    return os.path.relpath(os.path.join(d, fn), OUT)
def _url(req, rel):
    return str(req.base_url).rstrip("/") + "/file/" + rel.replace(os.sep, "/")

_L = {"name": None, "obj": None}
def _free():
    _L["obj"] = None; _L["name"] = None; gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
def _need(name, loader):
    if _L["name"] != name:
        _free(); print("[load]", name, flush=True); _L["obj"] = loader(); _L["name"] = name
    return _L["obj"]

def _load_kontext():
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=DTYPE)
    p.enable_model_cpu_offload(); return p
def _load_wan():
    from diffusers import WanImageToVideoPipeline
    p = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=DTYPE)
    p.enable_model_cpu_offload(); return p
def _load_upscaler():
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import urllib.request
    w = os.path.join(MODELS_ROOT, "RealESRGAN_x2plus.pth")
    if not os.path.exists(w):
        urllib.request.urlretrieve("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth", w)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    return RealESRGANer(scale=2, model_path=w, model=model, half=(DEVICE == "cuda"))
def _load_qwen():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    m = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=DTYPE, device_map="auto")
    return (m, AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct"))
def _load_f5():
    from f5_tts.api import F5TTS
    return F5TTS()

def h_kontext(req, ip):
    imgs = ip.get("images") or ([ip["image_url"]] if ip.get("image_url") else [])
    out = _need("kontext", _load_kontext)(image=_decode(imgs[0]), prompt=ip.get("prompt", ""),
        guidance_scale=3.5, num_inference_steps=int(ip.get("steps", 28))).images[0]
    return {"images": [{"url": _url(req, _save_img(out, ip, "kontext"))}]}
def h_upscale(req, ip):
    img = _decode(ip["image_url"]); f = int(ip.get("upscale_factor", 2))
    try:
        import numpy as np
        arr, _ = _need("upscale", _load_upscaler).enhance(np.array(img)[:, :, ::-1], outscale=f)
        out = Image.fromarray(arr[:, :, ::-1])
    except Exception as e:
        print("[upscale -> lanczos]", str(e)[:120], flush=True)
        out = img.resize((img.width * f, img.height * f), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 130))
    return {"images": [{"url": _url(req, _save_img(out, ip, "hq"))}]}
def h_video(req, ip):
    src = _decode(ip.get("image") or ip.get("image_url")); secs = int(float(ip.get("duration", 10)))
    fr = _need("wan", _load_wan)(image=src, prompt=ip.get("prompt", ""),
        num_frames=min(secs, 5) * 16, num_inference_steps=int(ip.get("steps", 30))).frames[0]
    from diffusers.utils import export_to_video
    d = _job_dir(ip); fn = "clip_%s.mp4" % uuid.uuid4().hex[:8]; export_to_video(fr, os.path.join(d, fn), fps=16)
    return {"video": {"url": _url(req, os.path.relpath(os.path.join(d, fn), OUT))}}
def h_tts(req, ip):
    dd = ip.get("inputs", ip)
    wav, sr, _ = _need("f5", _load_f5).infer(ref_file=os.environ.get("F5_REF_AUDIO", os.path.join(PERSIST, "voice/ref.wav")),
        ref_text=os.environ.get("F5_REF_TEXT", ""), gen_text=dd.get("text", ""))
    import soundfile as sf
    b = io.BytesIO(); sf.write(b, wav, sr, format="WAV")
    return {"audio_url": _url(req, _save_bytes(b.getvalue(), "wav", ip, "vo"))}
def h_vision(req, ip):
    img = _decode(ip.get("image_url") or (ip.get("image_urls") or [""])[0])
    m, pr = _need("qwen", _load_qwen)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the clothing/product briefly in Thai (color, neckline, sleeves, hem, fabric)."}]}]
    t = pr.apply_chat_template(msgs, add_generation_prompt=True)
    b = pr(text=[t], images=[img], return_tensors="pt").to(m.device)
    o = m.generate(**b, max_new_tokens=200)
    ans = pr.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return {"output": ans, "outputs": [ans]}
def h_vton(req, ip):
    raise RuntimeError("CatVTON wrapper ยังไม่ติดตั้ง — โหนดใส่ชุดใช้ fal ไปก่อน")

ROUTER = {"kolors-virtual-try-on": h_vton, "fashn/tryon": h_vton, "idm": h_vton,
    "flux-pro/kontext": h_kontext, "kontext": h_kontext, "clarity-upscaler": h_upscale, "upscaler": h_upscale,
    "image-to-video": h_video, "wan": h_video, "kling-video": h_video, "hailuo": h_video,
    "tts": h_tts, "elevenlabs": h_tts, "vision": h_vision, "any-llm": h_vision}
WARM = {"bg": ("kontext", _load_kontext), "video": ("wan", _load_wan), "upscale": ("upscale", _load_upscaler),
    "vision": ("qwen", _load_qwen), "voice": ("f5", _load_f5)}

@app.get("/health")
def health():
    used = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info(); used = round((total - free) / 1e9, 1)
    return {"ok": True, "device": DEVICE, "loaded": _L["name"], "profile": GPU_PROFILE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "vram_used_gb": used,
        "persist": PERSIST, "models_root": MODELS_ROOT, "out": OUT}

@app.post("/warmup")
async def warmup(req: Request):
    body = await req.json(); node = body.get("node", "")
    if node not in WARM: return JSONResponse({"ok": False, "error": "warmup node: bg|video|upscale|vision|voice"}, 400)
    name, loader = WARM[node]
    try:
        _need(name, loader); return {"ok": True, "loaded": name, "msg": "model in VRAM — app /infer will be fast now"}
    except Exception as e:
        import traceback; traceback.print_exc(); return JSONResponse({"ok": False, "error": str(e)[:500]}, 500)

@app.get("/file/{path:path}")
def get_file(path):
    p = os.path.join(OUT, path)
    return FileResponse(p) if os.path.isfile(p) else JSONResponse({"error": "not_found"}, 404)

@app.post("/shutdown")
def shutdown():
    _free(); return {"ok": True, "freed": True}

@app.post("/infer")
async def infer(req: Request):
    body = await req.json(); mid = body.get("model_id", ""); ip = body.get("inputs", {})
    fn = None
    for k, f in ROUTER.items():
        if k in mid: fn = f; break
    if not fn: return JSONResponse({"ok": False, "error": "no_handler:" + mid}, 400)
    try:
        t0 = time.time(); res = fn(req, ip); print("[infer]", mid, "ok", round(time.time() - t0, 1), flush=True); return {"ok": True, "result": res}
    except Exception as e:
        import traceback; traceback.print_exc(); return JSONResponse({"ok": False, "error": str(e)[:500]}, 500)
