"""
affiliate_colab_server.py — เซิร์ฟเวอร์อินเฟอเรนซ์ open-source สำหรับไลน์ผลิตคลิป (รันบน Google Colab)
=====================================================================================
แทนที่ fal.ai ด้วยโมเดล open ตัวเทียบเท่า · panel/pipeline ฝั่งแอปเดิมไม่ต้องแก้
รับ POST /infer {model_id, inputs}  ->  คืน result สคีมาเดียวกับ fal (images/video/audio + url)

map model_id (fal) -> โมเดล open:
  fal-ai/kling/v1-5/kolors-virtual-try-on   -> CatVTON / Kolors-VTON   (ใส่ชุด)
  fal-ai/flux-pro/kontext/max/multi         -> FLUX.1-Kontext-dev      (เปลี่ยนพื้นหลัง/แก้ภาพ)
  fal-ai/clarity-upscaler                   -> Real-ESRGAN / clarity   (ความคม)
  fal-ai/kling-video/v2.5-turbo/...i2v      -> Wan2.2-I2V              (วีดีโอ)
  fal-ai/elevenlabs/tts/eleven-v3           -> F5-TTS (เสียงโคลนไทย)   (เสียง)
  fal-ai/any-llm/vision                     -> Qwen2-VL               (อ่านภาพ)

★ VRAM: โหลดโมเดล "ทีละตัว" (lazy + unload) เพราะ Colab A100 40GB ไม่พอโหลดพร้อมกัน
★ รันบน Colab: ดู colab/RUN_ON_COLAB.md (ติดตั้ง deps -> วางไฟล์นี้ -> รัน -> ได้ ngrok URL)
"""
import base64, io, os, time, uuid, gc, json
from pathlib import Path

import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image

OUT = Path("/content/outputs"); OUT.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
GPU_PROFILE = os.environ.get("GPU_PROFILE", "a100")   # a100 | l4 | t4 (เลือกขนาดโมเดล)

# ---------- เก็บโมเดลแยกโฟลเดอร์ตามหน้าที่ บน Google Drive (persist ข้ามรอบ ไม่ต้องโหลดซ้ำ) ----------
# MODELS_ROOT ชี้ไปโฟลเดอร์ Drive ที่ mount แล้ว (ดู RUN_ON_COLAB.md cell mount)
MODELS_ROOT = Path(os.environ.get("MODELS_ROOT", "/content/drive/MyDrive/khanun-affiliate-models"))
MODEL_DIRS = {k: MODELS_ROOT / k for k in ["vton", "background", "upscale", "video", "voice", "vision"]}
for _p in MODEL_DIRS.values():
    try:
        _p.mkdir(parents=True, exist_ok=True)   # สร้างโฟลเดอร์ถ้ายังไม่มี
    except Exception as _e:
        print("[warn] mkdir", _p, _e)
# ให้ HuggingFace แคชน้ำหนักลง Drive แยกตามหน้าที่ (ไม่โหลดซ้ำทุกรอบ)
os.environ.setdefault("HF_HOME", str(MODELS_ROOT / "_hf_cache"))

app = FastAPI(title="Khanun Affiliate — Colab Inference")

# ---------- utils ----------
def _decode(data_uri_or_url: str) -> Image.Image:
    """รับ data-URI base64 (จากแอป) หรือ url -> PIL.Image"""
    s = data_uri_or_url
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
        return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    import urllib.request
    with urllib.request.urlopen(s, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")

def _save_img(img: Image.Image, prefix="img") -> str:
    fn = f"{prefix}_{uuid.uuid4().hex[:10]}.png"; img.save(OUT / fn)
    return fn

def _save_bytes(data: bytes, ext: str, prefix="f") -> str:
    fn = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"; (OUT / fn).write_bytes(data)
    return fn

def _url(req: Request, fn: str) -> str:
    base = str(req.base_url).rstrip("/")
    return f"{base}/file/{fn}"

# ---------- lazy model registry (โหลดทีละตัว กัน VRAM เต็ม) ----------
_LOADED = {"name": None, "obj": None}

def _free():
    _LOADED["obj"] = None; _LOADED["name"] = None
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

def _need(name: str, loader):
    """โหลดโมเดลชื่อ name ถ้ายังไม่ได้โหลด (ปลดตัวเก่าก่อน)"""
    if _LOADED["name"] != name:
        _free()
        print(f"[load] {name} ...")
        _LOADED["obj"] = loader(); _LOADED["name"] = name
    return _LOADED["obj"]

# ---------- loaders ----------
def _load_kontext():
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=DTYPE)
    p.enable_model_cpu_offload()            # ประหยัด VRAM
    return p

def _load_wan_i2v():
    # Wan 2.2 Image-to-Video (diffusers). ถ้า VRAM น้อยใช้รุ่น 5B / GGUF (ดู RUN_ON_COLAB.md)
    from diffusers import WanImageToVideoPipeline
    p = WanImageToVideoPipeline.from_pretrained("Wan-AI/Wan2.2-I2V-A14B-Diffusers", torch_dtype=DTYPE)
    p.enable_model_cpu_offload()
    return p

def _load_upscaler():
    # ใช้ Real-ESRGAN (เบา) แทน clarity-upscaler — คมพอสำหรับงานคลิป
    from RealESRGAN import RealESRGAN
    m = RealESRGAN(DEVICE, scale=2); m.load_weights("weights/RealESRGAN_x2.pth", download=True)
    return m

def _load_vton():
    # CatVTON (เบา ~<8GB) หรือ Kolors-VTON — ดู RUN_ON_COLAB.md (ต้อง clone repo + ckpt)
    from catvton_pipeline import CatVTONPipeline   # helper ที่ repo CatVTON ให้มา
    return CatVTONPipeline(device=DEVICE, dtype=DTYPE)

def _load_f5tts():
    from f5_tts.api import F5TTS
    return F5TTS()

def _load_qwen_vl():
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    m = Qwen2VLForConditionalGeneration.from_pretrained("Qwen/Qwen2-VL-7B-Instruct", torch_dtype=DTYPE, device_map="auto")
    pr = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")
    return (m, pr)

# ---------- node handlers (คืน result สคีมา fal) ----------
def h_kontext(req, inputs):
    imgs = inputs.get("images") or ([inputs["image_url"]] if inputs.get("image_url") else [])
    src = _decode(imgs[0]); prompt = inputs.get("prompt", "")
    pipe = _need("kontext", _load_kontext)
    out = pipe(image=src, prompt=prompt, guidance_scale=3.5,
               num_inference_steps=int(inputs.get("steps", 28))).images[0]
    return {"images": [{"url": _url(req, _save_img(out, "kontext"))}]}

def h_vton(req, inputs):
    person = _decode(inputs["human_image_url"] if "human_image_url" in inputs else inputs["model_image"])
    garment = _decode(inputs["garment_image_url"] if "garment_image_url" in inputs else inputs["garment_image"])
    pipe = _need("vton", _load_vton)
    out = pipe.try_on(person, garment, category=inputs.get("category", "auto"))
    return {"images": [{"url": _url(req, _save_img(out, "vton"))}]}

def h_upscale(req, inputs):
    src = _decode(inputs["image_url"]); factor = int(inputs.get("upscale_factor", 2))
    model = _need("upscale", _load_upscaler)
    out = model.predict(src)
    if factor >= 4:
        out = model.predict(out)
    return {"images": [{"url": _url(req, _save_img(out, "hq"))}]}

def h_video(req, inputs):
    src = _decode(inputs["image"] if "image" in inputs else inputs["image_url"])
    prompt = inputs.get("prompt", "")
    secs = int(float(inputs.get("duration", 10)))
    pipe = _need("wan_i2v", _load_wan_i2v)
    frames = pipe(image=src, prompt=prompt, num_frames=min(secs, 5) * 16,
                  num_inference_steps=int(inputs.get("steps", 30))).frames[0]
    from diffusers.utils import export_to_video
    fn = f"clip_{uuid.uuid4().hex[:10]}.mp4"; export_to_video(frames, str(OUT / fn), fps=16)
    return {"video": {"url": _url(req, fn)}}

def h_tts(req, inputs):
    ip = inputs.get("inputs", inputs); text = ip.get("text", "")
    f5 = _need("f5tts", _load_f5tts)
    ref = os.environ.get("F5_REF_AUDIO", "/content/voice/ref.wav")   # เสียงต้นแบบที่โคลน
    ref_txt = os.environ.get("F5_REF_TEXT", "")
    wav, sr, _ = f5.infer(ref_file=ref, ref_text=ref_txt, gen_text=text)
    import soundfile as sf
    buf = io.BytesIO(); sf.write(buf, wav, sr, format="WAV");
    return {"audio_url": _url(req, _save_bytes(buf.getvalue(), "wav", "vo"))}

def h_vision(req, inputs):
    # อ่านภาพสินค้า -> คำบรรยายไทยสั้น ๆ
    img = _decode(inputs.get("image_url") or (inputs.get("image_urls") or [""])[0])
    m, pr = _need("qwen_vl", _load_qwen_vl)
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
            "text": "บรรยายเสื้อผ้า/สินค้าในรูปสั้น ๆ เป็นภาษาไทย (สี ทรง คอ แขน ชาย เนื้อผ้า)"}]}]
    txt = pr.apply_chat_template(msgs, add_generation_prompt=True)
    batch = pr(text=[txt], images=[img], return_tensors="pt").to(DEVICE)
    out = m.generate(**batch, max_new_tokens=200)
    ans = pr.batch_decode(out[:, batch.input_ids.shape[1]:], skip_special_tokens=True)[0]
    return {"output": ans, "outputs": [ans]}

ROUTER = {
    "kolors-virtual-try-on": h_vton, "fashn/tryon": h_vton, "idm": h_vton,
    "flux-pro/kontext": h_kontext, "kontext": h_kontext,
    "clarity-upscaler": h_upscale, "upscaler": h_upscale,
    "image-to-video": h_video, "wan": h_video,
    "tts": h_tts, "elevenlabs": h_tts,
    "vision": h_vision, "any-llm": h_vision,
}

def _route(model_id: str):
    for k, fn in ROUTER.items():
        if k in model_id:
            return fn
    return None

# ---------- endpoints ----------
@app.get("/health")
def health():
    free = used = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info()
        used = round((total - free) / 1e9, 1)
    return {"ok": True, "device": DEVICE, "loaded": _LOADED["name"], "profile": GPU_PROFILE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None,
            "vram_used_gb": used, "models_root": str(MODELS_ROOT)}


@app.post("/shutdown")
def shutdown():
    """ปล่อยโมเดลทั้งหมด (คืน VRAM) — เรียกจากปุ่ม 'ปิด instance' ในแอป"""
    _free()
    return {"ok": True, "freed": True}


@app.get("/gpu")
def gpu():
    return {"ok": True, "profile": GPU_PROFILE,
            "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None}

@app.get("/file/{fn}")
def get_file(fn: str):
    p = OUT / fn
    return FileResponse(str(p)) if p.is_file() else JSONResponse({"error": "not_found"}, 404)

@app.post("/infer")
async def infer(req: Request):
    body = await req.json()
    model_id = body.get("model_id", ""); inputs = body.get("inputs", {})
    fn = _route(model_id)
    if not fn:
        return JSONResponse({"ok": False, "error": f"no_handler:{model_id}"}, 400)
    try:
        t0 = time.time()
        result = fn(req, inputs)
        print(f"[infer] {model_id} ok {round(time.time()-t0,1)}s")
        return {"ok": True, "result": result}
    except Exception as e:
        import traceback; traceback.print_exc()
        return JSONResponse({"ok": False, "error": str(e)[:400]}, 500)
