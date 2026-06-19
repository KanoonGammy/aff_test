# server.py - Khanun affiliate inference (RunPod) - v12
# v12 (2026-06-19): แก้ 3 จุดที่ทำให้ RAM ตัน + ภาพหลอน/ไม่คม (ดู log จริง)
#   (1) RAM FIX: เลิก pin qwen + เลิก cpu_offload → ใส่โมเดลใหญ่ทั้งก้อนลง GPU 48GB ทีละตัว
#       (เดิม offload ดันน้ำหนัก ~33GB ลง System RAM → RAM 90% ทั้งที่ VRAM ว่าง 2%)
#   (2) UPSCALE FIX: shim torchvision.transforms.functional_tensor → functional
#       (เดิม Real-ESRGAN พังเงียบ ๆ ทุกครั้ง fallback เป็น lanczos = ภาพไม่เคยถูกอัปจริง)
#   (3) PROMPT: trim ช่องว่าง/จุดท้าย กัน CLIP 77-token ตัดคำสำคัญทิ้งเงียบ ๆ
#   (4) /health รายงาน System RAM ด้วย (เดิมเห็นแค่ VRAM เลย debug ผิดตัว)
# ---- ประวัติเดิม ----
# v11: h_kontext รับ image_urls · kontext/wan ใช้ cpu_offload + wan vae tiling · cache เก็บ pinned+1
# v10: ปิด HF xet (HF_HUB_DISABLE_XET) กัน "Background writer channel closed"
# v9: PIN qwen/vision ใน VRAM · v8: cache หลายโมเดล LRU · v7: คืน base64 inline
# v6: รับ key VTON จริง + ใส่โมเดลลง GPU เต็ม · v5: Wan2.2-TI2V-5B · v4: CatVTON · v3: persist /workspace + /warmup
import base64, io, os, sys, time, uuid, gc, subprocess
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, ImageFilter

PERSIST = "/workspace" if os.path.isdir("/workspace") else os.getcwd()   # /workspace = ถาวร (ไม่โดน Stop ล้าง)
OUT = os.environ.get("OUT_DIR", os.path.join(PERSIST, "aff_outputs"))
MODELS_ROOT = os.environ.get("MODELS_ROOT", os.path.join(PERSIST, "aff_models"))
os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(MODELS_ROOT, "_hf"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # v10: ปิด xet → ดาวน์โหลดแบบคลาสสิก กัน "Background writer channel closed" ตอนดิสก์ตึง
# v12: ให้ allocator คืนหน่วยความจำเป็นชิ้นเล็ก ลด fragmentation ตอนสลับโมเดลใหญ่บน 48GB
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32   # CatVTON/SD-inpaint ชอบ fp16
GPU_PROFILE = os.environ.get("GPU_PROFILE", "a100")
CATVTON_DIR = os.path.join(MODELS_ROOT, "CatVTON_repo")

# v12: ตัวคุมพฤติกรรม (ปรับผ่าน env ได้ ไม่ต้องแก้โค้ด)
#   FORCE_CPU_OFFLOAD=1 → กลับไปใช้ offload (เผื่อย้ายไปการ์ด VRAM น้อย เช่น 4090 24GB)
FORCE_OFFLOAD = os.environ.get("FORCE_CPU_OFFLOAD", "0") == "1"

app = FastAPI(title="Khanun Affiliate Inference v12")

def _safe(name):
    return "".join(c for c in (name or "") if c.isalnum() or c in "-_ ")[:40].strip().replace(" ", "_") or "job"
def _decode(s):
    if s is None:
        raise ValueError("missing image")
    if s.startswith("data:"):
        s = s.split(",", 1)[1]; return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    import urllib.request
    with urllib.request.urlopen(s, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")
def _clean_prompt(p):
    # v12: ลดความเสี่ยง CLIP 77-token ตัดคำท้ายทิ้งเงียบ ๆ — ตัดช่องว่าง/จุดลอยท้าย + ยุบ space ซ้ำ
    p = " ".join((p or "").split())
    return p.strip(" .")
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
def _img_uri(img):
    b = io.BytesIO(); img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")
def _file_uri(path, mime):
    data = open(path, "rb").read()
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))

def _ram_gb():
    # v12: อ่าน System RAM (ตัวที่ตันจริง) — ใช้ psutil ถ้ามี ไม่งั้นอ่าน /proc/meminfo
    try:
        import psutil; m = psutil.virtual_memory()
        return {"used_gb": round(m.used / 1e9, 1), "total_gb": round(m.total / 1e9, 1), "percent": m.percent}
    except Exception:
        try:
            info = {}
            for ln in open("/proc/meminfo"):
                k, v = ln.split(":"); info[k] = int(v.split()[0]) * 1024
            total = info["MemTotal"]; avail = info.get("MemAvailable", info["MemFree"]); used = total - avail
            return {"used_gb": round(used / 1e9, 1), "total_gb": round(total / 1e9, 1), "percent": round(used / total * 100, 1)}
        except Exception:
            return None

# v8/v9: cache หลายโมเดลใน VRAM พร้อมกัน · v12: เก็บ "โมเดลใหญ่ทีละตัว" บน GPU 48GB เต็ม (เร็ว + RAM ว่าง)
from collections import OrderedDict
_CACHE = OrderedDict()   # name -> obj
# v12: pin เฉพาะตัวเล็กที่ใช้บ่อย (upscale ~ไม่กิน VRAM) — เลิก pin qwen (16GB) ที่บีบให้ kontext ต้อง offload
#      vision กับ bg/video เป็นคนละสเต็ปในไพป์ไลน์ ไม่รันพร้อมกัน → ไม่ต้องค้าง qwen ไว้
_PIN = {"upscale"}
def _free():
    _CACHE.clear(); gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
def _evict_lru():
    target = next((n for n in _CACHE if n not in _PIN), None) or next(iter(_CACHE), None)
    if target is not None:
        _CACHE.pop(target); print("[evict]", target, flush=True)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
def _need(name, loader):
    if name in _CACHE:
        _CACHE.move_to_end(name); return _CACHE[name]     # มีใน VRAM แล้ว → ใช้เลย ไม่โหลดซ้ำ
    # ก่อนโหลดตัวใหม่ → เขี่ย non-pinned ออกหมด (เก็บแค่ pinned + ตัวนี้) ให้โมเดลใหญ่ได้ VRAM เต็ม
    if name not in _PIN:
        for n in [k for k in list(_CACHE) if k not in _PIN]:
            _CACHE.pop(n); print("[evict]", n, "(เปิดที่ให้", name, ")", flush=True)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
    while True:
        try:
            print("[load]", name, flush=True); t0 = time.time()
            obj = loader()
            print("[load]", name, "done", round(time.time() - t0, 1), "s", flush=True)
            _CACHE[name] = obj; _CACHE.move_to_end(name); return obj
        except torch.cuda.OutOfMemoryError:
            if not _CACHE: raise                            # ไม่มีอะไรให้เขี่ยแล้ว = ใหญ่เกินจริง
            print("[load]", name, "OOM → เขี่ยตัวเก่า", flush=True); _evict_lru()

def _place(p):
    """v12: การ์ด 48GB → ใส่ทั้งโมเดลลง GPU = เร็ว + RAM ว่าง · VRAM ไม่พอ/บังคับ → cpu_offload
    (เลิกใช้ offload เป็นค่าตั้งต้น เพราะ offload ดันน้ำหนักลง System RAM = RAM ตัน 90% บน log จริง)"""
    if FORCE_OFFLOAD:
        print("[place] FORCE_CPU_OFFLOAD=1 → enable_model_cpu_offload", flush=True)
        p.enable_model_cpu_offload(); return p
    try:
        return p.to(DEVICE)
    except Exception as e:
        print("[place->cpu_offload]", str(e)[:120], flush=True)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        p.enable_model_cpu_offload(); return p

def _load_kontext():
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=DTYPE)
    return _place(p)   # v12: ลง GPU เต็ม (เดิม v11 enable_model_cpu_offload() = ช้า 5.5s/it + RAM ตัน)
def _load_wan():
    # v5: TI2V-5B (เบา) · v12: ลง GPU เต็ม + คง VAE tiling กัน OOM ตอน decode
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)   # VAE ต้อง fp32
    p = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=DTYPE)
    try: p.vae.enable_tiling()     # ลด VRAM ตอน VAE decode
    except Exception: pass
    return _place(p)   # v12: ลง GPU เต็ม (เดิม offload)
def _load_upscaler():
    # v12 FIX: basicsr import torchvision.transforms.functional_tensor (ถูกลบใน torchvision>=0.17)
    #   → shim ชี้ไป functional ก่อน import realesrgan = Real-ESRGAN ทำงานจริง (ไม่ fallback lanczos)
    #   แทน sed ใน RUNPOD-SNAPSHOT ขั้น 0.7 ที่มักลืมรัน
    try:
        import torchvision.transforms.functional_tensor  # noqa: F401  (มี = ไม่ต้อง shim)
    except ModuleNotFoundError:
        import torchvision.transforms.functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
        print("[upscale] shim functional_tensor -> functional", flush=True)
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

def _load_catvton():
    """โหลด CatVTON: clone repo (โค้ด model/utils) + snapshot น้ำหนัก + สร้าง pipeline + AutoMasker(densepose+schp)"""
    if not os.path.isdir(CATVTON_DIR):
        print("[catvton] clone repo ...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Zheng-Chong/CatVTON", CATVTON_DIR], check=True)
    if CATVTON_DIR not in sys.path:
        sys.path.insert(0, CATVTON_DIR)
    try:
        import detectron2  # noqa: F401
    except Exception:
        raise RuntimeError("detectron2 ยังไม่ลง — รัน: pip install 'git+https://github.com/facebookresearch/detectron2.git' (ดู RUNPOD-SNAPSHOT.md ขั้น D)")
    from huggingface_hub import snapshot_download
    repo = snapshot_download(repo_id="zhengchong/CatVTON")
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    pipe = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",
        attn_ckpt=repo, attn_ckpt_version="mix",
        weight_dtype=CAT_DTYPE, device=DEVICE, skip_safety_check=True)
    masker = AutoMasker(densepose_ckpt=os.path.join(repo, "DensePose"),
                        schp_ckpt=os.path.join(repo, "SCHP"), device=DEVICE)
    return (pipe, masker)

_CAT_MASK = {"tops": "upper", "upper": "upper", "top": "upper", "shirt": "upper",
             "bottoms": "lower", "lower": "lower", "pants": "lower", "skirt": "lower",
             "dress": "overall", "dresses": "overall", "auto": "overall", "overall": "overall", "full": "overall"}

def h_kontext(req, ip):
    imgs = ip.get("images") or ip.get("image_urls") or ([ip["image_url"]] if ip.get("image_url") else [])
    out = _need("kontext", _load_kontext)(image=_decode(imgs[0]), prompt=_clean_prompt(ip.get("prompt", "")),
        guidance_scale=3.5, num_inference_steps=int(ip.get("steps", 28))).images[0]
    _save_img(out, ip, "kontext"); return {"images": [{"url": _img_uri(out)}]}
def h_upscale(req, ip):
    img = _decode(ip["image_url"]); f = int(ip.get("upscale_factor", 2))
    try:
        import numpy as np
        arr, _ = _need("upscale", _load_upscaler).enhance(np.array(img)[:, :, ::-1], outscale=f)
        out = Image.fromarray(arr[:, :, ::-1])
    except Exception as e:
        print("[upscale -> lanczos]", str(e)[:120], flush=True)
        out = img.resize((img.width * f, img.height * f), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 130))
    _save_img(out, ip, "hq"); return {"images": [{"url": _img_uri(out)}]}
def h_video(req, ip):
    src = _decode(ip.get("image") or ip.get("image_url")); secs = int(float(ip.get("duration", 10)))
    nframes = min(secs, 5) * 16 + 1   # Wan ต้องการ 4n+1 เฟรม
    fr = _need("wan", _load_wan)(image=src, prompt=_clean_prompt(ip.get("prompt", "")),
        num_frames=nframes, num_inference_steps=int(ip.get("steps", 30))).frames[0]
    from diffusers.utils import export_to_video
    d = _job_dir(ip); fp = os.path.join(d, "clip_%s.mp4" % uuid.uuid4().hex[:8]); export_to_video(fr, fp, fps=16)
    return {"video": {"url": _file_uri(fp, "video/mp4")}}
def h_tts(req, ip):
    dd = ip.get("inputs", ip)
    wav, sr, _ = _need("f5", _load_f5).infer(ref_file=os.environ.get("F5_REF_AUDIO", os.path.join(PERSIST, "voice/ref.wav")),
        ref_text=os.environ.get("F5_REF_TEXT", ""), gen_text=dd.get("text", ""))
    import soundfile as sf
    b = io.BytesIO(); sf.write(b, wav, sr, format="WAV")
    _save_bytes(b.getvalue(), "wav", ip, "vo")
    return {"audio_url": "data:audio/wav;base64," + base64.b64encode(b.getvalue()).decode("ascii")}
def h_vision(req, ip):
    img = _decode(ip.get("image_url") or (ip.get("image_urls") or [""])[0])
    m, pr = _need("qwen", _load_qwen)
    instr = ("บรรยายชุด/สินค้าในภาพ สั้น กระชับ เป็นภาษาไทย ระบุเฉพาะที่เห็นชัด: "
             "ประเภท, สี, ทรงคอ, ความยาวแขน, ความยาวชาย, ลักษณะผ้า. "
             "ถ้าไม่ชัดให้เขียน 'ไม่แน่ใจ'. ห้ามเดา ห้ามพูดซ้ำคำเดิม ตอบ 1-2 ประโยค.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    t = pr.apply_chat_template(msgs, add_generation_prompt=True)
    b = pr(text=[t], images=[img], return_tensors="pt").to(m.device)
    o = m.generate(**b, max_new_tokens=120, repetition_penalty=1.3, do_sample=False)
    ans = pr.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return {"output": ans, "outputs": [ans]}
def h_vton(req, ip):
    """ใส่ชุดบนนางแบบด้วย CatVTON (ฟรี บน RunPod)"""
    person = _decode(ip.get("person") or ip.get("human_image_url") or ip.get("model_image") or ip.get("image_url"))
    garment = _decode(ip.get("garment") or ip.get("garment_image_url") or ip.get("garment_image") or ip.get("cloth") or ip.get("product"))
    cat = _CAT_MASK.get(str(ip.get("category") or "auto").lower(), "overall")
    pipe, masker = _need("catvton", _load_catvton)
    from utils import resize_and_crop, resize_and_padding
    W, H = 768, 1024
    person_r = resize_and_crop(person, (W, H))
    garment_r = resize_and_padding(garment, (W, H))
    mask = masker(person_r, cat)["mask"]
    gen = torch.Generator(device=DEVICE).manual_seed(int(ip.get("seed", 42)))
    out = pipe(image=person_r, condition_image=garment_r, mask=mask,
               num_inference_steps=int(ip.get("steps", 50)),
               guidance_scale=float(ip.get("guidance", 2.5)),
               height=H, width=W, generator=gen)[0]
    _save_img(out, ip, "tryon"); return {"images": [{"url": _img_uri(out)}]}

ROUTER = {"kolors-virtual-try-on": h_vton, "fashn/tryon": h_vton, "tryon": h_vton, "idm": h_vton, "catvton": h_vton,
    "flux-pro/kontext": h_kontext, "kontext": h_kontext, "clarity-upscaler": h_upscale, "upscaler": h_upscale,
    "image-to-video": h_video, "wan": h_video, "kling-video": h_video, "hailuo": h_video,
    "tts": h_tts, "elevenlabs": h_tts, "vision": h_vision, "any-llm": h_vision}
WARM = {"bg": ("kontext", _load_kontext), "video": ("wan", _load_wan), "upscale": ("upscale", _load_upscaler),
    "vision": ("qwen", _load_qwen), "voice": ("f5", _load_f5), "vton": ("catvton", _load_catvton)}

@app.get("/health")
def health():
    used = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info(); used = round((total - free) / 1e9, 1)
    return {"ok": True, "device": DEVICE, "loaded": list(_CACHE.keys()), "profile": GPU_PROFILE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "vram_used_gb": used,
        "ram": _ram_gb(),   # v12: ดู System RAM ที่ตันจริง
        "offload": FORCE_OFFLOAD, "persist": PERSIST, "models_root": MODELS_ROOT, "out": OUT,
        "version": "v12", "pinned": list(_PIN)}

@app.post("/warmup")
async def warmup(req: Request):
    body = await req.json(); node = body.get("node", "")
    if node not in WARM: return JSONResponse({"ok": False, "error": "warmup node: bg|video|upscale|vision|voice|vton"}, 400)
    name, loader = WARM[node]
    try:
        t0 = time.time(); _need(name, loader); el = round(time.time() - t0, 1)
        return {"ok": True, "loaded": name, "seconds": el, "msg": f"model in VRAM ({el}s) — app /infer will be fast now"}
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
