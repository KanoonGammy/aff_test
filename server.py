# server.py - Khanun affiliate inference (RunPod) - v9
# v9: PIN โมเดลอ่านชุด (qwen/vision) ไว้ใน VRAM ไม่ให้เขี่ยออก (ใช้ทุกงาน) — เขี่ยตัวอื่นก่อนเสมอ
# v8: cache หลายโมเดลใน VRAM พร้อมกัน (LRU · การ์ด 48GB) — ไม่ต้องโหลดซ้ำตอนสลับโหนด · VRAM ไม่พอ = เขี่ย LRU ออกเอง
# v7: คืนผลเป็น base64 inline (รูป/วิดีโอ) → backend ไม่ต้องดึง /file ซ้ำ (แก้ download_failed/WinError 10060 ตอน Pod คืน URL ภายใน)
# v6: (1) h_vton รับ key ของแอปจริง human_image_url/garment_image_url/model_image  (2) ใส่ทั้งโมเดลลง GPU (48GB การ์ด) เลิก cpu_offload = เร็ว+ใช้ GPU เต็ม  (3) warmup/infer บอกเวลา (วิ)
# v5: video เปลี่ยนเป็น Wan2.2-TI2V-5B (เบา RAM ~16GB) แทน A14B ที่ OOM บน Pod 46GB RAM
# v4: + CatVTON wrapper (h_vton ใส่ชุดจริง บน RunPod ฟรี) — clone repo + AutoMasker(densepose+schp) + pipeline
# v3: persist on /workspace (survives Stop) + /warmup endpoint (preload model inside Pod, avoids 100s proxy 524)
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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32   # CatVTON/SD-inpaint ชอบ fp16
GPU_PROFILE = os.environ.get("GPU_PROFILE", "a100")
CATVTON_DIR = os.path.join(MODELS_ROOT, "CatVTON_repo")

app = FastAPI(title="Khanun Affiliate Inference v9")

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
    # v7: คืนรูปเป็น data-URI (base64) → ส่งกลับใน JSON เลย backend ไม่ต้องดึง /file ซ้ำ
    b = io.BytesIO(); img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")
def _file_uri(path, mime):
    # v7: คืนไฟล์ (เช่น mp4/wav) เป็น data-URI base64
    data = open(path, "rb").read()
    return "data:%s;base64,%s" % (mime, base64.b64encode(data).decode("ascii"))

# v8/v9: cache หลายโมเดลใน VRAM พร้อมกัน (การ์ด 48GB) — LRU · VRAM ไม่พอตอนโหลด = เขี่ยตัวที่ใช้น้อยสุดออก
# v9: PIN โมเดลอ่านชุด (qwen) ไว้ไม่ให้เขี่ยออก เพราะใช้ทุกงาน (เขี่ยตัวอื่นก่อนเสมอ)
from collections import OrderedDict
_CACHE = OrderedDict()   # name -> obj
_PIN = {"qwen"}          # โมเดลที่ pin ไว้ (vision อ่านชุด) — เขี่ยเป็นตัวสุดท้ายเมื่อจำเป็นจริง ๆ
def _free():
    _CACHE.clear(); gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
def _evict_lru():
    # หาตัวที่ "ไม่ pin" เก่าสุดเขี่ยก่อน · ถ้าเหลือแต่ pinned ค่อยเขี่ย pinned
    target = next((n for n in _CACHE if n not in _PIN), None) or next(iter(_CACHE), None)
    if target is not None:
        _CACHE.pop(target); print("[evict]", target, "(เปิดที่ให้โมเดลใหม่)", flush=True)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
def _need(name, loader):
    if name in _CACHE:
        _CACHE.move_to_end(name); return _CACHE[name]     # มีใน VRAM แล้ว → ใช้เลย ไม่โหลดซ้ำ
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
    """การ์ด 48GB (RTX 6000 Ada/A40): ใส่ทั้งโมเดลลง GPU = เร็ว + ใช้ GPU เต็ม · VRAM ไม่พอ → fallback cpu_offload"""
    try:
        return p.to(DEVICE)
    except Exception as e:
        print("[place->cpu_offload]", str(e)[:120], flush=True)
        p.enable_model_cpu_offload(); return p

def _load_kontext():
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained("black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=DTYPE)
    return _place(p)
def _load_wan():
    # v5: ใช้ TI2V-5B (เบา RAM ~16GB / VRAM ~24GB) แทน A14B (~50GB RAM = OOM บน Pod 46GB)
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    mid = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
    vae = AutoencoderKLWan.from_pretrained(mid, subfolder="vae", torch_dtype=torch.float32)   # VAE ต้อง fp32
    p = WanImageToVideoPipeline.from_pretrained(mid, vae=vae, torch_dtype=DTYPE)
    return _place(p)
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

def _load_catvton():
    """โหลด CatVTON: clone repo (โค้ด model/utils) + snapshot น้ำหนัก + สร้าง pipeline + AutoMasker(densepose+schp)"""
    # 1) โค้ด CatVTON (model/, utils.py, densepose/) — ไม่มีบน pip ต้อง clone
    if not os.path.isdir(CATVTON_DIR):
        print("[catvton] clone repo ...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/Zheng-Chong/CatVTON", CATVTON_DIR], check=True)
    if CATVTON_DIR not in sys.path:
        sys.path.insert(0, CATVTON_DIR)
    # 2) เช็ค detectron2 (จำเป็นสำหรับ DensePose ทำ mask) — ถ้าไม่มีบอกชัด ๆ
    try:
        import detectron2  # noqa: F401
    except Exception:
        raise RuntimeError("detectron2 ยังไม่ลง — รัน: pip install 'git+https://github.com/facebookresearch/detectron2.git' (ดู RUNPOD-SNAPSHOT.md ขั้น D)")
    # 3) น้ำหนัก CatVTON (รวม DensePose + SCHP) จาก HF
    from huggingface_hub import snapshot_download
    repo = snapshot_download(repo_id="zhengchong/CatVTON")
    # 4) สร้าง pipeline + masker
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    pipe = CatVTONPipeline(
        base_ckpt="booksforcharlie/stable-diffusion-inpainting",   # SD-inpaint ฐาน (mirror ที่ CatVTON ใช้)
        attn_ckpt=repo, attn_ckpt_version="mix",
        weight_dtype=CAT_DTYPE, device=DEVICE, skip_safety_check=True)
    masker = AutoMasker(densepose_ckpt=os.path.join(repo, "DensePose"),
                        schp_ckpt=os.path.join(repo, "SCHP"), device=DEVICE)
    return (pipe, masker)

# หมวดเสื้อ -> ชนิด mask ของ CatVTON
_CAT_MASK = {"tops": "upper", "upper": "upper", "top": "upper", "shirt": "upper",
             "bottoms": "lower", "lower": "lower", "pants": "lower", "skirt": "lower",
             "dress": "overall", "dresses": "overall", "auto": "overall", "overall": "overall", "full": "overall"}

def h_kontext(req, ip):
    imgs = ip.get("images") or ([ip["image_url"]] if ip.get("image_url") else [])
    out = _need("kontext", _load_kontext)(image=_decode(imgs[0]), prompt=ip.get("prompt", ""),
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
    nframes = min(secs, 5) * 16 + 1   # Wan ต้องการ 4n+1 เฟรม (16*วิ หาร 4 ลงตัว → +1)
    fr = _need("wan", _load_wan)(image=src, prompt=ip.get("prompt", ""),
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
    # v4: prompt เจาะจง + ห้ามเดา/ห้ามซ้ำ (กันหลอน "แขนยาว แขนยาว")
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
    # v6: รับ key ของแอปจริง (Kolors=human_image_url/garment_image_url · fashn=model_image/garment_image)
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
        "persist": PERSIST, "models_root": MODELS_ROOT, "out": OUT, "version": "v9", "pinned": list(_PIN)}

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
