# server.py - Khanun affiliate inference (RunPod) - v13
# v13 (2026-06-19): โมเดล "ประจำโหนด" ครบทุกโหนด + ระบบลบโมเดลที่ไม่ใช้บน disk
#   vision  : Qwen2.5-VL-7B-Instruct  (แก้หลอน · repetition_penalty 1.1)
#   vton    : CatVTON (ตั้งต้น ใช้ได้จริง) | VTON_MODEL=idm (ยังเป็นโครง → fallback CatVTON อัตโนมัติ)
#   bg      : FLUX.1-Kontext-dev      (ลง GPU เต็ม เร็ว)
#   video   : Wan2.1-I2V-14B-720P (ตั้งต้น · ลง GPU เต็ม 48GB ไม่ offload) | WAN_MODEL=5B | A14B
#             v13.2 เลือกวินาที+แพลตฟอร์ม · v13.3 video registry (คุม fps/flow_shift ตามรุ่น)
#   upscale : Real-ESRGAN x4plus + GFPGAN face enhance
#   voice   : F5-TTS-THAI
#   DISK: GET /disk · POST /disk/prune {"confirm":true}
# v12: RAM fix (เลิก pin qwen + เลิก cpu_offload) · upscale shim · /health รายงาน RAM
# v11..v3: kontext image_urls · ปิด HF xet · pin qwen · cache LRU · base64 inline · CatVTON · Wan · persist
import base64, io, os, sys, time, uuid, gc, shutil, subprocess
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image, ImageFilter

PERSIST = "/workspace" if os.path.isdir("/workspace") else os.getcwd()
OUT = os.environ.get("OUT_DIR", os.path.join(PERSIST, "aff_outputs"))
MODELS_ROOT = os.environ.get("MODELS_ROOT", os.path.join(PERSIST, "aff_models"))
os.makedirs(OUT, exist_ok=True); os.makedirs(MODELS_ROOT, exist_ok=True)
os.environ.setdefault("HF_HOME", os.path.join(MODELS_ROOT, "_hf"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
CAT_DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
GPU_PROFILE = os.environ.get("GPU_PROFILE", "a100")
CATVTON_DIR = os.path.join(MODELS_ROOT, "CatVTON_repo")
IDM_DIR = os.path.join(MODELS_ROOT, "IDM-VTON_repo")
FORCE_OFFLOAD = os.environ.get("FORCE_CPU_OFFLOAD", "0") == "1"

# ===== โมเดลประจำโหนด (เปลี่ยนผ่าน env ได้) =====
VISION_MODEL = os.environ.get("VISION_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")
KONTEXT_MODEL = os.environ.get("KONTEXT_MODEL", "black-forest-labs/FLUX.1-Kontext-dev")
VTON_MODEL = os.environ.get("VTON_MODEL", "catvton").lower()   # default catvton (ใช้ได้จริง) · idm ยังเป็นโครง=fallback
FACE_ENHANCE = os.environ.get("FACE_ENHANCE", "1") == "1"

# ===== วิดีโอ: registry เลือกรุ่นได้ (คุม repo/fps/offload/flow_shift ให้ถูกตามรุ่น) =====
#   key -> (repo, fps_native, ต้อง_offload?, flow_shift)
WAN_CFG = {
    "WAN21": ("Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", 16, False, 5.0),   # ★ default: 14B เดี่ยว ~38-40GB ลงเต็ม 48GB · 16fps · flow_shift 5.0(720p)
    "5B":    ("Wan-AI/Wan2.2-TI2V-5B-Diffusers",      24, False, None),  # เบา/เร็ว · 24fps
    "A14B":  ("Wan-AI/Wan2.2-I2V-A14B-Diffusers",     24, True,  None),  # ดีสุดแต่ ~56GB → offload (RAM หนัก เสี่ยง OOM)
}
WAN_MODEL_KEY = os.environ.get("WAN_MODEL", "WAN21").upper()
if WAN_MODEL_KEY not in WAN_CFG: WAN_MODEL_KEY = "WAN21"
WAN_MODEL, WAN_FPS, WAN_OFFLOAD, WAN_FLOWSHIFT = WAN_CFG[WAN_MODEL_KEY]

# ===== วิดีโอ: เลือกวินาที + แพลตฟอร์ม (สัดส่วน) =====
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "10"))         # เพดานความยาว
DEFAULT_PLATFORM = os.environ.get("DEFAULT_PLATFORM", "tiktok").lower()
# ขนาด gen ~720p (หาร 16 ลงตัว) · อิงตาราง docs/hyperframes-reference.md (TikTok/Reels/Shorts=9:16, YouTube=16:9, Feed=1:1)
PLATFORM = {
    "tiktok": (720, 1280), "shopee": (720, 1280), "reels": (720, 1280), "shorts": (720, 1280),
    "9:16": (720, 1280), "vertical": (720, 1280), "portrait": (720, 1280),
    "youtube": (1280, 720), "16:9": (1280, 720), "landscape": (1280, 720),
    "feed": (720, 720), "1:1": (720, 720), "square": (720, 720),
}
VIDEO_NEG = os.environ.get("VIDEO_NEG",
    "blurry, low quality, low resolution, jpeg artifacts, flickering, jitter, ghosting, "
    "warping, deformed, distorted, extra fingers, extra limbs, watermark, text, "
    "oversaturated, cartoon, 3d, cgi, slow motion, shaky cam")
# โมเดลฐานของ CatVTON / IDM
SD_INPAINT_BASE = "booksforcharlie/stable-diffusion-inpainting"
IDM_REPO = "yisol/IDM-VTON"

app = FastAPI(title="Khanun Affiliate Inference v13")

# ---------- utils ----------
def _safe(name):
    return "".join(c for c in (name or "") if c.isalnum() or c in "-_ ")[:40].strip().replace(" ", "_") or "job"
def _decode(s):
    if s is None: raise ValueError("missing image")
    if s.startswith("data:"):
        s = s.split(",", 1)[1]; return Image.open(io.BytesIO(base64.b64decode(s))).convert("RGB")
    import urllib.request
    with urllib.request.urlopen(s, timeout=60) as r:
        return Image.open(io.BytesIO(r.read())).convert("RGB")
def _clean_prompt(p):
    return " ".join((p or "").split()).strip(" .")
def _job_dir(ip):
    d = os.path.join(OUT, _safe(ip.get("job") or ip.get("product"))); os.makedirs(d, exist_ok=True); return d
def _save_img(img, ip, p="img"):
    d = _job_dir(ip); fn = "%s_%s.png" % (p, uuid.uuid4().hex[:8]); img.save(os.path.join(d, fn))
    return os.path.relpath(os.path.join(d, fn), OUT)
def _save_bytes(data, ext, ip, p="f"):
    d = _job_dir(ip); fn = "%s_%s.%s" % (p, uuid.uuid4().hex[:8], ext); open(os.path.join(d, fn), "wb").write(data)
    return os.path.relpath(os.path.join(d, fn), OUT)
def _img_uri(img):
    b = io.BytesIO(); img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode("ascii")
def _file_uri(path, mime):
    return "data:%s;base64,%s" % (mime, base64.b64encode(open(path, "rb").read()).decode("ascii"))
def _ram_gb():
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

# ---------- model cache (โมเดลใหญ่ทีละตัวบน GPU 48GB) ----------
from collections import OrderedDict
_CACHE = OrderedDict()
_PIN = {"upscale"}
def _free():
    _CACHE.clear(); gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
def _evict_lru():
    target = next((n for n in _CACHE if n not in _PIN), None) or next(iter(_CACHE), None)
    if target is not None:
        _CACHE.pop(target); print("[evict]", target, flush=True); gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
def _need(name, loader):
    if name in _CACHE:
        _CACHE.move_to_end(name); return _CACHE[name]
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
            if not _CACHE: raise
            print("[load]", name, "OOM → เขี่ยตัวเก่า", flush=True); _evict_lru()

def _place(p):
    """48GB → ใส่ทั้งโมเดลลง GPU = เร็ว + RAM ว่าง · ไม่พอ/บังคับ → cpu_offload"""
    if FORCE_OFFLOAD:
        print("[place] FORCE_CPU_OFFLOAD → enable_model_cpu_offload", flush=True)
        p.enable_model_cpu_offload(); return p
    try:
        return p.to(DEVICE)
    except Exception as e:
        print("[place->cpu_offload]", str(e)[:120], flush=True)
        gc.collect()
        if DEVICE == "cuda": torch.cuda.empty_cache()
        p.enable_model_cpu_offload(); return p

# ---------- loaders ----------
def _load_kontext():
    from diffusers import FluxKontextPipeline
    p = FluxKontextPipeline.from_pretrained(KONTEXT_MODEL, torch_dtype=DTYPE)
    return _place(p)

def _load_wan():
    from diffusers import WanImageToVideoPipeline, AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(WAN_MODEL, subfolder="vae", torch_dtype=torch.float32)
    p = WanImageToVideoPipeline.from_pretrained(WAN_MODEL, vae=vae, torch_dtype=DTYPE)
    if WAN_FLOWSHIFT:   # Wan2.1 720p ต้องตั้ง flow_shift=5.0 ให้มอชชั่นถูกต้อง (ไม่งั้นภาพไหล/ช้า)
        try:
            from diffusers import UniPCMultistepScheduler
            p.scheduler = UniPCMultistepScheduler.from_config(p.scheduler.config, flow_shift=WAN_FLOWSHIFT)
        except Exception as e:
            print("[wan] flow_shift set fail:", str(e)[:80], flush=True)
    try: p.vae.enable_tiling()
    except Exception: pass
    if WAN_OFFLOAD:
        print("[wan]", WAN_MODEL_KEY, "→ enable_model_cpu_offload (ใหญ่เกิน 48GB · RAM หนัก)", flush=True)
        p.enable_model_cpu_offload(); return p
    return _place(p)   # WAN21 / 5B ลง GPU เต็มได้

def _load_upscaler():
    # v12 FIX: basicsr import torchvision.transforms.functional_tensor (ถูกลบใน torchvision>=0.17) → shim
    try:
        import torchvision.transforms.functional_tensor  # noqa
    except ModuleNotFoundError:
        import torchvision.transforms.functional as _F
        sys.modules["torchvision.transforms.functional_tensor"] = _F
        print("[upscale] shim functional_tensor -> functional", flush=True)
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import urllib.request
    w = os.path.join(MODELS_ROOT, "RealESRGAN_x4plus.pth")
    if not os.path.exists(w):
        urllib.request.urlretrieve("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth", w)
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    up = RealESRGANer(scale=4, model_path=w, model=model, half=(DEVICE == "cuda"))
    face = None
    if FACE_ENHANCE:
        try:
            from gfpgan import GFPGANer
            gw = os.path.join(MODELS_ROOT, "GFPGANv1.4.pth")
            if not os.path.exists(gw):
                urllib.request.urlretrieve("https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth", gw)
            face = GFPGANer(model_path=gw, upscale=4, arch="clean", channel_multiplier=2, bg_upsampler=up)
        except Exception as e:
            print("[upscale] GFPGAN ปิด (ลง gfpgan ไม่สำเร็จ):", str(e)[:100], flush=True)
    return (up, face)

def _load_qwen():
    from transformers import AutoProcessor
    try:
        from transformers import Qwen2_5_VLForConditionalGeneration as _VL
    except Exception:
        from transformers import Qwen2VLForConditionalGeneration as _VL
        print("[qwen] transformers เก่า → ใช้ Qwen2-VL (อัป transformers>=4.49 เพื่อใช้ 2.5-VL)", flush=True)
    m = _VL.from_pretrained(VISION_MODEL, torch_dtype=DTYPE, device_map="auto")
    return (m, AutoProcessor.from_pretrained(VISION_MODEL))

def _load_f5():
    from f5_tts.api import F5TTS
    return F5TTS()

def _load_catvton():
    if not os.path.isdir(CATVTON_DIR):
        print("[catvton] clone repo ...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/Zheng-Chong/CatVTON", CATVTON_DIR], check=True)
    if CATVTON_DIR not in sys.path: sys.path.insert(0, CATVTON_DIR)
    try:
        import detectron2  # noqa
    except Exception:
        raise RuntimeError("detectron2/fvcore ยังไม่ลง — ดู requirements.txt หัวไฟล์ (manual installs)")
    from huggingface_hub import snapshot_download
    repo = snapshot_download(repo_id="zhengchong/CatVTON")
    from model.pipeline import CatVTONPipeline
    from model.cloth_masker import AutoMasker
    pipe = CatVTONPipeline(base_ckpt=SD_INPAINT_BASE, attn_ckpt=repo, attn_ckpt_version="mix",
                           weight_dtype=CAT_DTYPE, device=DEVICE, skip_safety_check=True)
    masker = AutoMasker(densepose_ckpt=os.path.join(repo, "DensePose"),
                        schp_ckpt=os.path.join(repo, "SCHP"), device=DEVICE)
    return ("catvton", pipe, masker)

def _load_idm():
    """IDM-VTON: clone repo + โหลดน้ำหนัก (inference ยังเป็นโครง → h_vton จะ fallback CatVTON ให้)"""
    if not os.path.isdir(IDM_DIR):
        print("[idm] clone repo ...", flush=True)
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/yisol/IDM-VTON", IDM_DIR], check=True)
    if IDM_DIR not in sys.path: sys.path.insert(0, IDM_DIR)
    from huggingface_hub import snapshot_download
    repo = snapshot_download(repo_id=IDM_REPO)
    return ("idm", repo, None)

def _load_vton():
    """เลือก VTON ตาม env: catvton (ตั้งต้น) | idm (ยังเป็นโครง → h_vton fallback CatVTON)"""
    if VTON_MODEL == "idm":
        try:
            return _load_idm()
        except Exception as e:
            print("[vton] IDM โหลดไม่ได้ → CatVTON:", str(e)[:140], flush=True)
    return _load_catvton()

_CAT_MASK = {"tops": "upper", "upper": "upper", "top": "upper", "shirt": "upper",
             "bottoms": "lower", "lower": "lower", "pants": "lower", "skirt": "lower",
             "dress": "overall", "dresses": "overall", "auto": "overall", "overall": "overall", "full": "overall"}

# ---------- handlers ----------
def h_kontext(req, ip):
    imgs = ip.get("images") or ip.get("image_urls") or ([ip["image_url"]] if ip.get("image_url") else [])
    out = _need("kontext", _load_kontext)(image=_decode(imgs[0]), prompt=_clean_prompt(ip.get("prompt", "")),
        guidance_scale=float(ip.get("guidance", 3.5)), num_inference_steps=int(ip.get("steps", 30))).images[0]
    _save_img(out, ip, "kontext"); return {"images": [{"url": _img_uri(out)}]}

def h_upscale(req, ip):
    img = _decode(ip["image_url"]); f = int(ip.get("upscale_factor", 4))
    try:
        import numpy as np
        up, face = _need("upscale", _load_upscaler)
        bgr = np.array(img)[:, :, ::-1]
        if face is not None:
            _, _, out_bgr = face.enhance(bgr, has_aligned=False, only_center_face=False, paste_back=True)
            arr = out_bgr
        else:
            arr, _ = up.enhance(bgr, outscale=f)
        out = Image.fromarray(arr[:, :, ::-1])
    except Exception as e:
        print("[upscale -> lanczos]", str(e)[:140], flush=True)
        out = img.resize((img.width * f, img.height * f), Image.LANCZOS).filter(ImageFilter.UnsharpMask(2, 130))
    _save_img(out, ip, "hq"); return {"images": [{"url": _img_uri(out)}]}

def h_video(req, ip):
    # v13.2/3: เลือกวินาที + แพลตฟอร์ม · คุณภาพ 720p + fps/flow_shift ตามรุ่น (registry) + negative prompt
    src = _decode(ip.get("image") or ip.get("image_url"))
    secs = float(ip.get("seconds", ip.get("duration", 5)))
    secs = max(2.0, min(secs, MAX_SECONDS))
    fps = int(ip.get("fps", WAN_FPS))                        # fps ตามรุ่น (Wan2.1=16, Wan2.2-5B=24)
    want = int(round(secs * fps))
    nframes = max(17, (want // 4) * 4 + 1)                   # Wan ต้องการ 4n+1
    if secs > 5 and WAN_MODEL_KEY != "WAN21": print("[video] เตือน: >5 วิ บนรุ่นเล็กอาจหลุดโฟกัส", flush=True)
    plat = str(ip.get("platform") or ip.get("aspect") or DEFAULT_PLATFORM).lower()
    if ip.get("width") and ip.get("height"):
        W, H = int(ip["width"]), int(ip["height"])
    elif plat in PLATFORM:
        W, H = PLATFORM[plat]
    else:
        iw, ih = src.size; t = 1280
        if ih >= iw: H, W = t, int(round(iw / ih * t))
        else:        W, H = t, int(round(ih / iw * t))
        W = max(256, (W // 16) * 16); H = max(256, (H // 16) * 16)
    fr = _need("wan", _load_wan)(image=src, prompt=_clean_prompt(ip.get("prompt", "")),
        negative_prompt=ip.get("negative_prompt", VIDEO_NEG),
        height=H, width=W, num_frames=nframes,
        guidance_scale=float(ip.get("guidance", 5.0)),
        num_inference_steps=int(ip.get("steps", 40))).frames[0]
    from diffusers.utils import export_to_video
    d = _job_dir(ip); fp = os.path.join(d, "clip_%s.mp4" % uuid.uuid4().hex[:8]); export_to_video(fr, fp, fps=fps)
    print("[video]", WAN_MODEL_KEY, plat, "%dx%d" % (W, H), nframes, "frames", fps, "fps", round(secs, 1), "s", flush=True)
    return {"video": {"url": _file_uri(fp, "video/mp4"), "model": WAN_MODEL_KEY, "platform": plat, "w": W, "h": H, "seconds": round(secs, 1)}}

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
    instr = ("Describe ONLY what is clearly visible in this clothing/product photo. "
             "Report: type, color, neckline, sleeve length, hem length, fabric look. "
             "If a detail is unclear, write 'ไม่แน่ใจ'. Do NOT guess or invent. "
             "Answer in Thai, 1-2 concise sentences.")
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": instr}]}]
    t = pr.apply_chat_template(msgs, add_generation_prompt=True)
    b = pr(text=[t], images=[img], return_tensors="pt").to(m.device)
    o = m.generate(**b, max_new_tokens=110, repetition_penalty=1.1, do_sample=False)
    ans = pr.batch_decode(o[:, b.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
    return {"output": ans, "outputs": [ans]}

# ---------- VTON: CatVTON (ใช้จริง) + fallback อัตโนมัติเมื่อเลือก idm ----------
def _run_catvton(pipe, masker, person, garment, cat, ip):
    from utils import resize_and_crop, resize_and_padding
    W, H = 768, 1024
    person_r = resize_and_crop(person, (W, H)); garment_r = resize_and_padding(garment, (W, H))
    mask = masker(person_r, cat)["mask"]
    gen = torch.Generator(device=DEVICE).manual_seed(int(ip.get("seed", 42)))
    return pipe(image=person_r, condition_image=garment_r, mask=mask,
                num_inference_steps=int(ip.get("steps", 50)), guidance_scale=float(ip.get("guidance", 2.5)),
                height=H, width=W, generator=gen)[0]

def _force_catvton():
    _CACHE.pop("vton", None); gc.collect()
    if DEVICE == "cuda": torch.cuda.empty_cache()
    eng = _load_catvton(); _CACHE["vton"] = eng; return eng

def h_vton(req, ip):
    person = _decode(ip.get("person") or ip.get("human_image_url") or ip.get("model_image") or ip.get("image_url"))
    garment = _decode(ip.get("garment") or ip.get("garment_image_url") or ip.get("garment_image") or ip.get("cloth") or ip.get("product"))
    cat = _CAT_MASK.get(str(ip.get("category") or "auto").lower(), "overall")
    engine_t = _need("vton", _load_vton)
    if engine_t[0] != "catvton":
        # IDM ยังไม่รองรับ inference → fallback CatVTON อัตโนมัติ (เลิก 500 NotImplementedError)
        print("[vton] IDM inference ยังไม่พร้อม → ใช้ CatVTON แทน", flush=True)
        engine_t = _force_catvton()
    out = _run_catvton(engine_t[1], engine_t[2], person, garment, cat, ip)
    _save_img(out, ip, "tryon"); return {"images": [{"url": _img_uri(out)}]}

# ---------- disk management ----------
def _keep_repos():
    keep = {KONTEXT_MODEL, VISION_MODEL, WAN_MODEL, "zhengchong/CatVTON", SD_INPAINT_BASE}
    if VTON_MODEL == "idm": keep.add(IDM_REPO)
    return keep
def _hub_dir():
    return os.path.join(os.environ["HF_HOME"], "hub")
def _repo_to_cache(r):
    return "models--" + r.replace("/", "--")
def _dir_size(p):
    tot = 0
    for root, _, files in os.walk(p):
        for f in files:
            try: tot += os.path.getsize(os.path.join(root, f))
            except OSError: pass
    return tot
def _scan_disk():
    keep_cache = {_repo_to_cache(r) for r in _keep_repos()}
    items = []; hub = _hub_dir()
    if os.path.isdir(hub):
        for d in os.listdir(hub):
            if not d.startswith("models--"): continue
            full = os.path.join(hub, d); size = _dir_size(full)
            items.append({"name": d.replace("models--", "").replace("--", "/"),
                          "dir": full, "size_gb": round(size / 1e9, 2), "keep": d in keep_cache})
    strays = []
    for f in os.listdir(MODELS_ROOT):
        fp = os.path.join(MODELS_ROOT, f)
        if os.path.isfile(fp) and f.endswith((".pth", ".safetensors", ".bin")):
            used = f in ("RealESRGAN_x4plus.pth", "GFPGANv1.4.pth")
            strays.append({"file": f, "size_gb": round(os.path.getsize(fp) / 1e9, 2), "keep": used, "path": fp})
    return items, strays

@app.get("/disk")
def disk():
    items, strays = _scan_disk()
    used = round(sum(i["size_gb"] for i in items if i["keep"]) + sum(s["size_gb"] for s in strays if s["keep"]), 2)
    removable = round(sum(i["size_gb"] for i in items if not i["keep"]) + sum(s["size_gb"] for s in strays if not s["keep"]), 2)
    return {"ok": True, "keep_repos": sorted(_keep_repos()), "models": items, "stray_files": strays,
            "used_gb_keep": used, "removable_gb": removable,
            "hint": "ลบของที่ keep=false ด้วย POST /disk/prune {\"confirm\":true}"}

@app.post("/disk/prune")
async def disk_prune(req: Request):
    body = await req.json() if req.headers.get("content-length") else {}
    if not body.get("confirm"):
        return JSONResponse({"ok": False, "error": "ใส่ {\"confirm\":true} เพื่อยืนยันการลบ (ดูรายการก่อนที่ GET /disk)"}, 400)
    items, strays = _scan_disk(); freed = 0.0; removed = []
    for it in items:
        if not it["keep"]:
            try:
                shutil.rmtree(it["dir"]); freed += it["size_gb"]; removed.append(it["name"])
            except Exception as e:
                print("[prune] fail", it["name"], str(e)[:80], flush=True)
    for s in strays:
        if not s["keep"]:
            try:
                os.remove(s["path"]); freed += s["size_gb"]; removed.append(s["file"])
            except Exception as e:
                print("[prune] fail", s["file"], str(e)[:80], flush=True)
    return {"ok": True, "removed": removed, "freed_gb": round(freed, 2)}

# ---------- routing ----------
ROUTER = {"kolors-virtual-try-on": h_vton, "fashn/tryon": h_vton, "tryon": h_vton, "idm": h_vton, "catvton": h_vton,
    "flux-pro/kontext": h_kontext, "kontext": h_kontext, "clarity-upscaler": h_upscale, "upscaler": h_upscale,
    "image-to-video": h_video, "wan": h_video, "kling-video": h_video, "hailuo": h_video,
    "tts": h_tts, "elevenlabs": h_tts, "vision": h_vision, "any-llm": h_vision}
WARM = {"bg": ("kontext", _load_kontext), "video": ("wan", _load_wan), "upscale": ("upscale", _load_upscaler),
    "vision": ("qwen", _load_qwen), "voice": ("f5", _load_f5), "vton": ("vton", _load_vton)}

@app.get("/health")
def health():
    used = None
    if DEVICE == "cuda":
        free, total = torch.cuda.mem_get_info(); used = round((total - free) / 1e9, 1)
    return {"ok": True, "device": DEVICE, "loaded": list(_CACHE.keys()), "profile": GPU_PROFILE,
        "gpu": torch.cuda.get_device_name(0) if DEVICE == "cuda" else None, "vram_used_gb": used, "ram": _ram_gb(),
        "version": "v13", "pinned": list(_PIN), "offload": FORCE_OFFLOAD,
        "models": {"vision": VISION_MODEL, "bg": KONTEXT_MODEL, "video": WAN_MODEL,
                   "video_key": WAN_MODEL_KEY, "vton": VTON_MODEL, "face_enhance": FACE_ENHANCE},
        "video_opts": {"model": WAN_MODEL_KEY, "fps": WAN_FPS, "offload": WAN_OFFLOAD,
                       "default_platform": DEFAULT_PLATFORM, "max_seconds": MAX_SECONDS,
                       "models_available": sorted(WAN_CFG), "platforms": sorted(set(PLATFORM))},
        "persist": PERSIST, "models_root": MODELS_ROOT, "out": OUT}

@app.post("/warmup")
async def warmup(req: Request):
    body = await req.json(); node = body.get("node", "")
    if node not in WARM: return JSONResponse({"ok": False, "error": "warmup node: bg|video|upscale|vision|voice|vton"}, 400)
    name, loader = WARM[node]
    try:
        t0 = time.time(); _need(name, loader); el = round(time.time() - t0, 1)
        return {"ok": True, "loaded": name, "seconds": el}
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
