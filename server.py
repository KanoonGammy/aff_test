# ============================================================================
# requirements.txt — Khanun Affiliate Inference server v13
# ============================================================================
# ⚠️ ติดตั้งหลักด้วย:  pip install -r requirements.txt
#
# ❌ ของพวกนี้ใส่ในไฟล์นี้ไม่ได้ (ต้องรันมือ — ดู RUNBOOK ส่วน C):
#   1) torch/torchvision (ต้องระบุ CUDA index-url):
#        pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
#   2) detectron2 (ติดตั้งจาก git — สำหรับ CatVTON/IDM DensePose):
#        pip install 'git+https://github.com/facebookresearch/detectron2.git'
#   3) แพ็กเกจระบบ (apt — ไม่ใช่ pip):
#        apt-get update && apt-get install -y ffmpeg libgl1 libglib2.0-0
#   4) IDM-VTON manual: โค้ด repo (server clone ให้เอง git.io/yisol/IDM-VTON) + น้ำหนัก yisol/IDM-VTON (HF)
#        - inference (h_vton_idm) ยังเป็นโครง → ใช้ VTON_MODEL=catvton ไปก่อน แล้วแจ้งให้เขียนต่อ
#   5) HF token (สำหรับ FLUX gated):  export HF_TOKEN=hf_xxxx
# ============================================================================

# ---- web server ----
fastapi
uvicorn[standard]
python-multipart

# ---- core ML (torch ลงแยกด้านบน) ----
diffusers==0.38.0
transformers>=4.49.0          # ★ Qwen2.5-VL ต้องการ >=4.49 (เก่ากว่านี้ server จะ fallback Qwen2-VL)
accelerate
safetensors
sentencepiece
protobuf
huggingface_hub
qwen-vl-utils                 # vision: Qwen2.5-VL

# ---- image / video io ----
pillow
numpy<2                       # basicsr/realesrgan ยังไม่ชอบ numpy 2.x
opencv-python
imageio
imageio-ffmpeg
av
einops
scipy
omegaconf                     # detectron2/CatVTON config

# ---- upscale node ----
realesrgan
basicsr
gfpgan                        # face enhance (ถ้าไม่ต้องการ ตั้ง env FACE_ENHANCE=0)
facexlib

# ---- VTON node (CatVTON / IDM ใช้ร่วม) ----
fvcore                        # ★ แก้ "No module named 'fvcore'" (detectron2 dep)
cloth-segmentation; python_version >= "3.10"
onnxruntime-gpu

# ---- voice node ----
f5-tts                        # ★ แก้ "No module named 'f5_tts'"
soundfile

# ---- monitoring ----
psutil                        # ★ /health รายงาน System RAM (RAM ที่ตันจริง)
