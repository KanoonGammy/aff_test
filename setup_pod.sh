#!/usr/bin/env bash
# ============================================================================
# setup_pod.sh — ติดตั้งทุกอย่างบน RunPod ให้ "ลำดับถูก" ครั้งเดียวจบ (server v13)
# แก้ปัญหาที่เจอซ้ำ: torch/torchvision ผิดเวอร์ชัน (operator torchvision::nms / infer_schema)
# วิธีใช้:  cd /workspace/aff_test && bash setup_pod.sh
# ============================================================================
set -e
cd "$(dirname "$0")"

echo "==> [1/5] ไลบรารีหลัก (requirements.txt)"
pip install -r requirements.txt

# ★★★ ต้องลง torch "เป็นขั้นสุดท้าย" เสมอ — basicsr/realesrgan/f5-tts ดึง torchvision ตัวอื่นมาทับ
echo "==> [2/5] FIX torch/torchvision ให้แมตช์ (uninstall ก่อนแล้วลงคู่ cu124)"
pip uninstall -y torch torchvision || true
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124

echo "==> [3/5] detectron2 (DensePose ของ VTON — คอมไพล์สักครู่)"
pip install 'git+https://github.com/facebookresearch/detectron2.git' || \
  echo "   (detectron2 ลงไม่ผ่าน — VTON ใช้ catvton ต้องการตัวนี้ · IDM/อื่นยังไปต่อได้)"

echo "==> [4/5] แพ็กเกจระบบ (ffmpeg/libgl)"
apt-get update -y >/dev/null 2>&1 && apt-get install -y ffmpeg libgl1 libglib2.0-0 >/dev/null 2>&1 || \
  echo "   (apt ลงไม่ได้ — ถ้า ffmpeg มีอยู่แล้วข้ามได้)"

echo "==> [5/5] ★ VERIFY torch — ต้องได้ 'nms ok True' (ไม่งั้น vision/Qwen จะพัง)"
python -c "import torch, torchvision; from torchvision.ops import nms; \
print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| nms ok | cuda', torch.cuda.is_available())"

echo ""
echo "✅ เสร็จ — ต่อไป: export HF_TOKEN=hf_xxx  แล้วรัน server (ดู RUNBOOK ส่วน D-E)"
echo "   ตัวเลือก env: export WAN_MODEL=5B  VTON_MODEL=catvton  DEFAULT_PLATFORM=tiktok"
