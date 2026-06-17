# aff_test — เซิร์ฟเวอร์โมเดล open สำหรับไลน์ผลิตคลิป Khanun (รันบน RunPod Pod)

แทนที่ fal.ai · `/infer` รับ {model_id, inputs} → คืน result สคีมาเดียวกับ fal
**โมเดล + งานที่ออกมา เก็บในโฟลเดอร์โปรเจกต์นี้** (`./models`, `./outputs`) · งานแยกโฟลเดอร์ตามชื่องาน

## รันบน RunPod Pod (terminal)
```bash
git clone https://github.com/KanoonGammy/aff_test.git
cd aff_test
pip install -r requirements.txt
python -m uvicorn server:app --host 0.0.0.0 --port 8000
```
เช็ค: เปิด `https://<pod-id>-8000.proxy.runpod.net/health` → เห็น `{"ok":true,"gpu":"NVIDIA A40",...}` = พร้อม

## อัปเดตโค้ด (แก้ที่ GitHub → Pod ดึง)
```bash
cd aff_test && git pull && (กด Ctrl+C หยุด server เดิม) && python -m uvicorn server:app --host 0.0.0.0 --port 8000
```

## โมเดลเก็บที่ไหน
- ดาวน์โหลด **ลงบน Pod** ครั้งแรกที่ใช้แต่ละโหนด (lazy) → cache ใน `./models` (อยู่ในโปรเจกต์)
- ถ้า Pod มี Network Volume (mount /workspace) → clone repo ไว้ใน /workspace จะ persist ข้ามการ restart ไม่ต้องโหลดซ้ำ

## ติดตั้งโมเดลหนัก (ทำเมื่อใช้โหนดนั้นจริง)
| โหนด | ติดตั้งเพิ่ม |
|---|---|
| ความคม (เบาสุด ลองก่อน) | `pip install realesrgan` |
| พื้นหลัง (Flux Kontext) | `pip install diffusers transformers accelerate` + `huggingface-cli login` (ยอม license) |
| อ่านภาพ (Qwen2-VL) | `pip install transformers accelerate qwen-vl-utils` |
| เสียง (F5-TTS) | `pip install f5-tts soundfile` + วางไฟล์เสียงต้นแบบ `voice/ref.wav` |
| วีดีโอ (Wan2.2) | `pip install diffusers` + ffmpeg (รุ่น 5B/GGUF ถ้า VRAM น้อย) |
| ใส่ชุด (CatVTON) | `git clone https://github.com/Zheng-Chong/CatVTON` + ทำ `catvton_pipeline.py` ห่อ try_on |
