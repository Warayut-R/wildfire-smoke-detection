# 🌲 Wildfire Smoke Detection Project (YOLOv8)

โปรเจกต์ทดลองตรวจจับควันไฟป่าจาก Dataset รูปภาพที่สกัดมาจากวิดีโอเหตุการณ์จริง ด้วยสถาปัตยกรรม YOLOv8

## 📁 โครงสร้างโปรเจกต์
- `datasets/wildfire_data/test/images/`: ชุดข้อมูลภาพนิ่งควันไฟป่า (.jpg)
- `models/`: ไฟล์ Preset น้ำหนักโมเดลตั้งต้น (`yolov8n.pt`)
- `extract_frames.py`: สคริปต์ Python สำหรับตัดวิดีโอออกมาเป็นเฟรมรูปภาพ
- `run_experiment.py`: สคริปต์หลักสั่งประมวลผล Object Detection
- `requirements.txt`: รายชื่อไลบรารีสำหรับเตรียม Environment

## 🚀 วิธีการทดสอบรันโปรเจกต์

1. ติดตั้ง Dependencies:
   ```bash
   pip install -r requirements.txt
