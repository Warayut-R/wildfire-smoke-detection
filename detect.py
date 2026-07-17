import os
import cv2
from ultralytics import YOLO

# 1. โหลดโมเดล YOLO (ใช้น้ำหนักโมเดลในโฟลเดอร์ models หรือยึด yolov8n.pt / yolo11n.pt)
model_path = 'models/best.pt' if os.path.exists('models/best.pt') else 'yolov8n.pt'
print(f"📦 กำลังโหลดโมเดลจาก: {model_path}")
model = YOLO(model_path)

# 2. กำหนดโฟลเดอร์รูปภาพที่จะนำมาตรวจจับ และโฟลเดอร์เก็บผลลัพธ์
IMAGE_DIR = 'datasets/wildfire_data/test/images'
OUTPUT_DIR = 'runs/detect/predict'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 3. ดึงรายชื่อรูปภาพทั้งหมด
image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"🔍 พบรูปภาพทั้งหมด {len(image_files)} รูป กำลังเริ่มรันตรวจจับควันไฟ...\n")

# 4. วนลูปตรวจจับทีละรูป
for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)

    # สั่งให้ YOLO ทำการประมวลผล
    results = model(img_path, conf=0.25)

    # บันทึกรูปภาพผลลัพธ์ที่มีกรอบตรวจจับ
    for result in results:
        res_plotted = result.plot()
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, res_plotted)

print("--------------------------------------------------")
print(f"🎉 ตรวจจับควันไฟเสร็จสิ้นทุกรูป!")
print(f"📂 รูปภาพผลลัพธ์ถูกเก็บไว้ที่: {OUTPUT_DIR}")
print("--------------------------------------------------")