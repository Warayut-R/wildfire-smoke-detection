import os
import cv2
from ultralytics import YOLO

# 1. ปรับจุดที่ 1: เปลี่ยนไปใช้ไฟล์ Weight ควันไฟป่าที่เราโหลดมา (เช่น best.pt)
model_path = 'models/best.pt'

if not os.path.exists(model_path):
    print(f"❌ ไม่พบไฟล์ {model_path} กรุณานำไฟล์ best.pt ไปวางไว้ในโฟลเดอร์ models/ ก่อนครับ")
    exit()

print(f"📦 กำลังโหลดโมเดลจาก: {model_path}")
model = YOLO(model_path)

# 2. กำหนดโฟลเดอร์รูปภาพและโฟลเดอร์เก็บผลลัพธ์
IMAGE_DIR = 'datasets/wildfire_data/test/images'
OUTPUT_DIR = 'runs/detect/predict'

os.makedirs(OUTPUT_DIR, exist_ok=True)

image_files = [f for f in os.listdir(IMAGE_DIR) if f.endswith(('.jpg', '.jpeg', '.png'))]

print(f"🔍 พบรูปภาพทั้งหมด {len(image_files)} รูป กำลังเริ่มตรวจจับควันไฟ...\n")

# 3. วนลูปประมวลผล
for img_name in image_files:
    img_path = os.path.join(IMAGE_DIR, img_name)
    
    # 🎯 ปรับจุดที่ 2: ปรับ conf=0.40 หรือ 0.50 ขึ้นไป 
    # (เพื่อกรองกรอบที่ AI ทายมั่วความมั่นใจต่ำๆ เช่น cow 0.30 ออกไปให้หมด)
    results = model(img_path, conf=0.40)
    
    for result in results:
        res_plotted = result.plot()
        save_path = os.path.join(OUTPUT_DIR, img_name)
        cv2.imwrite(save_path, res_plotted)

print("--------------------------------------------------")
print(f"🎉 ประมวลผลเสร็จเรียบร้อย!")
print(f"📂 เปิดดูรูปผลลัพธ์ได้ที่: {OUTPUT_DIR}")
print("--------------------------------------------------")
