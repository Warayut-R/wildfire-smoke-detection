import cv2
import os

# 1. กำหนดโฟลเดอร์ต้นทางวิดีโอ และโฟลเดอร์ปลายทางเก็บบรรจุรูป
VIDEO_DIR = 'raw_videos'
OUTPUT_DIR = 'datasets/wildfire_data/test/images'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. ดึงรายชื่อไฟล์วิดีโอทั้งหมดใน raw_videos
video_files = [f for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi', '.mov'))]

print(f"🎬 พบไฟล์วิดีโอทั้งหมด {len(video_files)} ไฟล์ กำลังเริ่มสกัดรูปภาพ...")

total_saved = 0

# 3. วนลูปตัดภาพทีละวิดีโอ
for video_name in video_files:
    video_path = os.path.join(VIDEO_DIR, video_name)
    cap = cv2.VideoCapture(video_path)
    
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if int(cap.get(cv2.CAP_PROP_FPS)) > 0 else 30
    count = 0
    saved_in_video = 0
    base_name = os.path.splitext(video_name)[0]
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # แคปรูปภาพออกมาทุกๆ 1 วินาที
        if count % fps == 0:
            img_name = f"{base_name}_frame_{saved_in_video:04d}.jpg"
            save_path = os.path.join(OUTPUT_DIR, img_name)
            cv2.imwrite(save_path, frame)
            saved_in_video += 1
            total_saved += 1
            
        count += 1
        
    cap.release()
    print(f"  └─ คลิป {video_name}: สกัดได้ {saved_in_video} รูป")

print(f"\n🎉 สกัดภาพเสร็จสิ้น! ได้รูปภาพรวมทั้งหมด {total_saved} รูป")
print(f"📂 ไฟล์รูปภาพถูกเก็บไว้ที่: {OUTPUT_DIR}")