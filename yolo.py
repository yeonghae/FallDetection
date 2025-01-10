import os
from ultralytics import YOLO
import cv2

# YOLOv11 모델 로드
model = YOLO("yolo11n.pt")

# 입력 및 출력 디렉토리 설정
INPUT_VIDEO_DIR = "input/prison_falldown_night"
OUTPUT_VIDEO_DIR = "output/yolo/night"
os.makedirs(OUTPUT_VIDEO_DIR, exist_ok=True)

# 사람 클래스 ID (YOLO 모델에 따라 다를 수 있음, 일반적으로 COCO 데이터셋에서 0이 사람 클래스)
PERSON_CLASS_ID = 0

# 비디오에서 사람 객체 탐지
def detect_people_in_videos(input_dir, output_dir, model):
    for video_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, video_name)
        output_path = os.path.join(output_dir, video_name)

        # 비디오 캡처
        cap = cv2.VideoCapture(input_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 모델로 탐지 수행
            results = model(frame)

            # 탐지 결과 그리기
            for result in results[0].boxes:
                if int(result.cls) == PERSON_CLASS_ID:  # 사람 클래스 필터링
                    x1, y1, x2, y2 = map(int, result.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 결과 프레임 저장
            out.write(frame)

        cap.release()
        out.release()
        print(f"Processed and saved: {output_path}")

# 실행
if __name__ == "__main__":
    detect_people_in_videos(INPUT_VIDEO_DIR, OUTPUT_VIDEO_DIR, model)
