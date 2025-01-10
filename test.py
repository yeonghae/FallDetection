import os
import copy
import cv2
import torch
import numpy as np
import time
from ReID.reid import ReID
import PoseEstimation.mmlab.rtmo as rtmo

# 비디오 프레임 결과 저장 경로 설정
OUTPUT_DIR = "output/result"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def process_video(video_path):
    # ReID 모델 초기화
    reid = ReID()

    # 자세 추정 모델 로드
    inferencer, init_args, call_args, display_alias = rtmo.get_model()

    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    frame_index = 0

    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        detections = []
        skeletons = []
        images = []
        temp_call_args = copy.deepcopy(call_args)
        temp_call_args['inputs'] = frame

        # 자세 추정 모델 실행
        for result in inferencer(**temp_call_args):
            pred = result['predictions'][0]
            pred.sort(key=lambda x: x['bbox'][0][0])  # 좌측부터 정렬

            for p in pred:
                keypoints = p['keypoints']
                keypoints_scores = p['keypoint_scores']
                detection = [*p['bbox'][0], p['bbox_score']]

                # re-ID에 사용할 이미지 추출
                x1, y1, x2, y2 = map(int, detection[:4])
                image = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                images.append(image)

                detections.append(detection)
                skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])

        detections = np.array(detections, dtype=np.float32)
        skeletons = np.array(skeletons, dtype=np.float32)

        if len(detections) > 0:
            # re-ID를 통해 아이디 재설정
            det_id = reid.identify(images)

            # 시각화
            for det, id, skeleton in zip(detections, det_id, skeletons):
                x1, y1, x2, y2, score = det

                # 바운딩 박스 그리기
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # ID 표시
                text = f"ID: {id}"
                cv2.putText(frame, text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # 스켈레톤 그리기
                for joint in skeleton:
                    x, y, score = joint
                    if score > 0.5:  # 점수가 높은 경우에만 그리기
                        cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

        # 결과 프레임 저장
        output_path = os.path.join(OUTPUT_DIR, f"{video_name}_frame_{frame_index:04d}.jpg")
        cv2.imwrite(output_path, frame)

        frame_index += 1

    cap.release()

if __name__ == "__main__":
    # 처리할 비디오 폴더 경로
    VIDEO_DIR = "input/"

    if not os.path.exists(VIDEO_DIR):
        print(f"Video folder not found: {VIDEO_DIR}")
        exit(1)

    video_files = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.endswith(('.mp4', '.avi'))]

    for video_file in video_files:
        print(f"Processing video: {video_file}")
        process_video(video_file)
