import os
import cv2
import numpy as np
from Models.ultralytics.ultralytics import YOLO
from Models.segment_anything.segment_anything import sam_model_registry, SamPredictor
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from Models.Depth_Anything.depth_anything.dpt import DepthAnything
from Models.Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# YOLOv11 모델 로드
model = YOLO("yolo11n.pt")

# Segment Anything 모델 로드
sam_checkpoint = "checkpoints/sam_vit_h_4b8939.pth"  # SAM 체크포인트 경로
sam_model_type = "vit_h"
sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sam.to(device=device)
sam_predictor = SamPredictor(sam)

# Depth Anything 모델 로드
depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(device).eval()
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])

# 입력 및 출력 디렉토리 설정
INPUT_VIDEO_DIR = "input/prison_falldown_day"
OUTPUT_IMAGE_DIR = "output/segmentation"
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)

# 사람 클래스 ID (YOLO 모델에 따라 다를 수 있음, 일반적으로 COCO 데이터셋에서 0이 사람 클래스)
PERSON_CLASS_ID = 0

def adjust_bounding_boxes(masks):
    adjusted_bboxes = []
    for mask in masks:
        # 마스크에서 윤곽선 탐지
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            adjusted_bboxes.append((x, y, x + w, y + h))
    return adjusted_bboxes

def refine_segmentation_with_points(frame, predictor, input_box):
    # 바운딩 박스의 중앙점 계산
    center_x = (input_box[0] + input_box[2]) // 2
    center_y = (input_box[1] + input_box[3]) // 2
    point_coords = np.array([[center_x, center_y]])
    point_labels = np.array([1])  # 포인트는 포함 영역으로 표시

    # Segment Anything에서 포인트와 박스를 동시에 사용
    masks, scores, _ = predictor.predict(
        box=input_box, 
        point_coords=point_coords, 
        point_labels=point_labels, 
        multimask_output=True
    )

    # 최고 점수 마스크 선택
    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]

def calculate_depth_and_color(frame, depth_map, masks):
    colored_frame = frame.copy()
    for mask in masks:
        # 마스크 영역의 뎁스 맵 추출
        masked_depth = depth_map[mask > 0]
        if masked_depth.size > 0:
            avg_depth = np.mean(masked_depth)

            # 평균 뎁스값에 따라 색상 설정 (앞일수록 연하고 뒤로 갈수록 진함)
            intensity = int(255 - (avg_depth / 255.0) * 200)  # 뎁스에 따라 밝기를 반전하여 조정
            color = (intensity, intensity, 255)  # 연한 파란색에서 진한 파란색으로 변화

            # 색상을 마스크 영역에 적용
            colored_frame[mask > 0] = color

    return colored_frame

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0

    return inter_area / union_area

def detect_segment_and_depth(input_dir, output_seg_dir, model, predictor, depth_model, transform):
    for video_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, video_name)

        # 비디오 캡처
        cap = cv2.VideoCapture(input_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # YOLO 모델로 탐지 수행
            results = model(frame, iou=0.5)

            masks = []  # 객체 마스크 저장
            bboxes = []  # 바운딩 박스 저장

            for result in results[0].boxes:
                if int(result.cls) == PERSON_CLASS_ID:  # 사람 클래스 필터링
                    x1, y1, x2, y2 = map(int, result.xyxy[0])

                    # 탐지된 영역의 이미지 및 마스크 생성
                    predictor.set_image(frame)

                    # Segment Anything 예측 수행 (Refined with points)
                    input_box = np.array([x1, y1, x2, y2])
                    mask = refine_segmentation_with_points(frame, predictor, input_box)

                    # 마스크를 frame 크기로 변환
                    mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)

                    masks.append(mask_resized)
                    bboxes.append((x1, y1, x2, y2))

            # 마스크 기반 바운딩 박스 조정
            adjusted_bboxes = adjust_bounding_boxes(masks)

            # 뎁스 이미지 생성
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) / 255.0
            transformed_frame = transform({'image': frame_rgb})['image']
            transformed_frame = torch.from_numpy(transformed_frame).unsqueeze(0).to(device)

            with torch.no_grad():
                depth = depth_model(transformed_frame)

            depth = F.interpolate(depth[None], (frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            depth_map = depth.cpu().numpy().astype(np.uint8)

            # 마스크된 객체에 따라 뎁스 기반 색상 적용
            colored_frame = calculate_depth_and_color(frame, depth_map, masks)

            # 조정된 바운딩 박스를 원본 이미지에 추가
            for i, bbox in enumerate(adjusted_bboxes):
                color = (0, 255, 0)  # 기본 초록색
                for j, other_bbox in enumerate(adjusted_bboxes):
                    if i != j and calculate_iou(bbox, other_bbox) > 0.3:  # IoU가 0.5 이상이면 겹침으로 판단
                        color = (0, 0, 255)  # 빨간색으로 변경
                        break
                x1, y1, x2, y2 = bbox
                cv2.rectangle(colored_frame, (x1, y1), (x2, y2), color, 2)

            # 결과 이미지 저장
            seg_output_path = os.path.join(output_seg_dir, f"{os.path.splitext(video_name)[0]}_frame_{frame_idx}.png")
            cv2.imwrite(seg_output_path, colored_frame)

            frame_idx += 1

        cap.release()
        print(f"Processed video: {video_name}")

if __name__ == "__main__":
    detect_segment_and_depth(INPUT_VIDEO_DIR, OUTPUT_IMAGE_DIR, model, sam_predictor, depth_model, transform)
