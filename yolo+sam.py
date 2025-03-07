import cv2
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys

from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls, YOLOv7

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
from PIL import Image

########
#SAM
sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

predictor = SamPredictor(sam)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))
########

FRAME_STEP = 14 ## 테스트용 임시 !! TMEP

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

def ResizePadding(height, width):
    desized_size = (height, width)

    def resizePadding(image, **kwargs):
        old_size = image.shape[:2]
        max_size_idx = old_size.index(max(old_size))
        ratio = float(desized_size[max_size_idx]) / max(old_size)
        new_size = tuple([int(x * ratio) for x in old_size])

        if new_size > desized_size:
            min_size_idx = old_size.index(min(old_size))
            ratio = float(desized_size[min_size_idx]) / min(old_size)
            new_size = tuple([int(x * ratio) for x in old_size])

        image = cv2.resize(image, (new_size[1], new_size[0]))
        delta_w = desized_size[1] - new_size[1]
        delta_h = desized_size[0] - new_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)
        return image
    return resizePadding

def run():
    cam_source = '/home/workspace/Falldown_44.mp4'
    save_out = '/home/workspace/out_'

    inp_dets = 768 #384
    device = "cuda"
    pose_input_size = '288x160' # default=224x160(resnet50) # 288x224(resnet101)
    inp_pose = pose_input_size.split('x')
    pose_backbone = "resnet50"
    device = "cuda"

    resize_fn = ResizePadding(inp_dets, inp_dets)

    def preproc(image):
        """preprocess function for CameraLoader.
        """
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def init_detection_model(_inp_dets, model="YOLOv7"):
        print("YOLO7")
        return YOLOv7(_inp_dets, device=device)

    def init_pose_model(_param):
        pos = (int(_param[0]), int(_param[1]))
        return SPPE_FastPose(pose_backbone, pos[0], pos[1], device=device)

    detect_model = init_detection_model(inp_dets, model="YOLO7")
    pose_model = init_pose_model(inp_pose)
    tracker = Tracker(max_age=FRAME_STEP, n_init=3)

    # Use loader thread with Q for video file.
    cam = CamLoader_Q(cam_source, queue_size=10000, preprocess=preproc).start()
    time.sleep(1)

    f = 0
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()


        # Detect humans bbox in the frame with detector model.
        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        # Predict each tracks bbox of current frame from previous frames information with Kalman filter.
        tracker.predict()

        # Merge two source of predicted bbox together.
        for track in tracker.tracks:
            det = torch.tensor([track.to_tlbr().tolist() + [0.5, 1.0, 0.0]], dtype=torch.float32)
            detected = torch.cat([detected, det], dim=0) if detected is not None else det
        
        detections = []  # List of Detections object for tracking.
        if detected is not None:
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Predict skeleton pose of each bboxs.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        tracker.update(detections)
       
        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
####################################
            # f_bbox = [track.to_tlbr()]

            # #다중 입력 프롬프트 사용
            # predictor.set_image(image)

            # input_boxes = torch.tensor(f_bbox, device=predictor.device)
            # input_boxes_c = torch.tensor(f_bbox)


            # transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
            # masks, _, _ = predictor.predict_torch(
            #     point_coords=None,
            #     point_labels=None,
            #     boxes=transformed_boxes,
            #     multimask_output=False,
            # )

            # # 현재 마스크를 2차원으로 변형합니다.
            # mask_2d = masks.cpu().numpy().squeeze()
            # mask_2d = np.clip(mask_2d, 0, 1)

            # img = np.ones_like(image) * 255
            # img[mask_2d == 1] = image[mask_2d == 1]

            # poses_s = pose_model.predict(img, input_boxes_c, detected[:, 4]) # TODO bbox의 confidence는 따로 처리
            # detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
            #                         np.concatenate((ps['keypoints'].numpy(),
            #                                         ps['kp_score'].numpy()), axis=1),
            #                         ps['kp_score'].mean().numpy()) for ps in poses_s]

#             # track.keypoints_list[0] = detections[0].keypoints
#             # track.keypoints_list[-1] = detections[0].keypoints


# ####################################
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)
            
            # VISUALIZE.
            if track.time_since_update == 0: #TODO 임시코드
                    
                # 잘된거 골라서 보여주기
                # confidence = np.mean(track.keypoints_list[-1], axis=0)[2]
                # if 0 < len(detections) and confidence < detections[0].confidence :
                #     frame = draw_single(frame, detections[0].keypoints)
                # else:
                frame = draw_single(frame, track.keypoints_list[-1])

                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        path = str(f).zfill(4) + ".png"
        cv2.imwrite(path, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()

    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()