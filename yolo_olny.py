import cv2
import time
import torch
import numpy as np

from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls, YOLOv7

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

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
    cam_source = '/home/workspace/falldown_1.mp4'
    save_out = '/home/workspace/out_falldown_1.avi'

    inp_dets = 384
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
        if(model == "TinyYOLOv3_onecls"):
            print("YOLO3")
            return TinyYOLOv3_onecls(_inp_dets, device=device)
        else:
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

    outvid = False
    if save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(save_out, codec, 30, (inp_dets * 2, inp_dets * 2))

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
            # Predict skeleton pose of each bboxs.
            poses = pose_model.predict(frame, detected[:, 0:4], detected[:, 4])

            # Create Detections object.
            detections = [Detection(kpt2bbox(ps['keypoints'].numpy()),
                                    np.concatenate((ps['keypoints'].numpy(),
                                                    ps['kp_score'].numpy()), axis=1),
                                    ps['kp_score'].mean().numpy()) for ps in poses]

        # Update tracks by matching each track information of current and previous frame or
        # create a new track if no matched.
        tracker.update(detections)

        # Predict Actions of each track.
        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            # VISUALIZE.
            if track.time_since_update == 0:
                frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)


        if outvid:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            path = save_out + str(f).zfill(4) + ".png"
            cv2.imwrite(path, frame)
            writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clear resource.
    cam.stop()

    writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run()