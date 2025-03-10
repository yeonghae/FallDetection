"""
This script to extract skeleton joints position and score.
스켈레톤 관절을 추출하는 이 스크립트는 위치 및 점수를 추출합니다.

- This 'annot_folder' is a action class and bounding box for each frames that came with dataset.
    Should be in format of [frame_idx, action_cls, xmin, ymin, xmax, ymax]
        Use for crop a person to use in pose estimation model.
- If have no annotation file you can leave annot_folder = '' for use Detector model to get the bounding box.
"""
import os, sys
import cv2
import time
import torch
import pandas as pd
import numpy as np
import torchvision.transforms as transforms

# 상위 디렉토리 지정
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from DetectorLoader import YOLOv7
from PoseEstimateLoader import SPPE_FastPose
from fn import vis_frame_fast

# save_path = 'C:/Users/twoimo/Documents/GitHub/FallDetection/Data/Home_new-pose+score.csv'
# annot_file = 'C:/Users/twoimo/Documents/GitHub/FallDetection/Data/Home_new.csv'  # from create_dataset_1.py
# video_folder = 'C:/Users/twoimo/Documents/GitHub/FallDetection/Data/le2i_falldata/Home/Videos'
# annot_folder = 'C:/Users/twoimo/Documents/GitHub/FallDetection/Data/le2i_falldata/Home/Annotation_files'  # bounding box annotation for each frame.

save_path = 'C:/Users/twoimo/Documents/GitHub/FallDetection-original/__SyntheticData/PrisonAllClip_29+1_40-pose+score.csv'
annot_file = 'C:/Users/twoimo/Documents/GitHub/FallDetection-original/Data/output_file/PrisonAllClip_29+1_40.csv'  # from create_dataset_1.py
video_folder = 'C:/Users/twoimo/Documents/GitHub/FallDetection-original/Data/prison_data/PrisonAllClip_29+1_40'
annot_folder =  ''  # bounding box annotation for each frame.

# DETECTION MODEL.
input_size = 768
detector = YOLOv7(input_size=input_size)

# POSE MODEL. 320 x 256 (기본값)
inp_h = 288
inp_w = 160
pose_estimator = SPPE_FastPose(backbone='resnet50', input_height=inp_h, input_width=inp_w)

# with score.
columns = ['video', 'frame', 'Nose_x', 'Nose_y', 'Nose_s', 'LShoulder_x', 'LShoulder_y', 'LShoulder_s',
           'RShoulder_x', 'RShoulder_y', 'RShoulder_s', 'LElbow_x', 'LElbow_y', 'LElbow_s', 'RElbow_x',
           'RElbow_y', 'RElbow_s', 'LWrist_x', 'LWrist_y', 'LWrist_s', 'RWrist_x', 'RWrist_y', 'RWrist_s',
           'LHip_x', 'LHip_y', 'LHip_s', 'RHip_x', 'RHip_y', 'RHip_s', 'LKnee_x', 'LKnee_y', 'LKnee_s',
           'RKnee_x', 'RKnee_y', 'RKnee_s', 'LAnkle_x', 'LAnkle_y', 'LAnkle_s', 'RAnkle_x', 'RAnkle_y',
           'RAnkle_s', 'label']


def normalize_points_with_size(points_xy, width, height, flip=False):
    points_xy[:, 0] /= width
    points_xy[:, 1] /= height
    if flip:
        points_xy[:, 0] = 1 - points_xy[:, 0]
    return points_xy


annot = pd.read_csv(annot_file)
vid_list = annot['video'].unique()
for vid in vid_list:
    print(f'Process on: {vid}')
    df = pd.DataFrame(columns=columns)
    cur_row = 0

    # Pose Labels.
    if annot is not None and 'video' in annot.columns:
        frames_label = annot[annot['video'] == vid].reset_index(drop=True)
    else:
        pass

    cap = cv2.VideoCapture(os.path.join(video_folder, vid))
    frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Bounding Boxs Labels.
    annot_file = os.path.join(annot_folder, vid.split('.')[0]) + '.txt'
    annot = None
    if os.path.exists(annot_file):
        annot = pd.read_csv(annot_file, header=None,
                                  names=['frame_idx', 'class', 'xmin', 'ymin', 'xmax', 'ymax'])
        annot = annot.dropna().reset_index(drop=True)

        assert frames_count == len(annot), 'frame count not equal! {} and {}'.format(frames_count, len(annot))

    fps_time = 0
    i = 1
    while True:
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            filtered_label = frames_label[frames_label['frame'] == i]['label']
            bb = np.array([])
            if not filtered_label.empty:
                cls_idx = int(filtered_label.values[0])
            else:
                pass
            if annot is not None and not annot.empty:
                bb = np.array(annot.iloc[i-1, 2:].astype(int))
            else:
                detection_result = detector.detect(frame)
                if detection_result is not None:
                    bb = detection_result[0, :4].numpy().astype(int)
                else:
                    pass
            bb[:2] = np.maximum(0, bb[:2] - 5)
            bb[2:] = np.minimum(frame_size, bb[2:] + 5) if bb[2:].any() != 0 else bb[2:]
            
            result = []
            if bb.any() != 0:
                result = pose_estimator.predict(frame, torch.tensor(bb[None, ...]),
                                                torch.tensor([[1.0]]))

            if len(result) > 0:
                pt_norm = normalize_points_with_size(result[0]['keypoints'].numpy().copy(),
                                                     frame_size[0], frame_size[1])
                pt_norm = np.concatenate((pt_norm, result[0]['kp_score']), axis=1)

                #idx = result[0]['kp_score'] <= 0.05
                #pt_norm[idx.squeeze()] = np.nan
                row = [vid, i, *pt_norm.flatten().tolist(), cls_idx]
                scr = result[0]['kp_score'].mean()
            else:
                row = [vid, i, *[np.nan] * (13 * 3), cls_idx]
                scr = 0.0

            df.loc[cur_row] = row
            cur_row += 1

            # VISUALIZE.
            frame = vis_frame_fast(frame, result)
            if bb.any():
                frame = cv2.rectangle(frame, (bb[0], bb[1]), (bb[2], bb[3]), (0, 255, 0), 2)
            frame = cv2.putText(frame, 'Frame: {}, Pose: {}, Score: {:.4f}'.format(i, cls_idx, scr),
                                (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            frame = frame[:, :, ::-1]
            fps_time = time.time()
            i += 1

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    if os.path.exists(save_path):
        df.to_csv(save_path, mode='a', header=False, index=False)
    else:
        df.to_csv(save_path, mode='w', index=False)

