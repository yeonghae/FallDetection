"""
영상으로 부터 스켈레톤 추출
"""

import os
root = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, root)
import cv2
import pickle
import numpy as np
from tqdm import tqdm
from collections import deque
from mmpose.apis.inferencers import MMPoseInferencer

def load_videos(video_dir, methods):
    video_dict = {}
    for method in methods:
        video_dir_with_method = os.path.join(video_dir, method)
        video_list = []
        for f in os.listdir(video_dir_with_method):
            if f.endswith('.mp4'):
                video_info = {'name': f.split(".")[0], 'filename': f, 'filepath': os.path.join(video_dir_with_method, f)}    
                video_list.append(video_info)
        video_dict[method] = video_list
    return video_dict

def check(output_dir, method, name):
    check_path = os.path.join(output_dir, method, f"{name}.pickle")
    flag = os.path.exists(check_path)
    if flag:
        print(f"'{method}/{name}' already exists. Skipping...")
    else:
        print(f"Extracting '{method}/{name}'...")
    return flag
        

def save(output_dir, method, video_info, skeleton_data):
    output_dir_with_method = os.path.join(output_dir, method)
    if not os.path.exists(output_dir_with_method):
        os.makedirs(output_dir_with_method)
    output_file = os.path.join(output_dir_with_method,f"{video_info['name']}.pickle")
    with open(output_file, 'wb') as f:
        pickle.dump(skeleton_data, f)

def pre_process(skeletons):
    skeletons = deque(skeletons, maxlen=14)
    for i, sk in enumerate(skeletons):
        if i == 14:
            break
        indices_14 = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]        
        skeletons[i] = sk[indices_14]

    return np.array(skeletons)

def init_extractor():
    args = {
        'device': 'cuda:0',
        'init_args': {
            'pose2d': 'rtmo', 
            'pose2d_weights': './checkpoint/rtmo_checkpoint.pth', 
            'scope': 'mmpose', 
            'device': 'cuda:0', 
            'det_model': None, 
            'det_weights': None, 
            'det_cat_ids': 0, 
            'pose3d': None, 
            'pose3d_weights': None, 
            'show_progress': False
        },
        'call_args': {
            'show': False, 
            'draw_bbox': True, 
            'draw_heatmap': False, 
            'bbox_thr': 0.5, 
            'nms_thr': 0.65, 
            'pose_based_nms': True, 
            'kpt_thr': 0.3, 
            'tracking_thr': 0.3, 
            'use_oks_tracking': False, 
            'disable_norm_pose_2d': False, 
            'disable_rebase_keypoint': False, 
            'num_instances': 1, 
            'radius': 3, 
            'thickness': 1, 
            'skeleton_style': 'openpose', 
            'black_background': False, 
            'vis_out_dir': '', 
            'pred_out_dir': '', 
            'vis-out-dir': './'
        }
    }
    _init_args = args['init_args']
    _call_args = args['call_args']

    inferencer = MMPoseInferencer(**_init_args)

    return inferencer, _call_args

def extract(extractor, call_args, video_info):
    cap = cv2.VideoCapture(video_info['filepath'])
    call_args['inputs'] = video_info['filepath']
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pose_result = []
    if cap.isOpened():
        while True:
            ret, frame = cap.read()
            if ret:
                skeletons = []
                temp_call_args = call_args
                temp_call_args['inputs'] = frame
                results = extractor(**temp_call_args)
                for result in results:
                    pred = result['predictions'][0]
                    pred.sort(key = lambda x: x['bbox'][0][0])
                    for p in pred:
                            keypoints = p['keypoints']
                            keypoints_scores = p['keypoint_scores']
                            skeletons.append([a + [b] for a, b in zip(keypoints, keypoints_scores)])
                if len(skeletons) != 0:
                    skeletons = [skeletons[0] for _ in range(14)]
                    skeletons = np.array(skeletons, dtype=np.float32)
                    pre = skeletons
                    skeletons = pre_process(skeletons)
                else:
                    skeletons = [[]]

                pose_result.append(skeletons)
            else:
                break
    cap.release()
    return pose_result

def main():
    # 영상 경로
    video_dir = 'dataset/videos'
    # 영상 메소드
    # methods = ['day', 'night']
    # methods = ['day']
    methods = ['night']
    # 스켈레톤 저장 경로
    output_dir = 'dataset/skeletons'

    # 영상 로드
    video_dict = load_videos(video_dir, methods)

    # 추출기 초기화
    extractor, call_args = init_extractor()

    # 추출 및 저장
    for method in methods:
        print(f"Extracting {method} videos...")
        video_list = video_dict[method]
        for video_info in tqdm(video_list):
            flag = check(output_dir, method, video_info['name'])
            if not flag:
                pose_result = extract(extractor, call_args, video_info)
                save(output_dir, method, video_info, pose_result)

if __name__ == '__main__':
    main()