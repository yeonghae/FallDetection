"""
추출된 스켈레톤으로부터 포즈 추정을 수행
"""

import os
root = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, root)
import cv2
import math
import json
import torch
import pickle
from tqdm import tqdm
import numpy as np
from collections import deque
from mmpose.apis.inferencers import MMPoseInferencer
from module.Loader import TSSTG

def init_args():
    args = {
        'checkpoint': './checkpoint/falldown_checkpoint.pth',
        'threshhold': 0.8,
        'device': 'cuda:0',
    }
    return args

def load_skeleton(skeleton_dir, methods):
    skeleton_dict = {}
    for method in methods:
        skeleton_dir_with_method = os.path.join(skeleton_dir, method)
        skeleton_list = []
        for f in os.listdir(skeleton_dir_with_method):
            if f.endswith('.pickle'):                
                skeleton_info = {'name': f.split(".")[0], 'filename': f, 'filepath': os.path.join(skeleton_dir_with_method, f)}
                skeleton_list.append(skeleton_info)
        skeleton_dict[method] = skeleton_list
    return skeleton_dict

def read_skeleton(skeleton_info):
    with open(skeleton_info['filepath'], 'rb') as data:
        skeleton_data = pickle.load(data)
    return skeleton_data

def read_video_info(video_dir, method, skeleton_info):
    filename = skeleton_info['filename'].split(".")[0] + ".mp4"
    filepath = os.path.join(video_dir, method, filename)
    cap = cv2.VideoCapture(filepath)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_size = (frame_width, frame_height)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_info = {
        'filename': filename,
        'filepath': filepath,
        'fps': fps,
        'frame_size': frame_size,
        'frame_count': frame_count
    }
    return video_info

def check(output_dir, method, skeleton_info):
    output_dir_with_method = os.path.join(output_dir, method)
    output_filename = skeleton_info['name'] + ".json"
    output_filepath = os.path.join(output_dir_with_method, output_filename)
    flag = os.path.exists(output_filepath)
    if flag:
        print(f"'{output_filepath}' already exists. Skipping...")
    return flag

def save(output_dir, method, skeleton_info, json_data):
    output_dir_with_method = os.path.join(output_dir, method)
    os.makedirs(output_dir_with_method, exist_ok=True)
    output_filename = skeleton_info['name'] + ".json"
    output_filepath = os.path.join(output_dir_with_method, output_filename)
    with open(output_filepath, 'w') as f:
        json.dump(json_data, f)

def init_model(args):
    model = TSSTG(args['checkpoint'], args['device'])
    return model

def show_progress(cur, total, desc, log = "", bar_len=100):
    percent = math.ceil((cur / total) * 100)
    cur_len = math.ceil((cur / total) * bar_len)
    bar = '|' + '█' * cur_len + ' ' * (bar_len - cur_len) + '|'
    if cur == 0:
        print(f"[{desc}] {percent}% {bar} {cur}/{total} {log}", end='\r')
    else:
        print(f"[{desc}] {percent}% {bar} {cur}/{total} {log}", end='\r', flush=True)

def main():
    # 인자 초기화
    args = init_args()

    # 영상 경로
    video_dir = 'dataset/videos'
    # 스켈레톤 경로
    skeleton_dir = 'dataset/skeletons'
    # 메소드
    methods = ['day', 'night']
    # 결과 저장 경로
    output_dir = 'result'

    # 모델 초기화
    model = init_model(args)

    # 스켈레톤 목록 로드
    skeleton_dict = load_skeleton(skeleton_dir, methods)
    
    # 추출 및 저장
    for method in methods:
        print(f"Evaluate {method} videos...")
        prev_label = "normal"
        skeleton_list = skeleton_dict[method]
        total_length = len(skeleton_list)
        for total_idx, skeleton_info in enumerate(skeleton_list):
            total_progress = round((total_idx / total_length) * 100, 1)
            print(f"[TOTAL {total_progress}%] Current skeleton info: ", skeleton_info)
            flag = check(output_dir, method, skeleton_info)
            if not flag:
                video_info = read_video_info(video_dir, method, skeleton_info)
                skeleton_data = read_skeleton(skeleton_info)
                json_data = []
                for idx, pose_data in enumerate(skeleton_data):
                    try:
                        pred = model.predict(pose_data, video_info['frame_size'])
                        label = model.class_names[pred[0].argmax()]
                        confidence = pred[0][1]
                        prev_label = label
                    except:
                        label = prev_label
                    if label == "Normal":
                        label = "normal"
                    else:
                        label = "falldown"
                    show_progress(idx, video_info['frame_count'], f"Predict pose from {skeleton_info['name']}", f"Action: {label}")
                    json_data.append({'frame_idx': idx, 'label': label})
                save(output_dir, method, skeleton_info, json_data)  

if __name__ == '__main__':
    main()