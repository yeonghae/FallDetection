"""
This script to create dataset and labels by clean off some NaN, do a normalization,
label smoothing and label weights by scores.
"""
import os
import csv
import glob
import pickle
import natsort
import random
import shutil
import numpy as np
import pandas as pd

class_names = ['Normal', 'Fall Down']
main_parts = ['LShoulder_x', 'LShoulder_y', 'RShoulder_x', 'RShoulder_y', 'LHip_x', 'LHip_y', 'RHip_x', 'RHip_y']
main_idx_parts = [1, 2, 7, 8, -1]  # 1.5

def scale_pose(xy):
    """
    Normalize pose points by scale with max/min value of each pose.
    xy : (frames, parts, xy) or (parts, xy)
    """
    if xy.ndim == 2:
        xy = np.expand_dims(xy, 0)
    xy_min = np.nanmin(xy, axis=1)
    xy_max = np.nanmax(xy, axis=1)
    for i in range(xy.shape[0]):
        xy[i] = ((xy[i] - xy_min[i]) / (xy_max[i] - xy_min[i])) * 2 - 1
    return xy.squeeze()

# use_nframe_flatten: 노멀 애니메이션들의 프레임 비율을 고려한 샘플링
def seq_label_smoothing(labels, max_step=10, use_nframe_flatten=True):
    steps = 0
    remain_step = 0
    target_label = 0
    active_label = 0
    start_change = 0
    max_val = np.max(labels)
    min_val = np.min(labels)
    for i in range(labels.shape[0]):
        if remain_step > 0:
            if i >= start_change:
                labels[i][active_label] = max_val * remain_step / steps
                labels[i][target_label] = max_val * (steps - remain_step) / steps \
                    if max_val * (steps - remain_step) / steps else min_val
                remain_step -= 1
            continue

        diff_index = np.where(np.argmax(labels[i:i+max_step], axis=1) - np.argmax(labels[i]) != 0)[0]
        if len(diff_index) > 0:
            start_change = i + remain_step // 2
            steps = diff_index[0]
            remain_step = steps
            target_label = np.argmax(labels[i + remain_step])
            active_label = np.argmax(labels[i])
    return labels

def compose_and_label_smoothing(csv_pose_file, smooth_labels_step = 8, n_frames=30):

    # Params.
    # n_frames = 30
    skip_frame = 1
    
    annot = pd.read_csv(csv_pose_file)

    # Remove NaN.
    idx = annot.iloc[:, 2:-1][main_parts].isna().sum(1) > 0
    idx = np.where(idx)[0]
    annot = annot.drop(idx)
    # One-Hot Labels.
    label_onehot = pd.get_dummies(annot['label'])
    annot = annot.drop('label', axis=1).join(label_onehot)
    cols = label_onehot.columns.values

    feature_set = np.empty((0, n_frames, 14, 3))
    labels_set = np.empty((0, len(cols)))
    vid_list = annot['video'].unique()
    for vid in vid_list:
        print(f'Process on: {vid}')
        data = annot[annot['video'] == vid].reset_index(drop=True).drop(columns='video')
        
        vid_type = vid[0]

        # Label Smoothing.
        esp = 0.1
        data[cols] = data[cols] * (1 - esp) + (1 - data[cols]) * esp / (len(cols) - 1)
        data[cols] = seq_label_smoothing(data[cols].values, smooth_labels_step)

        # Separate continuous frames.
        frames = data['frame'].values
        frames_set = []
        fs = [0]
        for i in range(1, len(frames)):
            if frames[i] < frames[i-1] + 10:
                fs.append(i)
            else:
                frames_set.append(fs)
                fs = [i]
        frames_set.append(fs)

        for fs in frames_set:
            xys = data.iloc[fs, 1:-len(cols)].values.reshape(-1, 13, 3)
            # Scale pose normalize.
            xys[:, :, :2] = scale_pose(xys[:, :, :2])
            # Add center point.
            xys = np.concatenate((xys, np.expand_dims((xys[:, 1, :] + xys[:, 2, :]) / 2, 1)), axis=1)

            # Weighting main parts score. #정답 스켈레톤 쓸땐 안필요한것 같음. 오히려 잘못 학습될 가능성 존재
            scr = xys[:, :, -1].copy()
            scr[:, main_idx_parts] = np.minimum(scr[:, main_idx_parts] * 1.5, 1.0)
            # Mean score.
            scr = scr.mean(1)

            # Targets.
            lb = data.iloc[fs, -len(cols):].values
            # Apply points score mean to all labels.
            lb = lb * scr[:, None]

            for i in range(xys.shape[0] - n_frames):
                feature_set = np.append(feature_set, xys[i:i+n_frames][None, ...], axis=0)
                labels_set = np.append(labels_set, lb[i:i+n_frames].mean(0)[None, ...], axis=0)

    return feature_set, labels_set

def merge(w_path, f_path, n_path, s_path, num_fall=0, num_normal=0, out_file_name="temp.csv"):

    if os.path.exists(s_path):
        shutil.rmtree(s_path)
    os.mkdir(s_path)

    def data_merge(data_dir, save_file_name):
        if os.path.exists(save_file_name):
            os.remove(save_file_name)          
            
        # 모든 CSV 파일 읽기
        header = None
        for i, filename in enumerate(os.listdir(data_dir)):
            if not filename.endswith(".csv"):
                continue
            
            if i == 0:
                with open(os.path.join(data_dir, filename)) as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    header.insert(0, "video")
                    
                # 헤더 데이터 저장 TODO: 하기
                with open(save_file_name, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(header)

                # video	frame	Nose_x	Nose_y	Nose_s	LShoulder_x	LShoulder_y	LShoulder_s	RShoulder_x	RShoulder_y	RShoulder_s	LElbow_x	LElbow_y	LElbow_s	RElbow_x	RElbow_y	RElbow_s	LWrist_x	LWrist_y	LWrist_s	RWrist_x	RWrist_y	RWrist_s	LHip_x	LHip_y	LHip_s	RHip_x	RHip_y	RHip_s	LKnee_x	LKnee_y	LKnee_s	RKnee_x	RKnee_y	RKnee_s	LAnkle_x	LAnkle_y	LAnkle_s	RAnkle_x	RAnkle_y	RAnkle_s	label


            # CSV 파일 읽고 데이터 저장
            data = []
            with open(os.path.join(data_dir, filename), 'r+') as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    data.append([filename.replace('.csv', '.avi')] + row)


            # 각 데이터 파일을 한줄씩 저장
            with open(save_file_name, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(data)


    def extract_names(card_files):
        full_card_names = []
        for file in card_files:
            card_name = file.split('/')[-1].split('.csv')[0]
            full_card_names.append(card_name)
        return full_card_names
    
    def draw_data_balanced(num_draws, full_card_names, isfalldown=False):            
        from collections import Counter
        normal_data = { #임시. 
            "AirSquat": 111,
            "Bored": 321,
            "CrouchToStand2": 99,
            # "Entry": 312,
            # "Stand": 92,
            "Standing": 84,
            "StandingUp2": 117,
            "StartWalking": 87,
            "StopWalking": 90,
            "StopWalkingWithRifle": 69,
            "WalkIn-Circle": 522,
            "Walking1": 144,
            "Walking2": 49,
            "Walking3": 78,
            "WalkingBackwards": 66,
            "WalkingTurn180": 60,
            "WalkingWhileTexting": 120,
            "Waving2": 141,
            "GettingUp1": 258,
            "GettingUp2": 81,
            "Laying": 105,
            # "SleepingIdle1": 527,
            "SleepingIdle2": 207,
            "StandUp1": 249,
            "StandingUp1": 342,
            "StandingUp2": 183,

            "SittingThumbsUp":116,
            "SittingClap":171,
            "SittingDazed":251,
            "SittingIdle":327,
            "SittingIdle2":309,
            "Waking":56,
            "DodgeWhileSitting":96,
            "HonkingHorn":101         
        }

        fall_data = {
            "Dying1": 38,
            "Dying2": 38,
            "DyingBackwards": 40,
            "FallFlat": 60,
            "FallOver": 47,
            "FallingDown": 38,
            "KickToTheGroin1": 73,
            "KickToTheGroin2": 71,
            "KnockedDown": 56,
            "KnockedOut1": 57,
            "KnockedOut2": 78,
            "ShoulderHitAndFall": 65,
            "Slipping": 41,
            "StumbleBackwards": 57,
            "Stunned": 27
        }
        
        if isfalldown:
            card_data = fall_data
        else:
            card_data = normal_data

        drawn_data_list = []
        drawn_ani_list = []
        ani_per_frame = {}
        available_cards = full_card_names.copy()  

        for key in card_data:
            ani_per_frame[key] = 0

        for _ in range(num_draws):
            if not available_cards:
                break  # 사용 가능한 카드가 없으면 중단
            
            # 합계가 낮은 애니 선택
            min_frame_ani = min(ani_per_frame, key=ani_per_frame.get)

            # 해당 애니 중 하나를 랜덤으로 선택
            possible_cards = [card for card in available_cards if card.split('_')[1] == min_frame_ani]
            selected_card = random.choice(possible_cards)

            drawn_data_list.append(selected_card)
            drawn_ani_list.append(selected_card.split('_')[1])
            ani_per_frame[min_frame_ani] = drawn_ani_list.count(min_frame_ani) * card_data[min_frame_ani]
            available_cards.remove(selected_card)
        return drawn_data_list, ani_per_frame

    # 낙상 리스트 불러오기 및 셔플, 밸런스 뽑기
    if num_fall != 0:
        fall_source = glob.glob(os.path.join(f_path,'*.csv'))
        fall_source = natsort.natsorted(fall_source)
        fcsv_list = extract_names(fall_source)

        blanced_data_list, ani_per_frame = draw_data_balanced(num_fall, fcsv_list, isfalldown=True)
        import json
        ret_dict = {}
        ret_dict['ani_fer_frame'] = ani_per_frame 

        with open(os.path.join(s_path, "F" + str(num_fall) + ".json"), 'w') as output:
            json.dump(ret_dict, output, indent=4)
        
        for name in blanced_data_list:
            csv_name = name+".csv"
            shutil.copy(os.path.join(f_path, csv_name), os.path.join(s_path, "F_" + csv_name))



    # 노멀 데이터 뽑기
    if num_normal != 0:
        normal_source = glob.glob(os.path.join(n_path,'*.csv'))
        normal_source = natsort.natsorted(normal_source)
        ncsv_list = extract_names(normal_source)

        blanced_data_list, ani_per_frame = draw_data_balanced(num_normal, ncsv_list)
        import json
        ret_dict = {}
        ret_dict['ani_fer_frame'] = ani_per_frame 

        with open(os.path.join(s_path, "N" + str(num_normal) + ".json"), 'w') as output:
            json.dump(ret_dict, output, indent=4)
        
        for name in blanced_data_list:
            csv_name = name+".csv"
            shutil.copy(os.path.join(n_path, csv_name), os.path.join(s_path, "N_" + csv_name))

    data_merge(s_path, os.path.join(w_path, out_file_name))

if __name__ == '__main__':
    f_path = "_SyntheticData/Fall_Data/formating"
    n_path = "_SyntheticData/Normal_Data/formating"
    s_path = "_SyntheticData/sample"
    w_path = "_SyntheticData"

    num_fall = 0
    num_normal = 0
    smooth_labels_step = 4

    # 샘플링 따로 빼놓기 TODO

    num_list = [
    25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350,
    50, 100, 150, 200, 250, 300, 350, 400, 450
]
    
    # for i in range(len(num_list)):
    #     s_path = "_SyntheticData/sampleN" + str(num_list[i])
    #     num_normal = num_list[i] 
        
    #     merge(w_path=w_path, f_path=f_path, n_path=n_path, s_path=s_path,
    #         num_fall=num_fall, num_normal=num_normal, out_file_name=file_name+".csv")
    

    # for i in range(len(num_list)):
    #     for j in range(len(num_list)):


    s_path = "_SyntheticData/sample"
    num_fall = 200
    num_normal = 200
    file_name = "Synthetic_Fall" + str(num_fall) + "_Normal" + str(num_normal) + "_LS" + str(smooth_labels_step)

    # merge(w_path=w_path, f_path=f_path, n_path=n_path, s_path=s_path,
    #     num_fall=num_fall, num_normal=num_normal, out_file_name=file_name+".csv")



    def write_pkl(csv_path):
        feature_set, labels_set = compose_and_label_smoothing(csv_path, smooth_labels_step=smooth_labels_step)

        pkl_name = file_name + '.pkl'
        pkl_path = os.path.join(w_path, pkl_name)
        with open(pkl_path, 'wb') as f:
            pickle.dump((feature_set, labels_set), f)
            
    csv_pose_file = os.path.join(w_path, file_name + '.csv')
    write_pkl(csv_pose_file)