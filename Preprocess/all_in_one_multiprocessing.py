import os
import glob
import natsort
import random
import shutil
import pandas as pd
import natsort

subdir_list = ['labeling', 'formating']
# use_shotframe: 프레임 구간을 shot_frame으로 사용
def label_calc(raw_label_path="Dataset/synthetic_data_sample/animation_label.xlsx",
               working_path='', res_w=1920, res_h=1080, use_shotframe=True, use_fallframe=True):
    ERR_LOG = []

    data_path = raw_label_path.split('/animation_label.xlsx')[0]
    flabel_path = os.path.join(data_path, "Fall_Data")
    nlabel_path = os.path.join(data_path, "Normal_Data")

    # 작업 폴더 생성
    def rm_subdir(w_path):
        
        for subdir_name in subdir_list:
            sub_path = os.path.join(w_path, subdir_name)
            if os.path.exists(sub_path):
                shutil.rmtree(sub_path)
            os.makedirs(sub_path)

    w_f_path = os.path.join(working_path, "Fall_Data")
    w_n_path = os.path.join(working_path, "Normal_data")
    
    rm_subdir(w_f_path)
    rm_subdir(w_n_path)

    df = pd.read_excel(raw_label_path)

    fall_df = df[df['ani_label'] == 'F']
    normal_df = df[df['ani_label'] == 'N']


    def get_filelist(path):
        file_list = [f for f in os.listdir(path) if not os.path.isdir(os.path.join(path, f))]
        file_list = natsort.natsorted(file_list) # 파일명 정렬
        return file_list

    # 영상 프레임 길이로 파일명 변경하기
    def rename_with_label(data_path, param_df, w_path=working_path, use_all_falldown=False):
        NUM_OF_FRAME_PER_ROW = 13
        param_df = param_df.copy()

        csv_list = get_filelist(path=data_path)
        for fname in csv_list:
            
            try:
                file_path = os.path.join(data_path, fname)
                csv_df = pd.read_csv(file_path)            
            except:
                print(fname, "CSV 검증에러","파일 읽기 오류")
                continue
            
            # 애니메이션의 마지막 프레임
            last_frame = int(csv_df['frame'][-1:])
            
            # csv 가공
            before_len = len(csv_df)

            # 초반, 후반 1 프레임 제거 
            # 방법 1. 스켈레톤 높이
            # new_df = csv_df[0 < csv_df['y']].copy()

            #방법 2. 프레임 값
            new_df = csv_df[(1 < csv_df['frame']) & ((before_len/13) > csv_df['frame'])].copy()


            new_df['x'] = new_df['x']/res_w
            new_df['y'] = new_df['y']/res_h
            new_df['z'] = 1
            new_df['label'] = 0
            new_len = len(new_df)
            
            # 검증
            if (before_len - new_len) != NUM_OF_FRAME_PER_ROW * 2: # 스켈레톤 당 13 row, 더미 프레임 2프레임이 제거 되므로 26 row가 삭제될 것이라 기대
                print(fname, "CSV 검증에러", "더미 프레임 실패")
                continue               
            try:
                # 애니메이션 프레임과 csv의 마지막 프레임이 일치하는 경우
                
                animation_name = param_df.loc[param_df['total_frame'] == last_frame, 'animation'].values[0]
                new_file_path = os.path.join(w_path,subdir_list[0], fname.split('.')[0].zfill(5) + "_" + animation_name + ".csv")
            except:
                print(fname, "CSV 검증에러","프레임으로 애니메이션 식별 실패")
                continue
            
            # 라벨링 구간 초기화
            ani_df = param_df[param_df['animation'] == animation_name]

            normal_start = int(ani_df['normal_start'].values[0])
            normal_end = int(ani_df['normal_end'].values[0])
            
            fall_start = int(ani_df['fall_start'].values[0])
            fall_end = int(ani_df['fall_end'].values[0])
            
            lying_start = int(ani_df['lying_start'].values[0])
            lying_end = int(ani_df['lying_end'].values[0])
            
            # 프레임 구간별 라벨링 삽입
            if(use_all_falldown):
                for l in range(len(new_df['frame'])):
                    index = l+NUM_OF_FRAME_PER_ROW # 첫프레임이 제거 되었기 때문에 건너뜀
                    frame = new_df['frame'][index]
                    if normal_start <= frame and frame <= normal_end:
                        new_df.at[index, 'label'] = int(1) # 노멀
                    elif fall_start <= frame and frame <= fall_end:
                        new_df.at[index, 'label'] = int(1) # 낙상
                    elif lying_start <= frame and frame <= lying_end: 
                        new_df.at[index, 'label'] = int(1) # 누움 ##!!!! 0으로 임시로 이진분류

                if(use_shotframe):
                    total_frame = int(ani_df['total_frame'].values[0])
                    short_start = int(ani_df['short_start'].values[0])
                    short_end = int(ani_df['short_end'].values[0])               
                    # 값 클램핑
                    #앞뒤 프레임 1프레임 씩 제거하기 떄문에 totla_frame-2가 최대 크기임
                    if(total_frame-2 < normal_end):
                        normal_end = total_frame -2
                    if(short_start < 2):
                        short_start = 2               
                    new_df = new_df[(short_start <= new_df['frame']) & (new_df['frame'] <= short_end)]
                    
                    # 프레임 2부터 시작되도록 번호 맞추기
                    new_df['frame'] = new_df['frame'] - short_start + 2
                elif(use_fallframe):
                    fall_start = int(ani_df['fall_start'].values[0]) 
                    fall_end = int(ani_df['fall_end'].values[0]) 
                    new_df = new_df[(fall_start <= new_df['frame']) & (new_df['frame'] <= fall_end)]
                    
                    # 프레임 2부터 시작되도록 번호 맞추기
                    new_df['frame'] = new_df['frame'] - fall_start + 2  
                
            else:
                for l in range(len(new_df['frame'])):
                    index = l+NUM_OF_FRAME_PER_ROW # 첫프레임이 제거 되었기 때문에 건너뜀
                    frame = new_df['frame'][index]
                    if normal_start <= frame and frame <= normal_end:
                        new_df.at[index, 'label'] = int(0) # 노멀
                    elif fall_start <= frame and frame <= fall_end:
                        new_df.at[index, 'label'] = int(1) # 낙상
                    elif lying_start <= frame and frame <= lying_end: 
                        new_df.at[index, 'label'] = int(0) # 누움 ##!!!! 0으로 임시로 이진분류             

            # 다 처리하고 2부터 시작하는 프레임을 1부터 시작하도록 -1 
            new_df['frame'] = new_df['frame'] - 1
            new_df.to_csv(new_file_path, index = False, header = True)

    print("낙상 데이터 라벨 매칭")
    # rename_with_label(flabel_path, fall_df, w_path=w_f_path, use_all_falldown=True)
    print("노멀 데이터 라벨 매칭")
    rename_with_label(nlabel_path, normal_df, w_path=w_n_path)

    return 0

# S1
def formating(input_path):
    data_path = os.path.join(input_path, subdir_list[0])
    for filename in os.listdir(data_path):
        if filename.endswith('.csv'):
            f_path = os.path.join(data_path, filename)
            output_path =  os.path.join(input_path, 'formating', filename)
            
        data = []
        with open(f_path, 'r') as input_file:
            for line in input_file:
                if not line.startswith('frame,x,y,z,label'):
                    row = line.strip().split(',')  # 데이터는 탭으로 구분되어 있으므로 탭으로 분리합니다.
                    data.append(row)

        output = {}

        for row in data:
            frame = int(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            label = int(row[4])
            
            if frame not in output:
                output[frame] = []

            # 각 프레임별로 13가지 스켈레톤 데이터(x,y,z)를 1줄 저장
            output[frame].extend([x, y, z])
            
            # 프레임별로 레이블을 한 번만 저장
            if len(output[frame]) == 39:  # 13개의 스켈레톤 데이터(x,y,z)가 모두 저장된 후
                output[frame].append(label)  # 레이블을 추가

        with open(output_path, 'w') as output_file:
            header = 'frame,Nose_x,Nose_y,Nose_z,LShoulder_x,LShoulder_y,LShoulder_z,RShoulder_x,RShoulder_y,RShoulder_z,LElbow_x,LElbow_y,LElbow_z,RElbow_x,RElbow_y,RElbow_z,LWrist_x,LWrist_y,LWrist_z,RWrist_x,RWrist_y,RWrist_z,LHip_x,LHip_y,LHip_z,RHip_x,RHip_y,RHip_z,LKnee_x,LKnee_y,LKnee_z,RKnee_x,RKnee_y,RKnee_z,LAnkle_x,LAnkle_y,LAnkle_z,RAnkle_x,RAnkle_y,RAnkle_z,label'
            output_file.write(header + '\n')
            
            for frame, values in output.items():
                values_str = ",".join(map(str, [frame] + values))
                output_file.write(f'{values_str}\n')

    print("formating Done")

def create_folder(path):
    """
    Creates a new folder at the specified path if it doesn't already exist.
    """
    try:
        # Check if the folder already exists
        if not os.path.exists(path):
            os.makedirs(path)
            return f"Folder created at: {path}"
        else:
            return f"Folder already exists at: {path}"
    except Exception as e:
        return f"An error occurred: {e}"

if __name__ == '__main__':


    working_path = "_SyntheticData/"
    raw_label_path = "Dataset/_synthetic_sit/animation_label.xlsx" # 1만개 버전

    # 이 두개를 나눌 필요가 있나?
    print("프레임 라벨 매칭")
    frame_options = []
    if(label_calc(raw_label_path=raw_label_path, working_path=working_path, use_shotframe=False, use_fallframe=True) != 0):
        print("label_calc error")

    print("포멧팅")
    f_path = working_path + "Fall_Data/"
    n_path = working_path + "Normal_Data/"
    formating(f_path)
    formating(n_path)