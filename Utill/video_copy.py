import os
import shutil

def copy_video_files(src_folder, dst_folder, video_extensions):
    for root, dirs, files in os.walk(src_folder):
        for file in files:
            if file.endswith(tuple(video_extensions)):
                src_path = os.path.join(root, file)
                dst_path = os.path.join(dst_folder, file)
                shutil.copy2(src_path, dst_path)
                print(f"Copied {src_path} to {dst_path}")

# 설정: 원하는 비디오 파일 확장자를 지정하세요
video_extensions = ['.mp4', '.avi']  # 예시 확장자

# 실행: 현재 폴더로 복사
current_folder = os.getcwd()
source_folder = "/home/workspace/_ret/F300_N200_LS4_NF20_EP50_BS256_UseShotFrameAllFallLabel_UseEvalNF17" 
copy_video_files(source_folder, current_folder, video_extensions)
