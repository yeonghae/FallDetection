import os

def list_direct_subpaths(basepath):
    for entry in os.listdir(basepath):
        print(os.path.join(basepath, entry))

# 지정된 경로에서 바로 아래에 있는 파일과 디렉터리를 출력합니다.
path = '/home/workspace/Dataset/_testbad/falldown_1/'
list_direct_subpaths(path)