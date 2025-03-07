import os
import cv2
import time
import torch
import argparse
import numpy as np
import glob
import natsort

from Detection.Utils import ResizePadding
from CameraLoader import CamLoader, CamLoader_Q
from DetectorLoader import TinyYOLOv3_onecls, YOLOv7

from PoseEstimateLoader import SPPE_FastPose
from fn import draw_single

from Track.Tracker import Detection, Tracker
from ActionsEstLoader import TSSTG

from segment_anything import sam_model_registry, SamPredictor

import json
import pickle

import pandas as pd 

FRAME_STEP = 14 ## 테스트용 임시 !! TMEP

def kpt2bbox(kpt, ex=20):
    """Get bbox that hold on all of the keypoints (x,y)
    kpt: array of shape `(N, 2)`,
    ex: (int) expand bounding box,
    """
    return np.array((kpt[:, 0].min() - ex, kpt[:, 1].min() - ex,
                     kpt[:, 0].max() + ex, kpt[:, 1].max() + ex))

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

def check_skel_mode(mode):
    mode_list = ['SEKLETON_SAVE', 'SEKLETON_LOAD']

    if mode in mode_list: 
        return True
    else: 
        return False

## ! 임시. TEMP EVAL 쪽으로 이관
def init_GT_workbook_and_sheet(gt_path=None, m_name="None"):
    import openpyxl

    if gt_path == None:
        return None, None

    wb = openpyxl.load_workbook(gt_path)
    ws = wb.active

    # 'Video', 'Frame', 'GT_cls2', 'GT_cls3' 원래 있던 데이터
    headers = [m_name+'_Prediction', 'temp_video', 'temp_frame']
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(headers)

    return workbook, sheet

def refine_segmentation_with_points(frame, predictor, input_box):
    """Refine segmentation using SAM based on the input bounding box."""
    center_x = (input_box[0] + input_box[2]) // 2
    center_y = (input_box[1] + input_box[3]) // 2
    point_coords = np.array([[center_x, center_y]])
    point_labels = np.array([1])  # 포인트는 포함 영역으로 표시

    masks, scores, _ = predictor.predict(
        box=input_box, 
        point_coords=point_coords, 
        point_labels=point_labels, 
        multimask_output=True
    )

    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]

def run(source_path, model_path, out_dir="_ret", 
        bb_load=False,
        skeldir="skel", skel_mode=None, sheet=None, workbook=None):
    
    # 파라미터 체크 TODO 설정 값 관리하는 법 고민해보기
    if check_skel_mode(skel_mode) == False:
        print("This mode is not supported.")
        return -1

    cam_source = source_path.replace('\\', '/')
    video_name = source_path.split('/')[-1].split('.')[0]
    model_name = model_path.split('/')[-1].split('.')[0]
   
    xml_dir = os.path.join(source_path.split(video_name)[0], video_name)

    mout_dir = os.path.join(out_dir, model_name) # 모델 이름
    vout_dir = os.path.join(mout_dir, video_name) # 모델 이름-비디오 이름
    vout_path = vi_out_dir = os.path.join(vout_dir, args.save_out) # 모델 이름-비디오 이름
    viout_dir = os.path.join(vout_dir, 'img') # 모델 이름-비디오 이름-img
    create_folder(viout_dir)

    # SAM 모델 로드
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)

    #init with args
    inp_dets = args.detection_input_size
   
    resize_fn = ResizePadding(inp_dets, inp_dets) # 디텍션 모델 인풋 사이즈에 맞게 크기 리사이징

    def preproc(image):
        """preprocess function for CameraLoader.
        """
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

##########################
    inp_dets = args.detection_input_size
    inp_pose = args.pose_input_size.split('x')
    pose_backbone = args.pose_backbone
    device = args.device

    # Actions Estimate
    action_model = TSSTG(weight_file=model_path)
    resize_fn = ResizePadding(inp_dets, inp_dets) # 디텍션 모델 인풋 사이즈에 맞게 크기 리사이징
########################################################

    def preproc(image):
        """preprocess function for CameraLoader.
        """
        image = resize_fn(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    # Tracker
    tracker = Tracker(max_age=FRAME_STEP, n_init=3)

    action = 'pending..'
    action_name = 'pending..'
    fps_time = 0
    f = 0
    y_pred = []    
    l_ret = -1

    create_folder(skeldir)

    detect_model = YOLOv7(inp_dets, device=device)

    # POSE MODEL
    def init_pose_model(_param):
        pos = (int(_param[0]), int(_param[1]))
        return SPPE_FastPose(pose_backbone, pos[0], pos[1], device=device)

    pose_model = init_pose_model(inp_pose)

    # Use loader thread with Q for video file.
    cam = CamLoader_Q(cam_source, queue_size=10000, preprocess=preproc).start()
    time.sleep(1)

    outvid = False
    if args.save_out != '':
        outvid = True
        codec = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(vout_path, codec, 30, (inp_dets * 2, inp_dets * 2))

    detections_dict = {}
    while cam.grabbed():
        f += 1
        frame = cam.getitem()
        image = frame.copy()

        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        tracker.predict()

        detections = []  # List of Detections object for tracking.
        if detected is not None:
            # SAM 기반 세분화 및 자세 추정
            for det in detected:
                x1, y1, x2, y2 = map(int, det[:4])
                input_box = np.array([x1, y1, x2, y2])

                sam_predictor.set_image(frame)
                mask = refine_segmentation_with_points(frame, sam_predictor, input_box)
                mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                segmented_person = cv2.bitwise_and(frame, frame, mask=mask_resized)

                poses = pose_model.predict(segmented_person, torch.tensor([[x1, y1, x2, y2, 0.9, 1.0, 0]], dtype=torch.float32)[:, :4], torch.tensor([[0.9]], dtype=torch.float32))

                for ps in poses:
                    detections.append(Detection(kpt2bbox(ps['keypoints'].numpy()),
                                                np.concatenate((ps['keypoints'].numpy(),
                                                                ps['kp_score'].numpy()), axis=1),
                                                ps['kp_score'].mean().numpy()))

        detections_dict[f] = detections
        tracker.update(detections)

        for i, track in enumerate(tracker.tracks):
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            bbox = track.to_tlbr().astype(int)
            center = track.get_center().astype(int)

            action = 'pending..'
            action_name = "pending.."

            clr = (0, 255, 0)

            if len(track.keypoints_list) == FRAME_STEP:
                pts = np.array(track.keypoints_list, dtype=np.float32)
                out = action_model.predict(pts, frame.shape[:2])
                action_name = action_model.class_names[out[0].argmax()]
                action = '{}: {:.2f}%'.format(action_name, out[0].max() * 100)

                if action_name == 'Fall Down':
                    clr = (255, 0, 0)
                elif action_name == 'Lying Down':
                    clr = (255, 200, 0)

            if track.time_since_update == 0:
                if args.show_skeleton:
                    frame = draw_single(frame, track.keypoints_list[-1])
                frame = cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), clr, 1)
                frame = cv2.putText(frame, str(track_id), (center[0], center[1]), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, (255, 0, 0), 2)
                frame = cv2.putText(frame, action, (bbox[0] + 5, bbox[1] + 15), cv2.FONT_HERSHEY_COMPLEX,
                                    0.4, clr, 1)

        frame = cv2.resize(frame, (0, 0), fx=2., fy=2.)
        frame = cv2.putText(frame, '%d, FPS: %f' % (f, 1.0 / (time.time() - fps_time)),
                            (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        frame = frame[:, :, ::-1]
        fps_time = time.time()

        if outvid:
            cv2.imwrite(os.path.join(viout_dir, str(f).zfill(4) + ".png"), frame)
            writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if workbook != None:   
        workbook.save(os.path.join(out_dir, f"{video_name}_results.xlsx"))

    cam.stop()

    if outvid:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    testdata_dir = 'Dataset/test/'
    gt_path = 'Dataset/prison_test_data/GT.xlsx'

    vtypes = ['*.mp4', '*.avi']
    video_list = []
    for vt in vtypes:
        video_list.extend(glob.glob(os.path.join(testdata_dir, vt)))
    video_list = natsort.natsorted(video_list)

    model_dir = 'Models/falldown'
    base_fn = "__Fall200_Normal200_EP300_BS256_LS4"

    model_path = os.path.join(model_dir, base_fn+'.pth')

    out_path = '_ret'

    m_name = model_path.split('/')[-1].split('.')[0]

    workbook, sheet = init_GT_workbook_and_sheet(gt_path=gt_path, m_name=m_name)

    skel_mode = 'SEKLETON_SAVE' #SEKLETON_SAVE, SEKLETON_LOAD

    if skel_mode == 'SEKLETON_SAVE':
        skeldir = '_ret'
    elif skel_mode == 'SEKLETON_LOAD':
        skeldir = 'Dataset/skeleton'       
    else:
        print("skel mode err")
        exit()

    skel = "test"
    skeldir = os.path.join(skeldir, skel)

    for vid in video_list:
        v_name = vid.split('/')[-1].split('.')[0]
        outv_name = v_name + "_" + m_name + ".mp4"

        par = argparse.ArgumentParser(description='Human Fall Detection Demo.')
        par.add_argument('--detection_input_size', type=int, default=1088, help='Size of input in detection model in square must be divisible by 32 (int).')
        par.add_argument('--pose_input_size', type=str, default='288x160', help='Size of input in pose model must be divisible by 32 (h, w)')
        par.add_argument('--pose_backbone', type=str, default='resnet50', help='Backbone model for SPPE FastPose model.')
        par.add_argument('--show_skeleton', default=True, action='store_true', help='Show skeleton pose.')
        par.add_argument('--save_out', type=str, default=outv_name, help='Save display to video file.')
        par.add_argument('--device', type=str, default='cuda', help='Device to run model on cpu or cuda.')
        args = par.parse_args()

        run(source_path=vid, model_path=model_path, out_dir=out_path, skel_mode=skel_mode, skeldir=skeldir, workbook=workbook, sheet=sheet)
