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
import torch.nn.functional as F
from torchvision.transforms import Compose
from Depth_Anything.depth_anything.dpt import DepthAnything
from Depth_Anything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

import json
import pickle
import pandas as pd
import openpyxl

FRAME_STEP = 14  ## 테스트용 임시 !! TMEP

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

def init_GT_workbook_and_sheet(gt_path=None, m_name="None"):
    if gt_path == None:
        return None, None

    wb = openpyxl.load_workbook(gt_path)
    ws = wb.active

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
    point_labels = np.array([1])

    masks, scores, _ = predictor.predict(
        box=input_box, 
        point_coords=point_coords, 
        point_labels=point_labels, 
        multimask_output=True
    )

    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]

def calculate_mean_depth(box, depth_map):
    """Calculate mean depth for a bounding box."""
    x1, y1, x2, y2 = map(int, box)
    cropped_depth = depth_map[y1:y2, x1:x2]
    mean_depth = np.mean(cropped_depth)
    return mean_depth

def filter_overlapping_objects(detections, depth_map):
    """Filter overlapping objects based on depth."""
    filtered_detections = []
    for i, det in enumerate(detections):
        keep = True
        for j, other_det in enumerate(detections):
            if i != j:
                x1, y1, x2, y2 = det[:4]
                ox1, oy1, ox2, oy2 = other_det[:4]

                inter_x1 = max(x1, ox1)
                inter_y1 = max(y1, oy1)
                inter_x2 = min(x2, ox2)
                inter_y2 = min(y2, oy2)

                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                area1 = (x2 - x1) * (y2 - y1)
                area2 = (ox2 - ox1) * (oy2 - oy1)
                union_area = area1 + area2 - inter_area

                if union_area > 0 and inter_area / union_area > 0.5:
                    depth1 = calculate_mean_depth((x1, y1, x2, y2), depth_map)
                    depth2 = calculate_mean_depth((ox1, oy1, ox2, oy2), depth_map)

                    if depth1 > depth2:
                        keep = False
                        break
        if keep:
            filtered_detections.append(det)
    return filtered_detections

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def preproc(image):
    resize_fn = ResizePadding(args.detection_input_size, args.detection_input_size)
    image = resize_fn(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def run_with_depth(source_path, model_path, out_dir="_ret", 
                   bb_load=False,
                   skeldir="skel", skel_mode=None, sheet=None, workbook=None):
    if check_skel_mode(skel_mode) == False:
        print("This mode is not supported.")
        return -1

    cam_source = source_path.replace('\\', '/')
    video_name = source_path.split('/')[-1].split('.')[0]
    model_name = model_path.split('/')[-1].split('.')[0]

    mout_dir = os.path.join(out_dir, model_name)
    vout_dir = os.path.join(mout_dir, video_name)
    viout_dir = os.path.join(vout_dir, 'img')
    create_folder(viout_dir)

    sam_checkpoint = "sam_vit_h_4b8939.pth"
    sam_model_type = "vit_h"
    sam = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
    sam.to(device=args.device)
    sam_predictor = SamPredictor(sam)

    depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14').to(device).eval()
    transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

    cam = CamLoader_Q(cam_source, queue_size=10000, preprocess=preproc).start()
    time.sleep(1)

    detect_model = YOLOv7(args.detection_input_size, device=args.device)
    pose_model = SPPE_FastPose(args.pose_backbone, int(args.pose_input_size.split('x')[0]), int(args.pose_input_size.split('x')[1]), device=args.device)
    action_model = TSSTG(weight_file=model_path)

    while cam.grabbed():
        frame = cam.getitem()
        transformed_frame = transform({'image': frame / 255.0})['image']
        transformed_frame = torch.from_numpy(transformed_frame).unsqueeze(0).to(device)

        with torch.no_grad():
            depth = depth_model(transformed_frame)

        depth = F.interpolate(depth[None], (frame.shape[0], frame.shape[1]), mode='bilinear', align_corners=False)[0, 0]
        depth_map = depth.cpu().numpy()

        detected = detect_model.detect(frame, need_resize=False, expand_bb=10)

        if detected is not None:
            filtered_detections = filter_overlapping_objects(detected, depth_map)

            for det in filtered_detections:
                x1, y1, x2, y2 = map(int, det[:4])
                input_box = np.array([x1, y1, x2, y2])
                sam_predictor.set_image(frame)
                mask = refine_segmentation_with_points(frame, sam_predictor, input_box)
                mask_resized = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
                segmented_person = cv2.bitwise_and(frame, frame, mask=mask_resized)

                poses = pose_model.predict(segmented_person, torch.tensor([[x1, y1, x2, y2, 0.9, 1.0, 0]], dtype=torch.float32)[:, :4], torch.tensor([[0.9]], dtype=torch.float32))

                for ps in poses:
                    pass  # Handle pose and action recognition logic

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.stop()
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

    skel_mode = 'SEKLETON_SAVE'

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

        run_with_depth(source_path=vid, model_path=model_path, out_dir=out_path, skel_mode=skel_mode, skeldir=skeldir, workbook=workbook, sheet=sheet)
