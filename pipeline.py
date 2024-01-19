import cv2
import os
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.optical_flow import Raft_Small_Weights, raft_small
from ultralytics import YOLO


def preprocess(img1_batch, img2_batch):
    img1_batch = F.resize(img1_batch, size=[520, 960], antialias=False)
    img2_batch = F.resize(img2_batch, size=[520, 960], antialias=False)
    return transforms(img1_batch, img2_batch)

def cvrt(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame.transpose((2, 0, 1))).float() / 255.0
    return frame.to(device)


device = "cuda:0"
skip = 2

weight = Raft_Small_Weights.DEFAULT
transforms = weight.transforms()
raft = raft_small(weights=weight, progress=False).to(device).eval()
yolo = YOLO('yolov8m.pt').to(device)

file = '车牌检测'
cap = cv2.VideoCapture(file + '.mp4')
output_file = file + '_output.mp4'
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
fps, frame_width, frame_height = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
video_writer = cv2.VideoWriter(output_file, fourcc, fps, (int(frame_width), int(frame_height)))


# Pipeline
ret, frame = cap.read()
whole_width, whole_height = frame.shape[1], frame.shape[0]
frame_last = cvrt(frame)

max_dist = np.sqrt((whole_width/2)**2 + whole_height**2)

count = 0
while True:
    ret, frame = cap.read()
    if ret:
        # 跳帧，加快处理速度
        if count % skip != 0:
            count += 1
        else:
            count += 1
            # Optical Flow
            frame_now = cvrt(frame)
            img1_batch = torch.stack([frame_last])
            img2_batch = torch.stack([frame_now])
            frame_last = frame_now

            img1_batch, img2_batch = preprocess(img1_batch, img2_batch)
            flow = raft(img1_batch, img2_batch)[-1]
            flow = F.resize(flow, size=[whole_height, whole_width], antialias=False)[0]
            
            # filter
            flow[torch.abs(flow) < 0.1] = 0.0

            # YOLO
            result = yolo(frame)[0]
        
        # 遍历每个bbox，计算bbox内的光流平均值
        bbox_avg_flow = []
        bbox_flow = []
        idx = []
        for i, item in enumerate(result):
            bbox, conf, cls = item.boxes.xyxy, item.boxes.conf.item(), item.boxes.cls
            name = yolo.names[int(cls)]
            
            # filter
            if name not in ['car', 'truck', 'bus', 'van', 'SUV', ]:
                continue
            
            # get bbox
            x1, y1, x2, y2 = bbox[0].cpu().detach().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # 计算bbox的相对光流值，用bbox内光流平均值减去周围框的光流平均值，消除了背景光流（相机运动）
            size = 3
            flow_bbox = flow[:, y1:y2, x1:x2]
            flow_bbox_around = flow[:, y1-size:y2+size, x1-size:x2+size]
            bbox_pixel_num = flow_bbox[0].shape[0] * flow_bbox[0].shape[1]
            around_pixel_num = flow_bbox_around[0].shape[0] * flow_bbox_around[0].shape[1] - bbox_pixel_num
            horizon_around_avg = (flow_bbox_around[0].sum() - flow_bbox[0].sum()) / around_pixel_num
            vertical_around_avg = (flow_bbox_around[1].sum() - flow_bbox[1].sum()) / around_pixel_num
            horizon_bbox_avg = flow_bbox[0].sum() / bbox_pixel_num
            vertical_bbox_avg = flow_bbox[1].sum() / bbox_pixel_num
            horizon = (horizon_bbox_avg - horizon_around_avg)
            vertical = (vertical_bbox_avg - vertical_around_avg)
            
            # 保存bbox的光流值和bbox的平均光流值
            bbox_flow.append([horizon, vertical])
            bbox_avg_flow.append([horizon, vertical])
            idx.append(i)
        
        # # 计算所有bbox的光流值，如果绝对值大于平均值则减去平均值，相当于消除了相机运动；小于平均值则说明是静止的车，不减
        # bbox_avg_flow = torch.mean(torch.tensor(bbox_avg_flow), dim=0)
        # bbox_avg_flow_value = torch.sqrt(bbox_avg_flow[0]**2 + bbox_avg_flow[1]**2)
        # for i, id in enumerate(idx):
        #     item = result[id]
        #     bbox, conf, cls = item.boxes.xyxy, item.boxes.conf.item(), item.boxes.cls
        #     name = yolo.names[int(cls)]
            
        #     # filter
        #     if name not in ['car', 'truck', 'bus', 'van', 'SUV', ]:
        #         continue
            
            # get bbox
            x1, y1, x2, y2 = bbox[0].cpu().detach().numpy()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            
            move_value = torch.sqrt(horizon**2 + vertical**2)
            # if move_value > bbox_avg_flow_value:
            #     horizon = bbox_flow[i][0] - bbox_avg_flow[0]
            #     vertical = bbox_flow[i][1] - bbox_avg_flow[1]
                
            # 根据相对光流值的大小判断是否静止
            if move_value < 0.04:
                stop = True
            elif move_value < 20.0:
                horizon, vertical = horizon * (20-move_value), vertical * (20-move_value) # 为了视觉效果，速度慢的箭头给放大点
                stop = False
            else:
                stop = False
            
            # 根据光流方向画箭头, 画在bbox的中心
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            cv2.arrowedLine(frame, (center_x, center_y), (center_x + int(horizon), center_y + int(vertical)), (0, 0, 255), 2)
            
            # draw bbox & label
            if stop:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                cv2.putText(frame, f'stop', (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
                cv2.putText(frame, f'{name}', (x1, y1+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('frame', frame)
        video_writer.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
    
video_writer.release()
cap.release()
cv2.destroyAllWindows()


# 车牌检测
os.system(f'python plate_dect/detect.py --source {output_file} --output {output_file}_final --is_color False')