import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# use bfloat16 for the entire notebook
torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

from sam2.build_sam import build_sam2_video_predictor

sam2_checkpoint = "checkpoints/sam2_hiera_large.pt"
model_cfg = "sam2_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
video_dir = "video/lecture_frames"

# 영상을 받으면 1초단위로 샘플링 후 video JPEG로 저장

import cv2

def sample_video_frames(video_path, output_dir, interval=1):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video.")
        return
    
    # Get the frame rate of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate the interval in frames
    frame_interval = int(fps * interval)
    
    frame_count = 0
    saved_count = 0

    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Check if this frame is to be saved
        if frame_count % frame_interval == 0:
            # Save the frame as JPEG with a zero-padded filename
            frame_filename = os.path.join(output_dir, f"frame{saved_count:08d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        
        frame_count += 1

    # Release the video capture object
    cap.release()
    print(f"Saved {saved_count} frames to {output_dir}")

# Usage example
video_path = "video/0811.mp4"  # Path to your video file
sample_video_frames(video_path, video_dir)


# scan all the JPEG frame names in this directory
frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))


# 첫번째 프레임부터 사람 감지, 박스 
from ultralytics import YOLOv10

model = YOLOv10('../weights/yolov10m.pt')

# Initialize variables to store the bounding box and frame index
box = None
ann_frame_idx = None

# 첫 프레임부터 확인, 사람이 한사람만 발견되는 경우 break, 0명이거나 여러사람일 경우 넘기기, 해당 박스와 프레임 번호 변수로 저장 
# ex) box = np.array([300, 0, 500, 400], dtype=np.float32)
# ann_frame_idx = 0  # the frame index we interact with
for idx, frame in enumerate(frame_names):
    # Run the model prediction
    results = model(frame)

    # Extract the results
    boxes = results.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
    classes = results.boxes.cls.cpu().numpy()  # Class indices

    # Filter boxes for the 'person' class (usually class index 0 in COCO dataset)
    person_boxes = [box for box, cls in zip(boxes, classes) if cls == 0]

    # Check if exactly one person is detected
    if len(person_boxes) == 1:
        # Save the bounding box and frame index
        box = np.array(person_boxes[0], dtype=np.float32)
        ann_frame_idx = idx
        break

# Output the result
if box is not None and ann_frame_idx is not None:
    print(f"Person detected at frame {ann_frame_idx} with bounding box: {box}")
else:
    print("No frame with exactly one person detected.")



inference_state = predictor.init_state(video_path=video_dir)

predictor.reset_state(inference_state)


ann_obj_id = 4  # give a unique id to each object we interact with (it can be any integers)

# Let's add a box at (x_min, y_min, x_max, y_max) = (300, 0, 500, 400) to get started

_, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
    inference_state=inference_state,
    frame_idx=ann_frame_idx,
    obj_id=ann_obj_id,
    box=box,
)

# show the results on the current (interacted) frame
plt.figure(figsize=(12, 8))
plt.title(f"frame {ann_frame_idx}")
plt.imshow(Image.open(os.path.join(video_dir, frame_names[ann_frame_idx])))
show_box(box, plt.gca())
show_mask((out_mask_logits[0] > 0.0).cpu().numpy(), plt.gca(), obj_id=out_obj_ids[0])

# run propagation throughout the video and collect the results in a dict
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }

# render the segmentation results every few frames
vis_frame_stride = 15
plt.close("all")
for out_frame_idx in range(0, len(frame_names), vis_frame_stride):
    plt.figure(figsize=(6, 4))
    plt.title(f"frame {out_frame_idx}")
    plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
    for out_obj_id, out_mask in video_segments[out_frame_idx].items():
        show_mask(out_mask, plt.gca(), obj_id=out_obj_id)

