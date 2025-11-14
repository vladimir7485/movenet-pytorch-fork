import torch
import cv2
import time
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt


from movenet.models.model_factory import load_model
from movenet.utils import _process_input, draw_skel_and_kp
# cropping related functions
# from movenet.utils import init_crop_region, determine_crop_region

# videopose
from poseaug.models.model_factory import load_model as load_model_pose
from poseaug.utils import create_2d_data, show3Dpose

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_path', type=str, required=True, help='Path to output video file')
parser.add_argument('--conf_thres', type=float, default=0.3)
parser.add_argument('--cropping', action='store_false')
args = parser.parse_args()
if args.model == "movenet_lightning":
    args.size = 192
    args.ft_size = 48
else:
    args.size = 256
    args.ft_size = 64


def main():
    model = load_model(args.model)
    pose_aug = load_model_pose().cuda().eval()
    model = model.cuda()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize matplotlib figure for 3D visualization
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    
    # Get 3D visualization dimensions (draw a dummy frame first)
    show3Dpose(np.zeros((16, 3)), ax)  # Dummy pose to get dimensions (16 keypoints for 3D model)
    fig.canvas.draw()
    # Compatibility with both old and new matplotlib versions
    try:
        # Old matplotlib API
        img_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    except AttributeError:
        # New matplotlib API (>= 3.5)
        buf = fig.canvas.buffer_rgba()
        img_3d = np.asarray(buf)
        img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2RGB)
    h_3d, w_3d = img_3d.shape[:2]
    ax.clear()
    
    # Calculate output dimensions (side by side: 3D on left, 2D on right)
    # Match heights by resizing 3D visualization to match video height
    if height != h_3d:
        scale = height / h_3d
        w_3d_scaled = int(w_3d * scale)
        h_3d_scaled = height
    else:
        w_3d_scaled = w_3d
        h_3d_scaled = h_3d
    
    # Output will be side by side
    output_width = w_3d_scaled + width
    output_height = height
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (output_width, output_height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {args.output_path}")
        cap.release()
        return

    start = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        input_image, display_image = _process_input(frame, size=args.size)
        
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()
            kpt_with_conf = model(input_image)[0, 0, :, :]
            inputs_2d = create_2d_data(kpt_with_conf) 
            outputs_3d = pose_aug(inputs_2d)
            outputs_3d = outputs_3d[:, :, :] - outputs_3d[:, :1, :]
            
        kpt_with_conf = kpt_with_conf.detach().cpu().numpy()
        outputs_3d = outputs_3d[0].detach().cpu().numpy() 

        show3Dpose(outputs_3d, ax)
        # redraw the canvas
        fig.canvas.draw()
        
        # convert canvas to image (compatibility with both old and new matplotlib)
        try:
            # Old matplotlib API
            img_3d = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img_3d = img_3d.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            # img is rgb, convert to opencv's default bgr
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGB2BGR)
        except AttributeError:
            # New matplotlib API (>= 3.5)
            buf = fig.canvas.buffer_rgba()
            img_3d = np.asarray(buf)
            # img is rgba, convert to bgr
            img_3d = cv2.cvtColor(img_3d, cv2.COLOR_RGBA2BGR)
        
        # Clear for next frame
        ax.clear()

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = draw_skel_and_kp(
            display_image, kpt_with_conf, conf_thres=args.conf_thres)

        # Resize images to match output dimensions
        img_3d = cv2.resize(img_3d, (w_3d_scaled, h_3d_scaled))
        overlay_image = cv2.resize(overlay_image, (width, output_height))
        
        # Combine side by side
        combined = np.hstack([img_3d, overlay_image])
        
        out.write(combined)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    print(f'\nProcessed {frame_count} frames')
    print('Average FPS: ', frame_count / (time.time() - start))
    cap.release()
    out.release()
    plt.close(fig)
    print(f"Output video saved to: {args.output_path}")


if __name__ == "__main__":
    main()

