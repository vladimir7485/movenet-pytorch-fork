import torch
import cv2
import time
import argparse

from movenet.models.model_factory import load_model
from movenet.utils import _process_input, draw_skel_and_kp
# cropping related functions
# from movenet.utils import init_crop_region, determine_crop_region

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="movenet_lightning", choices=["movenet_lightning", "movenet_thunder"])
parser.add_argument('--video_path', type=str, required=True, help='Path to input video file')
parser.add_argument('--output_path', type=str, required=True, help='Path to output video file')
parser.add_argument('--conf_thres', type=float, default=0.3)
args = parser.parse_args()

if args.model == "movenet_lightning":
    args.size = 192
    args.ft_size = 48
else:
    args.size = 256
    args.ft_size = 64

def main():
    model = load_model(args.model, ft_size=args.ft_size)
    # model = model.cuda()

    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {args.video_path}")
        return

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
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
            input_image = torch.Tensor(input_image) #.cuda()
            kpt_with_conf = model(input_image)[0, 0, :, :]
            kpt_with_conf = kpt_with_conf.numpy()

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = draw_skel_and_kp(
            display_image, kpt_with_conf, conf_thres=args.conf_thres)

        # Resize overlay_image to match original video dimensions if needed
        if overlay_image.shape[:2] != (height, width):
            overlay_image = cv2.resize(overlay_image, (width, height))
        
        out.write(overlay_image)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...", end='\r')

    print(f'\nProcessed {frame_count} frames')
    print('Average FPS: ', frame_count / (time.time() - start))
    cap.release()
    out.release()
    print(f"Output video saved to: {args.output_path}")


if __name__ == "__main__":
    main()

