#!/usr/bin/env python3
"""
Script to split a video horizontally (along X axis) into left and right parts.
Saves two output videos: <input_file_name>_left.mp4 and <input_file_name>_right.mp4
"""

import cv2
import os
import argparse
from pathlib import Path


def split_video_horizontal(input_path, split_percentage=50.0):
    """
    Split a video horizontally at a given percentage.
    
    Args:
        input_path: Path to the input video file
        split_percentage: Percentage from left edge to split (0-100, default: 50.0)
    
    Returns:
        Tuple of (left_output_path, right_output_path) if successful, (None, None) otherwise
    """
    # Validate split percentage
    if not 0 < split_percentage < 100:
        print(f"Error: Split percentage must be between 0 and 100, got {split_percentage}")
        return None, None
    
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return None, None
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None, None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate split point
    split_x = int(width * (split_percentage / 100.0))
    left_width = split_x
    right_width = width - split_x
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    print(f"\nSplit configuration:")
    print(f"  Split percentage: {split_percentage}%")
    print(f"  Split point (X): {split_x} pixels")
    print(f"  Left width: {left_width} pixels")
    print(f"  Right width: {right_width} pixels")
    
    # Generate output paths
    input_path_obj = Path(input_path)
    input_stem = input_path_obj.stem
    input_parent = input_path_obj.parent
    
    left_output_path = str(input_parent / f"{input_stem}_left.mp4")
    right_output_path = str(input_parent / f"{input_stem}_right.mp4")
    
    # Initialize video writers
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    left_out = cv2.VideoWriter(left_output_path, fourcc, fps, (left_width, height))
    right_out = cv2.VideoWriter(right_output_path, fourcc, fps, (right_width, height))
    
    if not left_out.isOpened():
        print(f"Error: Could not create left output video file {left_output_path}")
        cap.release()
        return None, None
    
    if not right_out.isOpened():
        print(f"Error: Could not create right output video file {right_output_path}")
        cap.release()
        left_out.release()
        return None, None
    
    print(f"\nSplitting video...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Split frame horizontally
        left_frame = frame[:, :split_x]  # Left part: from 0 to split_x
        right_frame = frame[:, split_x:]  # Right part: from split_x to end
        
        left_out.write(left_frame)
        right_out.write(right_frame)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')
    
    print(f"\nSplitting complete!")
    print(f"Processed {frame_count} frames")
    print(f"Left video saved to: {left_output_path}")
    print(f"Right video saved to: {right_output_path}")
    
    cap.release()
    left_out.release()
    right_out.release()
    
    return left_output_path, right_output_path


def main():
    parser = argparse.ArgumentParser(description='Split video horizontally into left and right parts')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input MP4 video file')
    parser.add_argument('--split', type=float, default=50.0,
                       help='Split percentage from left edge (0-100, default: 50.0)')
    args = parser.parse_args()
    
    # Get the script directory and resolve relative paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input if not os.path.isabs(args.input) else Path(args.input)
    
    split_video_horizontal(str(input_path), args.split)


if __name__ == "__main__":
    main()

