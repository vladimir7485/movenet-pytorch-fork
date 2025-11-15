#!/usr/bin/env python3
"""
Script to convert video files to MP4 format.
Reads the video file and converts it to MP4 using OpenCV.
"""

import cv2
import os
import argparse
from pathlib import Path


def convert_video_to_mp4(input_path, output_path=None):
    """
    Convert a video file to MP4 format.
    
    Args:
        input_path: Path to the input video file
        output_path: Path to the output MP4 file (optional, defaults to input_path with .mp4 extension)
    
    Returns:
        Path to the output file if successful, None otherwise
    """
    # Check if input file exists
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return None
    
    # Generate output path if not provided
    if output_path is None:
        input_path_obj = Path(input_path)
        output_path = str(input_path_obj.with_suffix('.mp4'))
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {input_path}")
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Input video properties:")
    print(f"  Resolution: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Total frames: {total_frames}")
    print(f"  Duration: {total_frames/fps:.2f} seconds")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video file {output_path}")
        cap.release()
        return None
    
    print(f"\nConverting video...")
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            print(f"Progress: {frame_count}/{total_frames} frames ({progress:.1f}%)", end='\r')
    
    print(f"\nConversion complete!")
    print(f"Processed {frame_count} frames")
    print(f"Output video saved to: {output_path}")
    
    cap.release()
    out.release()
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Convert video file to MP4 format')
    parser.add_argument('--input', type=str, 
                       default='videos/Screen Recording 2025-11-15 at 11.12.28.mov',
                       help='Path to input video file (default: videos/Screen Recording 2025-11-15 at 11.12.28.mov)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to output MP4 file (default: input path with .mp4 extension)')
    args = parser.parse_args()
    
    # Get the script directory and resolve relative paths
    script_dir = Path(__file__).parent.parent
    input_path = script_dir / args.input
    
    # Resolve output path
    if args.output:
        output_path = script_dir / args.output
    else:
        output_path = None
    
    convert_video_to_mp4(str(input_path), str(output_path) if output_path else None)


if __name__ == "__main__":
    main()

