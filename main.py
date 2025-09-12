import argparse
import os
import numpy as np
import supervision as sv

from ultralytics import YOLO

model = YOLO("yolo11l.pt")

def parse_coordinates(coord_str):
    """Parse coordinate string 'x,y' to tuple (x, y)"""
    try:
        x, y = map(int, coord_str.split(','))
        return (x, y)
    except:
        raise ValueError(f"Invalid coordinate format: {coord_str}. Expected format: 'x,y'")

def parse_arguments():
    """Parse command line arguments and environment variables"""
    parser = argparse.ArgumentParser(description='Person counting with YOLO')

    # Add arguments with defaults from environment variables
    parser.add_argument('--video-path', 
                       type=str,
                       default=os.getenv('VIDEO_PATH'),
                       help='Path to input video file (required)')
    parser.add_argument('--debug', 
                       type=str,
                       default=os.getenv('DEBUG', 'false').lower() in ('true', '1', 'yes'),
                       help='Enable debug mode (true/false)')
    parser.add_argument('--line-start', 
                       type=str,
                       default=os.getenv('LINE_START'),
                       help='Start coordinates of counting line (x,y) (required)')

    parser.add_argument('--line-end',
                       type=str,
                       default=os.getenv('LINE_END'), 
                       help='End coordinates of counting line (x,y) (required)')

    args = parser.parse_args()

    # Check required arguments
    missing_args = []
    if not args.video_path:
        missing_args.append('video-path (--video-path or args.video_path env var)')
    
    if missing_args:
        parser.error(f"Missing required arguments: {', '.join(missing_args)}")

    args.line_start = parse_coordinates(args.line_start)
    args.line_end = parse_coordinates(args.line_end)

    return args

# Parse arguments and environment variables
args = parse_arguments()

START = sv.Point(*args.line_start)
END = sv.Point(*args.line_end)

line_zone = sv.LineZone(start=START, end=END)
byte_tracker = sv.ByteTrack()
if args.debug:
    line_zone_annotator = sv.LineZoneAnnotator(
        thickness=4,
        text_thickness=4,
        text_scale=2)
    bounding_box_annotator = sv.BoxAnnotator(thickness=4)
    label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)

def callback(frame: np.ndarray, index:int) -> np.ndarray:
    results = model(frame, imgsz=640, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)

    if args.debug:
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(detections.confidence, detections.class_id, detections.tracker_id)
        ]

        annotated_frame = frame.copy()
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame,
            detections=detections)
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame,
            detections=detections,
            labels=labels)

    line_zone.trigger(detections)

    if args.debug:
        return  line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
    else:
        return frame

if args.debug:
    sv.process_video(
        source_path = args.video_path,
        target_path = f"debug-{os.path.basename(args.video_path)}",
        callback=callback,
        show_progress=True,
    )
else:
    # Process video without output
    video_info = sv.VideoInfo.from_video_path(args.video_path)
    frame_generator = sv.get_video_frames_generator(args.video_path)

    for frame in frame_generator:
        callback(frame, 0)

print(f"line_zone.in_count: {line_zone.in_count}")
print(f"line_zone.out_count: {line_zone.out_count}")
