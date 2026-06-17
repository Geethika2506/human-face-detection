"""
Live Face Detection Demo for Student Attendance System

This script demonstrates the face detection module in real-time using a webcam.
It visualizes detected faces with bounding boxes and confidence scores.

Features:
    - Live webcam capture
    - Real-time face detection with bounding boxes
    - Confidence score display
    - FPS counter
    - Frame information logging
    - Ready-to-integrate with recognition module

Usage:
    python demo.py [--camera CAMERA_INDEX] [--confidence THRESHOLD]
    
    Press 'q' to quit
    Press 's' to save detected face crops to disk
"""

import cv2
import numpy as np
import argparse
from datetime import datetime
from pathlib import Path
import sys

from face_detection import FaceDetector, DetectionResult, Face


class FaceDetectionDemo:
    """
    Real-time face detection demo with visualization.
    """
    
    def __init__(self, camera_index: int = 0, confidence_threshold: float = 0.5):
        """
        Initialize the demo.
        
        Args:
            camera_index: Index of camera to use (0 for default)
            confidence_threshold: Minimum confidence for face detection
        """
        self.detector = FaceDetector(confidence_threshold=confidence_threshold)
        self.camera_index = camera_index
        self.frame_count = 0
        self.saved_faces_dir = Path('saved_faces')
        self.saved_faces_dir.mkdir(exist_ok=True)
        
        # FPS calculation
        self.fps_start_time = datetime.now()
        self.fps_frame_count = 0
        self.fps = 0
    
    def calculate_fps(self) -> float:
        """Calculate frames per second."""
        self.fps_frame_count += 1
        elapsed = (datetime.now() - self.fps_start_time).total_seconds()
        
        if elapsed >= 1.0:
            self.fps = self.fps_frame_count / elapsed
            self.fps_frame_count = 0
            self.fps_start_time = datetime.now()
        
        return self.fps
    
    def save_detected_faces(self, detection_result: DetectionResult) -> None:
        """
        Save all detected face crops to disk for inspection.
        Useful for building a training dataset for the recognition module.
        """
        for idx, face in enumerate(detection_result.detected_faces):
            filename = (
                self.saved_faces_dir / 
                f"face_{detection_result.frame_index}_{idx}_"
                f"conf_{face.confidence:.2f}.jpg"
            )
            cv2.imwrite(str(filename), face.crop)
            print(f"Saved: {filename}")
    
    def draw_detections(
        self,
        frame: np.ndarray,
        detection_result: DetectionResult
    ) -> np.ndarray:
        """
        Draw bounding boxes and confidence scores on the frame.
        
        Args:
            frame: Input frame
            detection_result: Detection results
        
        Returns:
            Annotated frame
        """
        annotated_frame = frame.copy()
        
        for face in detection_result.detected_faces:
            x, y, w, h = face.bbox
            
            # Draw bounding box
            color = (0, 255, 0)  # Green for detected faces
            thickness = 2
            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Draw confidence score label
            confidence_text = f"Conf: {face.confidence:.3f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5
            text_thickness = 1
            text_color = (0, 255, 0)
            
            # Background for text
            text_size = cv2.getTextSize(confidence_text, font, font_scale, text_thickness)[0]
            bg_x1, bg_y1 = x, y - text_size[1] - 4
            bg_x2, bg_y2 = x + text_size[0] + 4, y
            
            cv2.rectangle(annotated_frame, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 255, 0), -1)
            cv2.putText(
                annotated_frame,
                confidence_text,
                (x + 2, y - 2),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                text_thickness
            )
            
            # Draw face index
            face_index_text = f"ID: {len(detection_result.detected_faces) + 1}"
            cv2.putText(
                annotated_frame,
                face_index_text,
                (x, y + h + 15),
                font,
                font_scale,
                text_color,
                text_thickness
            )
        
        return annotated_frame
    
    def draw_info_panel(self, frame: np.ndarray, detection_result: DetectionResult) -> np.ndarray:
        """
        Draw information panel with statistics.
        """
        info_frame = frame.copy()
        
        # Prepare info text
        info_lines = [
            f"Frame: {detection_result.frame_index}",
            f"Faces Detected: {len(detection_result.detected_faces)}",
            f"FPS: {self.fps:.1f}",
            f"Detector: {self.detector.get_stats()['detector_type']}",
            f"Confidence Threshold: {self.detector.confidence_threshold:.2f}",
        ]
        
        if detection_result.detected_faces:
            avg_conf = np.mean([f.confidence for f in detection_result.detected_faces])
            info_lines.append(f"Avg Confidence: {avg_conf:.4f}")
        
        # Draw semi-transparent background for info panel
        panel_height = len(info_lines) * 20 + 10
        overlay = info_frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, panel_height), (0, 0, 0), -1)
        info_frame = cv2.addWeighted(overlay, 0.3, info_frame, 0.7, 0)
        
        # Draw text
        y_offset = 20
        for line in info_lines:
            cv2.putText(
                info_frame,
                line,
                (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
            y_offset += 20
        
        return info_frame
    
    def print_detection_summary(self, detection_result: DetectionResult) -> None:
        """Print detection summary to console."""
        print("\n" + "="*60)
        print(f"DETECTION SUMMARY - Frame {detection_result.frame_index}")
        print("="*60)
        print(f"Total faces detected: {len(detection_result.detected_faces)}")
        
        if detection_result.detected_faces:
            for idx, face in enumerate(detection_result.detected_faces):
                x, y, w, h = face.bbox
                print(f"\nFace {idx + 1}:")
                print(f"  Position: ({x}, {y})")
                print(f"  Size: {w}x{h} pixels")
                print(f"  Detection Confidence: {face.confidence:.4f}")
                print(f"  Crop Shape: {face.crop.shape}")
                print(f"  Status: Ready for recognition module ✓")
        else:
            print("No faces detected in this frame")
        
        print("="*60 + "\n")
    
    def run(self):
        """Run the live face detection demo."""
        cap = cv2.VideoCapture(self.camera_index)
        
        if not cap.isOpened():
            print(f"Error: Cannot open camera {self.camera_index}")
            return
        
        # Try to set higher resolution for better detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("=" * 60)
        print("FACE DETECTION DEMO - STUDENT ATTENDANCE SYSTEM")
        print("=" * 60)
        print("\nStarting webcam capture...")
        print("Press 'q' to quit")
        print("Press 's' to save detected face crops")
        print("-" * 60)
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Failed to read frame")
                    break
                
                # Run face detection
                detection_result = self.detector.detect(frame)
                
                # Calculate FPS
                fps = self.calculate_fps()
                
                # Draw detections on frame
                annotated_frame = self.draw_detections(frame, detection_result)
                
                # Draw info panel
                annotated_frame = self.draw_info_panel(annotated_frame, detection_result)
                
                # Print summary to console (optional - can be verbose)
                if len(detection_result.detected_faces) > 0:
                    self.print_detection_summary(detection_result)
                
                # Display the annotated frame
                cv2.imshow('Face Detection Demo', annotated_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    self.save_detected_faces(detection_result)
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released. Demo completed.")
            print(f"Total frames processed: {self.detector.frame_count}")
            print(f"Detector stats: {self.detector.get_stats()}")


def main():
    """Main entry point for the demo."""
    parser = argparse.ArgumentParser(
        description='Live face detection demo for student attendance system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                          # Use default camera
  python demo.py --camera 0               # Specify camera index
  python demo.py --confidence 0.6         # Raise confidence threshold
        """
    )
    
    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera index (default: 0)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help='Minimum confidence threshold (default: 0.5)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not 0.0 <= args.confidence <= 1.0:
        print("Error: Confidence must be between 0.0 and 1.0")
        sys.exit(1)
    
    # Run demo
    demo = FaceDetectionDemo(
        camera_index=args.camera,
        confidence_threshold=args.confidence
    )
    demo.run()


if __name__ == '__main__':
    main()
