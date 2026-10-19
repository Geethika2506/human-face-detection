"""
Advanced Usage Examples for Face Detection Module

The recognition demo is documented in README.md. Run it with: python demo.py

This file contains practical examples demonstrating various use cases:
1. Batch processing video files
2. Custom confidence thresholding per frame
3. Face tracking across frames
4. Integration points for recognition module
5. Building a training dataset for recognition model
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from face_detection import FaceDetector, DetectionResult, AttendanceLogger


# ==============================================================================
# Example 1: Process Multiple Video Files and Extract Statistics
# ==============================================================================

def batch_process_videos(video_directory: str, output_directory: str):
    """
    Process multiple video files and extract face detection statistics.
    
    This example is useful for:
    - Analyzing classroom videos for attendance accuracy
    - Building datasets for recognition model training
    - Benchmarking detector performance
    
    Args:
        video_directory: Directory containing .mp4 or .avi files
        output_directory: Where to save extracted faces
    """
    video_dir = Path(video_directory)
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    detector = FaceDetector(confidence_threshold=0.6)
    
    stats = {
        'total_frames': 0,
        'total_faces': 0,
        'videos_processed': 0,
        'avg_confidence': []
    }
    
    for video_file in video_dir.glob('*.mp4'):
        print(f"\nProcessing: {video_file.name}")
        
        cap = cv2.VideoCapture(str(video_file))
        video_output_dir = output_dir / video_file.stem
        video_output_dir.mkdir(exist_ok=True)
        
        frame_count = 0
        face_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            stats['total_frames'] += 1
            
            # Detect faces
            detection_result = detector.detect(frame)
            
            # Extract and save faces
            for idx, face in enumerate(detection_result.detected_faces):
                face_count += 1
                stats['total_faces'] += 1
                stats['avg_confidence'].append(face.confidence)
                
                # Save with confidence in filename
                filename = (
                    video_output_dir / 
                    f"face_{frame_count:04d}_{idx:02d}_"
                    f"conf_{face.confidence:.3f}.jpg"
                )
                cv2.imwrite(str(filename), face.crop)
        
        cap.release()
        
        stats['videos_processed'] += 1
        print(f"  Frames: {frame_count}, Faces: {face_count}")
    
    # Print summary
    print("\n" + "="*70)
    print("BATCH PROCESSING SUMMARY")
    print("="*70)
    print(f"Videos processed: {stats['videos_processed']}")
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total faces: {stats['total_faces']}")
    if stats['avg_confidence']:
        print(f"Average confidence: {np.mean(stats['avg_confidence']):.4f}")
    print(f"Output directory: {output_directory}")
    print("="*70)
    
    return stats


# ==============================================================================
# Example 2: Adaptive Confidence Thresholding
# ==============================================================================

def detect_with_adaptive_threshold(frame: np.ndarray, detector: FaceDetector):
    """
    Detect faces with adaptive confidence thresholding based on frame brightness.
    
    Use case: Different lighting conditions in a classroom may require
    different confidence thresholds for optimal detection.
    
    Args:
        frame: Input frame
        detector: FaceDetector instance
    
    Returns:
        DetectionResult with adaptively filtered faces
    """
    # Calculate frame brightness
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    
    # Adapt threshold based on brightness
    if brightness < 80:
        # Low light: be more permissive
        adaptive_threshold = 0.4
        lighting = "LOW"
    elif brightness > 200:
        # High light: be more strict
        adaptive_threshold = 0.7
        lighting = "HIGH"
    else:
        # Normal light
        adaptive_threshold = 0.5
        lighting = "NORMAL"
    
    # Temporarily set detector threshold
    original_threshold = detector.confidence_threshold
    detector.confidence_threshold = adaptive_threshold
    
    # Detect with adaptive threshold
    result = detector.detect(frame)
    
    # Restore original threshold
    detector.confidence_threshold = original_threshold
    
    print(f"Lighting: {lighting} (brightness={brightness:.1f}), "
          f"Threshold: {adaptive_threshold:.2f}, "
          f"Faces: {len(result.detected_faces)}")
    
    return result


# ==============================================================================
# Example 3: Face Tracking Across Frames
# ==============================================================================

class FaceTracker:
    """
    Track individual faces across multiple frames.
    
    Useful for:
    - Preventing duplicate recognition of same student
    - Measuring confident detections over time
    - Building temporal coherence in recognition
    """
    
    def __init__(self, max_frames_without_detection: int = 30):
        """
        Initialize tracker.
        
        Args:
            max_frames_without_detection: Frames before removing a track
        """
        self.tracks = {}  # Track ID -> list of faces
        self.next_track_id = 0
        self.max_frames_without_detection = max_frames_without_detection
    
    def _distance(self, face1_bbox, face2_bbox) -> float:
        """Calculate distance between two bounding boxes."""
        x1, y1, w1, h1 = face1_bbox
        x2, y2, w2, h2 = face2_bbox
        
        # Center distance
        c1 = (x1 + w1/2, y1 + h1/2)
        c2 = (x2 + w2/2, y2 + h2/2)
        
        return np.sqrt((c1[0] - c2[0])**2 + (c1[1] - c2[1])**2)
    
    def update(self, detection_result: DetectionResult) -> dict:
        """
        Update tracks with new detections.
        
        Returns:
            Dictionary mapping track IDs to Face objects
        """
        matched_tracks = {}
        unmatched_detections = list(range(len(detection_result.detected_faces)))
        
        # Try to match detected faces to existing tracks
        for track_id, track_faces in self.tracks.items():
            if not track_faces:
                continue
            
            last_face = track_faces[-1]
            best_match_idx = -1
            best_distance = float('inf')
            
            # Find closest detection
            for det_idx in unmatched_detections:
                face = detection_result.detected_faces[det_idx]
                dist = self._distance(last_face.bbox, face.bbox)
                
                if dist < best_distance:
                    best_distance = dist
                    best_match_idx = det_idx
            
            # If close enough, add to track
            if best_distance < 100:  # Threshold in pixels
                matched_tracks[track_id] = detection_result.detected_faces[best_match_idx]
                self.tracks[track_id].append(
                    detection_result.detected_faces[best_match_idx]
                )
                unmatched_detections.remove(best_match_idx)
        
        # Create new tracks for unmatched detections
        for det_idx in unmatched_detections:
            track_id = self.next_track_id
            self.next_track_id += 1
            self.tracks[track_id] = [detection_result.detected_faces[det_idx]]
            matched_tracks[track_id] = detection_result.detected_faces[det_idx]
        
        return matched_tracks
    
    def get_high_confidence_faces(self, min_detections: int = 3) -> dict:
        """
        Get faces that have been consistently detected.
        
        Returns:
            Dictionary of track_id -> Face objects (for recognition)
        """
        high_confidence_faces = {}
        
        for track_id, track_faces in self.tracks.items():
            if len(track_faces) >= min_detections:
                avg_confidence = np.mean([f.confidence for f in track_faces])
                
                # Use face with highest confidence for recognition
                best_face = max(track_faces, key=lambda f: f.confidence)
                best_face.track_id = track_id
                best_face.track_length = len(track_faces)
                best_face.avg_track_confidence = avg_confidence
                
                high_confidence_faces[track_id] = best_face
        
        return high_confidence_faces


def track_faces_in_video(video_path: str):
    """
    Example: Track faces across video frames.
    """
    detector = FaceDetector(confidence_threshold=0.5)
    tracker = FaceTracker()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect faces
        detection_result = detector.detect(frame)
        
        # Update tracker
        tracked_faces = tracker.update(detection_result)
        
        # Get consistently detected faces (ready for recognition)
        high_confidence_faces = tracker.get_high_confidence_faces(min_detections=5)
        
        print(f"Frame {frame_count}: "
              f"New detections: {len(tracked_faces)}, "
              f"High-confidence tracks: {len(high_confidence_faces)}")
        
        # [FUTURE] These high_confidence_faces are ideal for recognition
        # as they represent faces that have been seen consistently
        for track_id, face in high_confidence_faces.items():
            print(f"  Track {track_id}: "
                  f"appearances={face.track_length}, "
                  f"avg_conf={face.avg_track_confidence:.4f}")
    
    cap.release()


# ==============================================================================
# Example 4: Prepare Data for Recognition Module Training
# ==============================================================================

def extract_training_dataset(
    video_path: str,
    student_label: str,
    output_dir: str,
    min_confidence: float = 0.7
):
    """
    Extract high-quality face images from a video for training recognition model.
    
    This is useful for building a training dataset where each student's faces
    are collected and labeled.
    
    Args:
        video_path: Path to video file
        student_label: Student ID or name
        output_dir: Where to save training faces
        min_confidence: Only save faces above this confidence
    """
    detector = FaceDetector(confidence_threshold=0.5)
    tracker = FaceTracker()
    
    dataset_dir = Path(output_dir) / student_label
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_faces = 0
    
    print(f"Extracting training data for: {student_label}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        detection_result = detector.detect(frame)
        tracker.update(detection_result)
        
        # Get high-confidence faces
        high_conf_faces = tracker.get_high_confidence_faces(min_detections=1)
        
        for track_id, face in high_conf_faces.items():
            if face.confidence >= min_confidence:
                filename = (
                    dataset_dir /
                    f"{student_label}_frame_{frame_count:05d}_"
                    f"track_{track_id:03d}_conf_{face.confidence:.3f}.jpg"
                )
                cv2.imwrite(str(filename), face.crop)
                saved_faces += 1
    
    cap.release()
    
    print(f"Extraction complete!")
    print(f"Frames processed: {frame_count}")
    print(f"Faces saved: {saved_faces}")
    print(f"Output directory: {dataset_dir}")
    print(f"\nDataset structure ready for recognition model training:")
    print(f"{dataset_dir}/")
    print(f"├── student_001_frame_00001_track_000_conf_0.923.jpg")
    print(f"├── student_001_frame_00045_track_000_conf_0.891.jpg")
    print(f"└── ... (more faces)")


# ==============================================================================
# Example 5: Real-time Detection with Logging and Analysis
# ==============================================================================

def analyze_detection_performance(video_path: str, num_frames: int = 100):
    """
    Analyze face detection performance metrics.
    
    Returns statistics useful for:
    - Evaluating detector accuracy
    - Identifying problematic frames
    - Estimating system throughput
    """
    detector = FaceDetector(confidence_threshold=0.5)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    stats = {
        'frames_with_faces': 0,
        'frames_without_faces': 0,
        'total_faces_detected': 0,
        'confidence_scores': [],
        'face_sizes': [],
        'processing_times': []
    }
    
    import time
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Measure detection time
        start_time = time.time()
        result = detector.detect(frame)
        elapsed = time.time() - start_time
        
        stats['processing_times'].append(elapsed)
        
        if result.detected_faces:
            stats['frames_with_faces'] += 1
            stats['total_faces_detected'] += len(result.detected_faces)
            
            for face in result.detected_faces:
                stats['confidence_scores'].append(face.confidence)
                
                _, _, w, h = face.bbox
                face_area = w * h
                stats['face_sizes'].append(face_area)
        else:
            stats['frames_without_faces'] += 1
    
    cap.release()
    
    # Print analysis
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    print(f"Frames analyzed: {frame_count}")
    print(f"Frames with faces: {stats['frames_with_faces']} "
          f"({100*stats['frames_with_faces']/frame_count:.1f}%)")
    print(f"Frames without faces: {stats['frames_without_faces']} "
          f"({100*stats['frames_without_faces']/frame_count:.1f}%)")
    print(f"\nDetection statistics:")
    print(f"  Total faces: {stats['total_faces_detected']}")
    print(f"  Avg faces per frame: {stats['total_faces_detected']/frame_count:.2f}")
    
    if stats['confidence_scores']:
        print(f"\nConfidence scores:")
        print(f"  Mean: {np.mean(stats['confidence_scores']):.4f}")
        print(f"  Median: {np.median(stats['confidence_scores']):.4f}")
        print(f"  Min: {np.min(stats['confidence_scores']):.4f}")
        print(f"  Max: {np.max(stats['confidence_scores']):.4f}")
        
        print(f"\nFace sizes (pixels²):")
        print(f"  Mean: {np.mean(stats['face_sizes']):.0f}")
        print(f"  Min: {np.min(stats['face_sizes']):.0f}")
        print(f"  Max: {np.max(stats['face_sizes']):.0f}")
    
    if stats['processing_times']:
        print(f"\nProcessing time:")
        print(f"  Mean: {np.mean(stats['processing_times'])*1000:.2f} ms")
        print(f"  FPS: {1/np.mean(stats['processing_times']):.1f}")
    
    print("="*70)
    
    return stats


# ==============================================================================
# Main: Run Examples
# ==============================================================================

if __name__ == '__main__':
    print("Advanced Face Detection Examples")
    print("="*70)
    print("\nAvailable examples:")
    print("1. batch_process_videos()")
    print("2. detect_with_adaptive_threshold()")
    print("3. FaceTracker() - Track faces across frames")
    print("4. extract_training_dataset() - Build recognition training data")
    print("5. analyze_detection_performance() - Performance metrics")
    print("\nImport this module in your scripts to use these functions")
