"""
Quick Start Guide - Face Detection Module

The recognition demo is documented in README.md. Run it with: python demo.py

This file contains the most common usage patterns to get started quickly.
Copy and paste these snippets into your own scripts.
"""

# ============================================================================
# QUICK START 1: Live Webcam Face Detection (30 seconds)
# ============================================================================

def quickstart_webcam():
    """Detect and display faces from webcam."""
    import cv2
    from face_detection import FaceDetector
    
    detector = FaceDetector(confidence_threshold=0.5)
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        result = detector.detect(frame)
        
        # Draw boxes
        for face in result.detected_faces:
            x, y, w, h = face.bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw confidence
            text = f"Conf: {face.confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.imshow('Faces', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


# ============================================================================
# QUICK START 2: Detect Faces in an Image
# ============================================================================

def quickstart_image(image_path):
    """Detect faces in a single image."""
    import cv2
    from face_detection import FaceDetector
    
    frame = cv2.imread(image_path)
    detector = FaceDetector(confidence_threshold=0.6)
    
    result = detector.detect(frame)
    
    print(f"Faces detected: {len(result.detected_faces)}")
    for i, face in enumerate(result.detected_faces):
        print(f"\nFace {i+1}:")
        print(f"  Position: {face.bbox}")
        print(f"  Confidence: {face.confidence:.4f}")
        print(f"  Crop size: {face.crop.shape}")


# ============================================================================
# QUICK START 3: Save Face Crops
# ============================================================================

def quickstart_save_crops(video_path, output_dir='faces'):
    """Extract and save all face crops from a video."""
    import cv2
    import os
    from face_detection import FaceDetector
    
    os.makedirs(output_dir, exist_ok=True)
    detector = FaceDetector()
    cap = cv2.VideoCapture(video_path)
    
    frame_idx = 0
    face_idx = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_idx += 1
        result = detector.detect(frame)
        
        for face in result.detected_faces:
            filename = f"{output_dir}/face_{face_idx:04d}_conf_{face.confidence:.2f}.jpg"
            cv2.imwrite(filename, face.crop)
            face_idx += 1
        
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, saved {face_idx} faces")
    
    cap.release()
    print(f"Done! Saved {face_idx} face crops to {output_dir}/")


# ============================================================================
# QUICK START 4: Get Detection Statistics
# ============================================================================

def quickstart_statistics(video_path, num_frames=500):
    """Analyze detection statistics from a video."""
    import cv2
    import numpy as np
    from face_detection import FaceDetector
    
    detector = FaceDetector(confidence_threshold=0.5)
    cap = cv2.VideoCapture(video_path)
    
    frame_count = 0
    total_faces = 0
    confidences = []
    
    while frame_count < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        result = detector.detect(frame)
        
        total_faces += len(result.detected_faces)
        for face in result.detected_faces:
            confidences.append(face.confidence)
    
    cap.release()
    
    print(f"Analysis of {frame_count} frames:")
    print(f"  Total faces: {total_faces}")
    print(f"  Avg faces/frame: {total_faces/frame_count:.2f}")
    if confidences:
        print(f"  Avg confidence: {np.mean(confidences):.4f}")
        print(f"  Min confidence: {np.min(confidences):.4f}")
        print(f"  Max confidence: {np.max(confidences):.4f}")


# ============================================================================
# QUICK START 5: Custom Confidence Per Frame
# ============================================================================

def quickstart_adaptive_confidence(video_path):
    """Detect faces with threshold that adapts to lighting."""
    import cv2
    import numpy as np
    from face_detection import FaceDetector
    
    detector = FaceDetector()
    cap = cv2.VideoCapture(video_path)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Adapt threshold based on brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < 80:
            detector.confidence_threshold = 0.4  # Dark: be permissive
        elif brightness > 200:
            detector.confidence_threshold = 0.7  # Bright: be strict
        else:
            detector.confidence_threshold = 0.5  # Normal
        
        result = detector.detect(frame)
        print(f"Brightness: {brightness:.0f}, Threshold: {detector.confidence_threshold}, "
              f"Faces: {len(result.detected_faces)}")


# ============================================================================
# QUICK START 6: Prepare Data for Recognition (Phase 2)
# ============================================================================

def quickstart_prep_for_recognition(video_path):
    """
    Extract faces ready for recognition module.
    This demonstrates how to prepare data for the next phase.
    """
    import cv2
    from face_detection import FaceDetector, FaceRecognizer
    
    detector = FaceDetector(confidence_threshold=0.7)
    recognizer = FaceRecognizer()  # Placeholder for now
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Step 1: Detect faces
        detection_result = detector.detect(frame)
        
        # Step 2: Get faces ready for recognition
        faces_for_recognition = detection_result.get_faces_for_recognition()
        
        print(f"Frame {frame_count}: {len(faces_for_recognition)} faces ready for recognition")
        
        # Step 3: [FUTURE] Send to recognition module
        # for face in faces_for_recognition:
        #     student_id, name, confidence = recognizer.recognize(face.crop)
        #     face.recognized_student_id = student_id
        #     face.recognized_name = name
        #     face.recognition_confidence = confidence
    
    cap.release()


# ============================================================================
# QUICK START 7: Using Multiple Confidence Thresholds
# ============================================================================

def quickstart_multi_threshold(frame):
    """Compare detection results at different thresholds."""
    import cv2
    from face_detection import FaceDetector
    
    thresholds = [0.3, 0.5, 0.7, 0.9]
    
    for threshold in thresholds:
        detector = FaceDetector(confidence_threshold=threshold)
        result = detector.detect(frame)
        
        print(f"Threshold {threshold}: {len(result.detected_faces)} faces")
        for face in result.detected_faces:
            print(f"  - Confidence: {face.confidence:.4f}")


# ============================================================================
# QUICK START 8: Face Tracking Across Frames
# ============================================================================

def quickstart_tracking(video_path):
    """Track individual faces across video frames."""
    import cv2
    from examples import FaceTracker
    from face_detection import FaceDetector
    
    detector = FaceDetector()
    tracker = FaceTracker(max_frames_without_detection=30)
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while frame_count < 300:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Detect and track
        detection_result = detector.detect(frame)
        tracked_faces = tracker.update(detection_result)
        
        # Get high-confidence tracks (faces seen consistently)
        high_conf_faces = tracker.get_high_confidence_faces(min_detections=5)
        
        if high_conf_faces:
            print(f"Frame {frame_count}: {len(high_conf_faces)} high-confidence tracks")
            for track_id, face in high_conf_faces.items():
                print(f"  Track {track_id}: avg_confidence={face.avg_track_confidence:.4f}")
    
    cap.release()


# ============================================================================
# ADVANCED: Process Multiple Videos in Parallel
# ============================================================================

def quickstart_batch_process(video_list):
    """Process multiple videos and collect statistics."""
    from pathlib import Path
    from face_detection import FaceDetector
    import cv2
    
    detector = FaceDetector()
    overall_stats = {
        'total_frames': 0,
        'total_faces': 0,
        'videos_processed': 0
    }
    
    for video_path in video_list:
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        face_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            result = detector.detect(frame)
            face_count += len(result.detected_faces)
        
        cap.release()
        
        overall_stats['total_frames'] += frame_count
        overall_stats['total_faces'] += face_count
        overall_stats['videos_processed'] += 1
        
        print(f"Video {Path(video_path).name}: {frame_count} frames, {face_count} faces")
    
    print(f"\nOverall: {overall_stats['total_faces']} faces in "
          f"{overall_stats['total_frames']} frames across "
          f"{overall_stats['videos_processed']} videos")


# ============================================================================
# CHEATSHEET: Common Patterns
# ============================================================================

"""
┌─────────────────────────────────────────────────────────────────────┐
│                          QUICK REFERENCE                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. Basic Detection:                                                │
│     detector = FaceDetector(confidence_threshold=0.5)              │
│     result = detector.detect(frame)                                │
│     for face in result.detected_faces:                             │
│         print(face.confidence, face.bbox, face.crop)              │
│                                                                     │
│  2. Adjust Threshold:                                               │
│     detector.confidence_threshold = 0.7                            │
│                                                                     │
│  3. Get Frame Info:                                                 │
│     print(result.frame_index, result.frame_shape)                  │
│                                                                     │
│  4. Extract Crops:                                                  │
│     for face in result.detected_faces:                             │
│         crop = face.crop  # Already extracted!                     │
│                                                                     │
│  5. Access Statistics:                                              │
│     stats = detector.get_stats()                                   │
│                                                                     │
│  6. Prepare for Recognition:                                        │
│     faces = result.get_faces_for_recognition()                    │
│         # Later: populate face.recognition_confidence, etc.       │
│                                                                     │
│  7. Process Multiple Frames:                                        │
│     for frame_idx in range(num_frames):                            │
│         result = detector.detect(frame)                            │
│         # frame_idx is in result.frame_index                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
"""

# ============================================================================
# Run Examples
# ============================================================================

if __name__ == '__main__':
    print("Face Detection - Quick Start Examples")
    print("="*70)
    print("\nAvailable quick starts:")
    print("  1. quickstart_webcam() - Live camera feed")
    print("  2. quickstart_image(path) - Single image")
    print("  3. quickstart_save_crops(path) - Extract face crops")
    print("  4. quickstart_statistics(path) - Get detection stats")
    print("  5. quickstart_adaptive_confidence(path) - Adaptive threshold")
    print("  6. quickstart_prep_for_recognition(path) - Prepare for recognition")
    print("  7. quickstart_tracking(path) - Track faces across frames")
    print("  8. quickstart_batch_process([paths]) - Process multiple videos")
    print("\nImport this module and call any function to get started!")
    print("\nExample:")
    print("  from quickstart import quickstart_webcam")
    print("  quickstart_webcam()")
