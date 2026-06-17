"""
Configuration constants for the Student Attendance System
"""

# ============================================================================
# Face Detection Configuration
# ============================================================================

# Confidence thresholds
DETECTION_CONFIDENCE_MIN = 0.4
DETECTION_CONFIDENCE_DEFAULT = 0.5
DETECTION_CONFIDENCE_HIGH = 0.7

# Face size constraints (in pixels)
MIN_FACE_WIDTH = 20
MIN_FACE_HEIGHT = 20
MAX_FACE_WIDTH = 800
MAX_FACE_HEIGHT = 800

# Detection method
# Options: 'cascade', 'dnn', 'auto' (try DNN first, fallback to cascade)
DETECTION_METHOD = 'auto'

# ============================================================================
# Camera Configuration
# ============================================================================

# Default camera index
DEFAULT_CAMERA_INDEX = 0

# Camera frame size
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
CAMERA_FPS = 30

# Region of Interest (ROI) - set None to use full frame
# Format: (x, y, width, height) - crops detection to this region
CAMERA_ROI = None  # Use full frame

# ============================================================================
# Face Recognition Configuration (for future phase)
# ============================================================================

# Recognition confidence thresholds
RECOGNITION_CONFIDENCE_MIN = 0.6
RECOGNITION_CONFIDENCE_HIGH = 0.8

# Student matching distance threshold (for embedding-based recognition)
EMBEDDING_DISTANCE_THRESHOLD = 0.6

# Maximum number of faces to recognize per frame
MAX_FACES_PER_FRAME = 50

# ============================================================================
# Logging Configuration
# ============================================================================

# Log levels: 'DEBUG', 'INFO', 'WARNING', 'ERROR'
LOG_LEVEL_DETECTION = 'INFO'
LOG_LEVEL_RECOGNITION = 'INFO'

# Log file paths
LOG_FILE_DETECTION = 'logs/detection.log'
LOG_FILE_RECOGNITION = 'logs/recognition.log'
LOG_FILE_ATTENDANCE = 'logs/attendance.log'

# ============================================================================
# Detection Result Storage
# ============================================================================

# Directory to save detected faces
SAVE_FACES_DIR = 'saved_faces'

# Directory to save annotated frames
SAVE_FRAMES_DIR = 'saved_frames'

# Keep saved faces (True) or clean up after processing (False)
KEEP_SAVED_FACES = True

# ============================================================================
# Performance Tuning
# ============================================================================

# Process every Nth frame (set to 1 to process all frames)
FRAME_SKIP = 1

# Enable GPU acceleration if available
USE_GPU = True

# Maximum queue size for face processing
FACE_QUEUE_SIZE = 100

# ============================================================================
# Face Tracking Configuration
# ============================================================================

# Maximum distance (pixels) between detections to consider same face
FACE_TRACKING_DISTANCE_THRESHOLD = 100

# Number of frames without detection before removing a track
FACE_TRACKING_MAX_FRAMES_LOST = 30

# Minimum consecutive detections to mark a face as "confident"
FACE_TRACKING_MIN_DETECTIONS = 3

# ============================================================================
# Video Processing Configuration
# ============================================================================

# Supported video formats
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.flv']

# Frame extraction rate for batch processing
BATCH_PROCESSING_SKIP_FRAMES = 5

# ============================================================================
# Database Configuration (for future attendance marking)
# ============================================================================

# Student database connection string (placeholder)
DB_CONNECTION_STRING = "sqlite:///student_attendance.db"

# Enable automatic database updates
AUTO_UPDATE_DB = True

# ============================================================================
# UI/Visualization Configuration
# ============================================================================

# Bounding box color (BGR format)
BBOX_COLOR = (0, 255, 0)  # Green

# Confidence score display color
CONFIDENCE_COLOR = (0, 255, 0)  # Green

# Text font
TEXT_FONT = 'FONT_HERSHEY_SIMPLEX'
TEXT_FONT_SCALE = 0.5
TEXT_THICKNESS = 1

# ============================================================================
# System Constraints
# ============================================================================

# Maximum concurrent recognition processes
MAX_RECOGNITION_WORKERS = 4

# Memory limit for face crops cache (MB)
FACE_CACHE_SIZE_MB = 512

# ============================================================================
# Alert Configuration
# ============================================================================

# Enable alerts for low confidence detections
ALERT_LOW_CONFIDENCE = False
LOW_CONFIDENCE_THRESHOLD = 0.3

# Enable alerts for multiple faces in frame
ALERT_MULTIPLE_FACES = False
MULTIPLE_FACES_THRESHOLD = 5
