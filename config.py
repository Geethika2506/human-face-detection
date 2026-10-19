"""Settings for the local webcam face-recognition demo."""

# YuNet detector. The published OpenCV sample starts at 0.9.
DETECTION_CONFIDENCE_DEFAULT = 0.9
DETECTION_NMS_THRESHOLD = 0.3
DETECTION_TOP_K = 5000
# Longest side of the image passed to YuNet. Boxes are mapped back to the full frame.
DETECT_MAX_SIDE = 640

# Laptop webcam.
DEFAULT_CAMERA_INDEX = 0
CAMERA_FRAME_WIDTH = 1280
CAMERA_FRAME_HEIGHT = 720
# Run detection and recognition on every Nth grabbed frame.
PROCESS_EVERY_N_FRAMES = 3
MAX_CONSECUTIVE_READ_FAILURES = 30

# SFace cosine similarity. 0.363 is the published LFW starting point, not a measured
# threshold for this gallery. The margin is a local starting point.
ACCEPT_THRESHOLD = 0.363
MATCH_MARGIN = 0.08
EMBEDDING_DIM = 128
AGGREGATION_METHOD = "max_cosine"

# Enrollment quality gates.
MIN_FACE_SIZE = 80
MIN_DETECTOR_SCORE = 0.9
MIN_LAPLACIAN_VARIANCE = 80.0
MIN_BRIGHTNESS = 40.0
MAX_BRIGHTNESS = 210.0
MAX_ROLL_DEGREES = 20.0
MAX_YAW_PROXY = 0.35
# Skip a frame when its embedding is this close to the previous accepted frame.
MAX_REFERENCE_COSINE = 0.92
ENROLL_TARGET_FRAMES = 12
ENROLL_MIN_FRAMES = 10
ENROLL_MAX_FRAMES = 20

GALLERY_DIR = "data/gallery"
MODELS_DIR = "models"
REPORTS_DIR = "data/reports"

# Short identity smoothing. A name is shown only after this many consecutive
# known results for the same person on the same track.
TRACK_IOU_THRESHOLD = 0.3
TRACK_SMOOTH_WINDOW = 3
TRACK_MAX_MISSED = 8
