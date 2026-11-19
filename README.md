# Face-recognition attendance with a camera

Local attendance for this laptop. Students who agreed to take part are enrolled once. During a class the Mac camera acts as a CCTV camera: every enrolled student it recognizes is checked in with a **timestamp** and a **location**. Someone who is not enrolled is shown as **Unknown**. Two enrolled people who score too close together are shown as **Uncertain**, and neither is checked in.

Run every command from the repository root, with the virtual environment activated.

## What you can do

1. Download the YuNet and SFace model files.
2. Enroll students from the webcam, or from photos you place in `data/photos`.
3. Take attendance with `attend.py run`, then print or export the report with `attend.py report`.
4. Run the plain recognition demo (`demo.py`) to check names and scores without recording anything.
5. Score a separate set of photos with `evaluate.py` when you want measured rates.

Quick start:

```bash
python enroll.py webcam --person-id s001 --name "Ada Lovelace"
python attend.py run --session "CS101 Lecture" --location "Room 101, Science Block" --late-after 10
python attend.py report --session-id 1 --csv data/attendance/cs101.csv
```

Photos and embeddings stay on this computer. `scripts/download_models.py` is the only network step, and it downloads the two OpenCV model files.

## Requirements

- macOS on Apple silicon, or another platform with a wheel for the pinned OpenCV build
- Python 3.14 (the pinned `opencv-contrib-python` 4.14.0.94 wheel used here is the macOS arm64 build)
- A webcam for live enrollment and the demo
- Camera permission for the terminal or IDE: System Settings > Privacy & Security > Camera

Pinned packages in `requirements.txt`:

| Package | Version |
| --- | --- |
| opencv-contrib-python | 4.14.0.94 |
| numpy | 2.5.4 |
| pytest | 9.1.1 |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_models.py
```

`scripts/download_models.py` writes `models/*.onnx` and checks the SHA-256 values below. If a file is missing or the hash differs, download it again with the same command.

## Repository layout

```text
config.py                  Thresholds, camera size, gallery path
face_detection.py          YuNet detector and SFace recognizer
matching.py                Max-cosine decision, threshold, and margin
gallery.py                 Local profiles under data/gallery
quality.py                 Enrollment quality gates
camera.py                  Open the webcam or a stream URL, read the latest frame, release it
tracking.py                IoU tracks and label smoothing
attendance.py              Camera stations, check-ins, sightings, SQLite store, reports
attend.py                  Attendance command: run, sessions, report, delete-session
model_assets.py            Pinned model names, hashes, and download
enroll.py                  Webcam or folder enrollment
demo.py                    Webcam demo
evaluate.py                Offline metrics on a separate photo set
scripts/download_models.py Download and verify the ONNX files
tests/                     Unit tests
data/photos/               Put enrollment photos here
data/gallery/              Saved profiles (created when you enroll)
data/reports/              JSON evaluation reports
data/attendance/           attendance.db and optional check-in snapshots (created on first run)
models/                    YuNet and SFace weights
eval_data/                 Calibration and final photo sets
```

Personal photos, gallery files, attendance records, reports, and model weights are ignored by git. The empty folders are kept so the layout is visible.

## Put a picture in data/photos

Drop `.jpg`, `.jpeg`, `.png`, or `.bmp` files directly in `data/photos`. Each file needs exactly one face. Use several varied photos when you can: ordinary light, a slightly different angle, and a clear view of the face. Nested folders are skipped.

```bash
python enroll.py folder --person-id me --name "Your Name" --images data/photos
```

The command prints a short explanation and continues only after you type `yes`.

## Enroll

Enrollment continues only after you type `yes`. Enroll yourself, or a person who has agreed. Original photos are discarded unless you pass `--keep-images`.

A person id is 1–64 characters and may use letters, numbers, `_`, and `-`. A display name is 1–80 characters.

### Webcam

About 12 varied frames are kept. The allowed range is 10–20. Turn your head slightly and change expression between frames. Blurry, tiny, badly exposed, strongly turned, or near-duplicate frames are skipped.

```bash
python enroll.py webcam --person-id me --name "Your Name"
```

- `q` cancels and saves nothing
- space saves once at least 10 frames have been accepted
- capture stops at 20 frames

### Folder

Each file must contain exactly one face that passes the same quality checks.

```bash
python enroll.py folder --person-id me --name "Your Name" --images data/photos
```

If fewer than 10 images are accepted, the profile is still saved and the command says that 10 or more varied references are recommended. Skipped files are listed with the reason.

### List and delete

```bash
python enroll.py list
python enroll.py delete --person-id me
```

Delete asks you to type the person id. `--yes` skips that prompt.

### Flags

| Flag | Commands | Meaning |
| --- | --- | --- |
| `--person-id` | webcam, folder, delete | Profile id |
| `--name` | webcam, folder | Name drawn on screen |
| `--images` | folder | Folder of photos. Use `data/photos` |
| `--keep-images` | webcam, folder | Also store the accepted crops under the profile |
| `--replace` | webcam, folder | Overwrite an existing profile |
| `--gallery` | all | Profile directory. Default `data/gallery` |
| `--camera` | webcam | Camera index. Default `0` |
| `--width`, `--height` | webcam | Capture size. Default `1280x720` |
| `--count` | webcam | Target frames, from 10 to 20. Default `12` |
| `--yes` | delete | Skip the confirmation prompt |

### Where a profile is stored

```text
data/gallery/<person_id>/profile.json
data/gallery/<person_id>/embeddings.npy
data/gallery/<person_id>/images/    only with --keep-images
```

`profile.json` stores the display name, timestamps, OpenCV version, and the YuNet and SFace hashes used at enrollment. A later model file with a different hash will refuse to match until that profile is enrolled again. `enroll.py list` prints `MODEL MISMATCH` for that profile.

## Demo

```bash
python demo.py
```

The laptop webcam is camera `0` at 1280x720. Press `q` to quit. The window shows boxes, the name or status, the cosine score, and the measured FPS. Recognition runs every 3rd frame (`--process-every`). The capture buffer is one frame, so a slow loop does not build a queue of old frames.

```bash
python demo.py --camera 0 --width 1280 --height 720 \
  --confidence 0.9 --accept-threshold 0.363 --margin 0.08 \
  --process-every 3 --detect-max-side 640 --gallery data/gallery
```

A name appears after the same person is known on that track for 3 consecutive recognition steps. The label clears when the face disappears, the match becomes Unknown or Uncertain, the track is lost, or the matched person id changes. The camera is released and the windows are closed on quit and on errors.

If the camera does not open, allow Camera access, then try again or pass `--camera 1`.

The demo does not save video and does not enroll anyone it sees.

## Attendance

```bash
python attend.py run --session "CS101 Lecture" --location "Room 101, Science Block"
```

The camera window shows each face with its name and attendance state: `checking in...`, then `present 09:02:13` or `late 09:14:40`. The panel shows the session, location, clock, how many enrolled students are checked in, and the latest check-in. Each check-in is also printed in the terminal. Press `q` to end the session. The session is closed and the report is printed.

### When a student is checked in

1. The tracker shows a confirmed name after 3 consecutive matches for the same person on the same face track (`TRACK_SMOOTH_WINDOW`).
2. That confirmed name must then hold for 5 more recognition steps (`--confirm-steps`, `ATTENDANCE_CONFIRM_STEPS`). At the default `--process-every 3` that is roughly half a second of steady recognition.
3. When two faces in the same frame resolve to the same student, neither is marked and both are labeled `duplicate face, not marked`. A phone photo held next to the real student is a typical cause.
4. A student is checked in once per session. After that, their last-seen time and position are updated at most every 10 seconds (`ATTENDANCE_SIGHTING_INTERVAL_SECONDS`), and each update is kept as a sighting.

### Timestamp and location

| Field | Meaning |
| --- | --- |
| `first_seen` | Check-in time, local time with UTC offset, e.g. `2026-11-19T09:02:13+01:00` |
| `last_seen` | Most recent sighting |
| `status` | `present`, `late` (after `--late-after` minutes), or `absent` (enrolled, never seen) |
| `location` | Where the student checked in: the `--location` of that camera |
| `zone` | Where the face was in the camera view, on a 3x3 grid: `top-left` … `bottom-right` |
| `last_location`, `last_zone` | Where the student was most recently seen |
| sightings | Every throttled sighting with time, camera, location, optional GPS, zone, face center (0–1) and score |

The location of a student is the location of the camera that saw them, as with a fixed CCTV camera. A Mac has no GPS, so pass coordinates yourself if you want them stored: `--lat 52.3702 --lon 4.8952`. Zones are in the camera's own view (not mirrored): a student sitting on the camera's left appears in a `-left` zone. Zones in the top row usually mean the back of the room when the camera faces the class from the front.

### Several cameras, or a real CCTV camera

Each camera is a station with its own `--camera-id` and `--location`. A second camera joins an existing session, and both write to the same database:

```bash
# Terminal 1: built-in camera at the front of the room
python attend.py run --session "CS101 Lecture" --camera-id front --location "Room 101"
# Terminal 2: an iPhone (Continuity Camera) or USB camera at the door
python attend.py run --join 1 --camera 1 --camera-id door --location "Room 101 door"
```

`--camera` also accepts an RTSP or HTTP stream URL, or a video file. That is how an IP/CCTV camera is used once you have one, for example `--camera "rtsp://user:pass@192.168.1.20:554/stream1"`. The trail (`report --trail`) then shows which camera saw each student, and when.

### Run flags

| Flag | Meaning |
| --- | --- |
| `--session NAME` | Start a new session |
| `--join ID` | Add this camera to a running session instead |
| `--location` | Where this camera is. Default `Classroom` |
| `--camera-id` | Name of this camera, letters, numbers, `_`, `-`. Default `mac-camera` |
| `--lat`, `--lon` | Optional coordinates stored with each sighting |
| `--late-after MIN` | Check-ins after this many minutes are `late` |
| `--save-snapshots` | Save the face crop at check-in under `data/attendance/snapshots/session_<id>/` as evidence |
| `--confirm-steps` | Stable recognition steps before check-in. Default `5` |
| `--camera` | Webcam index, stream URL, or video file. Default `0` |
| `--db` | Database file. Default `data/attendance/attendance.db` |

`--width`, `--height`, `--confidence`, `--accept-threshold`, `--margin`, `--process-every`, and `--gallery` work as in the demo.

### Reports

```bash
python attend.py sessions
python attend.py report --session-id 1
python attend.py report --session-id 1 --trail --csv data/attendance/cs101.csv
python attend.py delete-session --session-id 1
```

`report` lists present and late students with check-in time, last seen time, and where they were, followed by every enrolled student who was not seen as `absent`. `--trail` adds the timestamped sightings. `--csv` writes the same rows to a spreadsheet-friendly file. `delete-session` removes the session, its sightings, and its snapshots.

Students enrolled from another terminal while a session is running are picked up automatically. The gallery is reloaded when a profile file changes.

## How a match is decided

YuNet finds each face and five landmarks. Frames are detected at a longest side of 640 px (`--detect-max-side`) and landmarks are mapped back to the full frame. SFace aligns the face and returns a 128-d embedding. Embeddings are L2-normalized in this project.

The identity score is the **maximum cosine** against that person's stored references (nearest reference). Cosine similarity is the dot product of two L2-normalized embeddings. It is a score from -1 to 1, shown as `score 0.512`. It is not a percentage and it is not the YuNet detector score.

- Best score below the acceptance threshold: **Unknown**
- Best score passes, and another identity is within the margin: **Uncertain**
- Otherwise: the enrolled display name
- One enrolled person has no second identity, so only the acceptance threshold applies
- An empty gallery shows **Unknown** and no score

`recognized_name` and `recognized_student_id` are set only for a known match. `recognized_student_id` stores the person id. `recognition_confidence` is left empty so it is not read as a probability. `match_score` holds the cosine score.

SFace's published LFW cosine threshold is **0.363**. That number is a starting point from OpenCV's benchmark table. It is not a measured result for this gallery. The identity margin starts at **0.08** and is not a published SFace threshold. Both can need a new calibration when more people are enrolled. Set a measured pair in `config.py` only after you run the calibration split below.

### Enrollment quality gates

A reference is kept when the face is large enough, sharp enough, reasonably exposed, and not turned past these limits. YuNet landmarks are ordered right eye, left eye, nose, right mouth corner, left mouth corner. Roll is the angle of the line between the eyes. The yaw proxy is the nose's horizontal offset from the eye midpoint, divided by the distance between the eyes. A value near 0 is frontal.

| Check | Default |
| --- | --- |
| Minimum face size | 80 px |
| Minimum YuNet score | 0.9 |
| Minimum Laplacian variance | 80 |
| Brightness | 40 to 210 |
| Maximum roll | 20 degrees |
| Maximum yaw proxy | 0.35 |
| Near-duplicate cosine | 0.92 against the previous accepted frame |

## Evaluate

Collect enrollment, calibration, and final sessions separately. Keep neighboring frames from one capture in the same set. The gallery should contain only you and people who agreed to be enrolled.

```text
eval_data/calibration/enrolled/<person_id>/*.jpg
eval_data/calibration/unenrolled/<label>/*.jpg
eval_data/final/enrolled/<person_id>/*.jpg
eval_data/final/unenrolled/<label>/*.jpg
```

Include ordinary light, dimmer light, small head turns, glasses when you wear them, more than one distance, and at least one consenting person who is not enrolled.

```bash
python evaluate.py --split calibration --data-dir eval_data
python evaluate.py --split final --data-dir eval_data
```

Calibration prints a threshold sweep and does not change the saved result. The sweep tries acceptance thresholds `0.25, 0.30, 0.35, 0.363, 0.40, 0.45, 0.50, 0.55, 0.60` and margins `0.02, 0.05, 0.08, 0.12`. Copy a chosen pair into `config.py`, then run `final` on the other capture.

The report includes correct identification, false identification of unenrolled people, enrolled unknown and uncertain rates, confusion between enrolled people, detection failures, median and p95 latency, hardware, gallery size, and sample counts. Rates use images with exactly one detected face. Images with no face or more than one face are detection failures and stay out of those rates. A JSON copy is written to `data/reports/eval_<split>.json`.

A small volunteer run describes that capture. It does not establish production reliability.

## Configuration

Defaults live in `config.py`.

| Setting | Default | Role |
| --- | --- | --- |
| `DETECTION_CONFIDENCE_DEFAULT` | 0.9 | YuNet score cutoff |
| `DETECTION_NMS_THRESHOLD` | 0.3 | YuNet non-maximum suppression |
| `DETECTION_TOP_K` | 5000 | YuNet top-k |
| `DETECT_MAX_SIDE` | 640 | Longest side passed to YuNet |
| `DEFAULT_CAMERA_INDEX` | 0 | Laptop webcam |
| `CAMERA_FRAME_WIDTH` | 1280 | Capture width |
| `CAMERA_FRAME_HEIGHT` | 720 | Capture height |
| `PROCESS_EVERY_N_FRAMES` | 3 | Recognize every Nth frame |
| `MAX_CONSECUTIVE_READ_FAILURES` | 30 | Stop after this many failed reads |
| `ACCEPT_THRESHOLD` | 0.363 | Minimum cosine for a known match |
| `MATCH_MARGIN` | 0.08 | Required gap to the next identity |
| `EMBEDDING_DIM` | 128 | SFace embedding length |
| `ENROLL_TARGET_FRAMES` | 12 | Webcam enrollment target |
| `ENROLL_MIN_FRAMES` | 10 | Minimum recommended references |
| `ENROLL_MAX_FRAMES` | 20 | Webcam enrollment cap |
| `GALLERY_DIR` | `data/gallery` | Profile root |
| `REPORTS_DIR` | `data/reports` | Evaluation JSON |
| `TRACK_IOU_THRESHOLD` | 0.3 | Track match overlap |
| `TRACK_SMOOTH_WINDOW` | 3 | Consecutive known results before a name is shown |
| `TRACK_MAX_MISSED` | 8 | Frames a track can miss before it is dropped |
| `ATTENDANCE_DB` | `data/attendance/attendance.db` | Attendance database |
| `ATTENDANCE_SNAPSHOT_DIR` | `data/attendance/snapshots` | Check-in face crops with `--save-snapshots` |
| `ATTENDANCE_CONFIRM_STEPS` | 5 | Stable confirmed steps before check-in |
| `ATTENDANCE_SIGHTING_INTERVAL_SECONDS` | 10 | Minimum gap between stored sightings per student |
| `DEFAULT_CAMERA_ID` | `mac-camera` | Camera name when `--camera-id` is omitted |
| `DEFAULT_LOCATION` | `Classroom` | Location when `--location` is omitted |

## Models

Detection uses OpenCV YuNet (`face_detection_yunet_2023mar.onnx`) through `cv2.FaceDetectorYN`. Recognition uses OpenCV SFace (`face_recognition_sface_2021dec.onnx`) through `cv2.FaceRecognizerSF`. Both run on CPU. Nothing is trained here.

| File | Role | License | SHA-256 |
| --- | --- | --- | --- |
| `face_detection_yunet_2023mar.onnx` | Detection and five landmarks | [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE) (Shiqi Yu, OpenCV Zoo) | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| `face_recognition_sface_2021dec.onnx` | 128-d face embedding | [Apache-2.0](https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE) (OpenCV Zoo) | `0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79` |

OpenCV itself is [Apache-2.0](https://github.com/opencv/opencv/blob/4.x/LICENSE). Official walkthrough: [DNN face detection and recognition](https://docs.opencv.org/4.13.0/d0/dd4/tutorial_dnn_face.html).

YuNet's published operating range is roughly 10–300 px faces in the network input.

Celebrity benchmark photos are not downloaded, and those names are not used as enrolled people.

## Tests

```bash
source .venv/bin/activate
python -m pytest
```

`tests/test_models_integration.py` runs the real ONNX files when they are present. The other tests use stand-ins and do not need the webcam. `tests/test_attendance.py` covers check-in rules, late status, duplicate faces, sighting throttling, reports, CSV export, and the gallery cache. 70 tests pass as of 19 November 2026.

## Checked on 19 October 2026

- `python -m pytest`: 53 passed.
- `python scripts/download_models.py` saved both ONNX files and their hashes matched.
- A synthetic aligned crop embedded with the real SFace model, stored, and matched against itself returned `known` with cosine 1.0. That check uses a blank image, so it does not measure recognition of a person.
- `enroll.py list` on an empty gallery prints `No enrolled profiles.`
- Declining the consent prompt exits without writing a profile.

Opening camera 0 from that session returned OpenCV's "not authorized to capture video" error, so the preview never started. Live enrollment and the on-screen demo still need Camera permission, then this checklist. No consenting evaluation photos were collected, so correct-identification and false-identification rates are unmeasured. Do not treat 0.363 or 0.08 as a result for this gallery.

```bash
source .venv/bin/activate
python enroll.py webcam --person-id me --name "Your Name"
python demo.py
```

1. Allow Camera access for the terminal, enroll yourself, and confirm at least 10 accepted frames.
2. Start the demo and confirm your display name and a cosine score.
3. Have an unenrolled person stand in frame and confirm the label is Unknown.
4. Enroll a second consenting person. When the two scores are close, confirm Uncertain.
5. Step out of frame, press `q`, and confirm the camera indicator turns off.

Still open: enroll real people, capture separate calibration and final sessions, and set thresholds from the calibration sweep before trusting the final report. Revisit the margin and acceptance threshold when the gallery grows.

## Limits

A photograph or screen showing an enrolled person can be accepted, and would mark that student present. This is ordinary face matching. It does not check liveness. The duplicate-face rule only catches a photo shown next to the real student. Use `--save-snapshots` if check-ins need to be reviewed.

Attendance records are personal data. Enroll only students who agreed, tell them what is recorded (time, camera location, position in view, and optionally a face crop), and delete sessions you no longer need.

Detector confidence is the YuNet face score. The number drawn next to a name is cosine similarity.

Thresholds will drift as the gallery grows and should be recalibrated on a fresh capture. No accuracy figure is claimed until you measure one with `evaluate.py`.

## Modules

| Module | Responsibility |
| --- | --- |
| `face_detection.py` | `Face`, `DetectionResult`, `FaceDetector`, `FaceRecognizer` |
| `matching.py` | `decide_identity`, L2 normalization, max cosine |
| `gallery.py` | Save, list, replace, and delete profiles. Refuse a model-hash mismatch |
| `quality.py` | Size, blur, brightness, pose, and near-duplicate checks |
| `camera.py` | Open, latest frame, and release, including after an error |
| `tracking.py` | Match boxes by IoU and hold a name only while the identity stays stable |
| `attendance.py` | `Station`, zones, `AttendanceStore` (SQLite), `AttendanceMarker`, report and CSV |
| `attend.py` | Live attendance session on top of the demo loop, plus report commands |
| `enroll.py` | Consent prompt, webcam capture, and folder enrollment |
| `demo.py` | Draw boxes, status, score, and FPS |
| `evaluate.py` | Per-image decisions, rates, calibration sweep, and JSON report |
| `model_assets.py` | Filenames, hashes, verify, and download |
| `config.py` | Defaults used by the commands above |
