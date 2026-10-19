# Local webcam face recognition

Opt-in demo for this laptop. You enroll yourself or a consenting person, then start the webcam and read the name on screen. Someone who is not enrolled is shown as **Unknown**. Two enrolled people who score too close together are shown as **Uncertain**.

Run every command from the repository root, with the virtual environment activated.

## What you can do

1. Download the YuNet and SFace model files.
2. Enroll from the webcam, or from photos you place in `data/photos`.
3. Run the live demo and read the name, status, and cosine score.
4. Score a separate set of photos with `evaluate.py` when you want measured rates.

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
camera.py                  Open the webcam, read the latest frame, release it
tracking.py                IoU tracks and label smoothing
model_assets.py            Pinned model names, hashes, and download
enroll.py                  Webcam or folder enrollment
demo.py                    Webcam demo
evaluate.py                Offline metrics on a separate photo set
scripts/download_models.py Download and verify the ONNX files
tests/                     Unit tests
data/photos/               Put enrollment photos here
data/gallery/              Saved profiles (created when you enroll)
data/reports/              JSON evaluation reports
models/                    YuNet and SFace weights
eval_data/                 Calibration and final photo sets
```

`quickstart.py` and `examples.py` are an earlier detection-only draft. The demo, enrollment, and evaluation commands above are the working path.

Personal photos, gallery files, reports, and model weights are ignored by git. The empty folders are kept so the layout is visible.

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

`tests/test_models_integration.py` runs the real ONNX files when they are present. The other tests use stand-ins and do not need the webcam.

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

`AttendanceLogger` in `face_detection.py` remains importable for the earlier examples. The demo does not call it. Attendance marking, CCTV, remote cameras, and background monitoring are outside this demo.

## Limits

A photograph or screen showing an enrolled person can be accepted. This is ordinary face matching. It does not check liveness.

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
| `enroll.py` | Consent prompt, webcam capture, and folder enrollment |
| `demo.py` | Draw boxes, status, score, and FPS |
| `evaluate.py` | Per-image decisions, rates, calibration sweep, and JSON report |
| `model_assets.py` | Filenames, hashes, verify, and download |
| `config.py` | Defaults used by the commands above |
