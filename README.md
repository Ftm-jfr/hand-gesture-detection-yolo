# Hand Gesture Detection with YOLO11

Real-time detection of six hand gestures (one to five raised fingers and a thumbs-up) with a YOLO11n model trained on a self-recorded dataset. Training labels were generated automatically with MediaPipe Hands, so the whole dataset was built from raw videos without manual annotation.

This project was developed as a course project for **<Deep Learning>** at the **University of Isfahan**.

## Overview

- **Task:** object detection of hand gestures (bounding box + class).
- **Classes:** `one_finger`, `two_fingers`, `three_fingers`, `four_fingers`, `five_fingers`, `like`.
- **Model:** YOLO11n (Ultralytics), fine-tuned from COCO-pretrained weights.
- **Data:** about 4,600 frames extracted from short videos of one hand per gesture (3,683 training and 903 validation images).
- **Labels:** bounding boxes computed from MediaPipe Hands landmarks (with a 30 px margin); the class is taken from the folder of the source video.

## Pipeline

| Step | File | What it does |
|---|---|---|
| 1. Record | n/a | One or more videos per gesture, stored in `videos/<Class>/` with class folders `One`, `Two`, `Three`, `Four`, `Five`, `Like`. |
| 2. Extract frames | `extract-frame-from-video.py` | Saves every 5th frame of each video into `videos/<Class>/frames/`. |
| 3. Auto-label | `labeling.py` | Runs MediaPipe Hands on every frame, converts the hand landmarks to a YOLO-format box, and writes `dataset/images/all` and `dataset/labels/all`. Frames where no hand is detected are skipped. |
| 4. Train | `train.ipynb` | Fine-tunes YOLO11n for 60 epochs (image size 640, batch 32) with rotation, scale, flip, mosaic and mixup augmentation. Written for Kaggle. |
| 5. Run | `run.py` | Live inference on a camera or IP-camera stream with colored boxes by confidence. |

`data.yaml` describes the dataset layout (`dataset/images/{train,val}`) and the class names.

## Results

Validation set (903 images), final epoch:

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| one_finger | 0.477 | 1.000 | 0.930 | 0.830 |
| two_fingers | 0.820 | 0.537 | 0.763 | 0.639 |
| three_fingers | 0.502 | 0.825 | 0.755 | 0.689 |
| four_fingers | 0.708 | 0.813 | 0.706 | 0.629 |
| five_fingers | 0.512 | 0.667 | 0.735 | 0.696 |
| like | 0.680 | 0.958 | 0.950 | 0.805 |
| **all** | 0.617 | 0.800 | **0.807** | **0.715** |

Inference takes about 1.3 ms per image on the Kaggle GPU used for training.

## Dataset and weights

- **Dataset:** self-recorded videos of the author's hand (not included in this repository). To reproduce the dataset, record your own videos and follow the pipeline above.
- **Trained weights (YOLO11n, about 5 MB):** included in this repository as `best.pt`.

The notebook expects the extracted and labeled dataset as a Kaggle dataset named `6class-finger`, laid out as `dataset/images/{train,val}` and `dataset/labels/{train,val}`.

## Setup

```bash
pip install ultralytics opencv-python mediapipe
```

`mediapipe` is only needed for `labeling.py`; `run.py` needs `ultralytics` and `opencv-python`. `labeling.py` uses the `mp.solutions.hands` API, so use a MediaPipe release that still ships it.

## Usage

**Build the dataset from your own videos**

```bash
python extract-frame-from-video.py   # reads videos/<Class>/*.mp4, writes frames/
python labeling.py                   # writes dataset/images/all and dataset/labels/all
```

Then split `dataset/images/all` and `dataset/labels/all` into `train` and `val` folders and train with `train.ipynb` (or the same `model.train(...)` call locally, pointing `data.yaml` to your dataset path).

**Run live detection**

Edit `run.py`:

- Weights: the script loads `best.pt` from the same folder, which is the trained model included in this repository.
- Video source: the default is a phone camera over Wi-Fi (for example the IP Webcam app); replace `<phone-ip>` with your phone's address. To use a laptop webcam (`0`) or a video file instead, comment out that line and uncomment option 2 or 3. Only one source should be active.

```bash
python run.py
```


