"""
# SENTRY RIG BUILDER
# Modular robot builder - configure, save, load, control

Dependencies:
    pip install opencv-python-headless pillow --break-system-packages
    # Pi 5:
    pip install lgpio --break-system-packages
    # Pi 1-4:
    pip install RPi.GPIO --break-system-packages

Run:
    python3 sentry_rig_builder.py
"""

import tkinter as tk
from tkinter import font as tkfont, messagebox, filedialog, ttk, simpledialog
import json, os, sys, math, threading, time, copy
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

try:
    import scipy.sparse as sp
    HAS_SCIPY = True
except ImportError:
    sp = None
    HAS_SCIPY = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    pd = None
    HAS_PANDAS = False
import urllib.request, urllib.error

# Suppress ALSA/JACK error spam on Pi - must happen before any audio import
import os as _os
_devnull = _os.open(_os.devnull, _os.O_WRONLY)
_saved_stderr = _os.dup(2)
_os.dup2(_devnull, 2)
try:
    import ctypes
    _asound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _asound.snd_lib_error_set_handler(ctypes.CFUNCTYPE(None)(lambda *a: None))
except Exception:
    pass
_os.dup2(_saved_stderr, 2)  # restore stderr for real errors
_os.close(_devnull)

# ── PIL ───────────────────────────────────────────────────────────────────────
PIL_IMPORT_ERROR = None
try:
    from PIL import Image, ImageTk
    HAS_PIL = True
except Exception as e:
    HAS_PIL = False
    PIL_IMPORT_ERROR = str(e)
    print(f"[PIL] import failed: {e}")

# ── pyserial (for Flock Detector / oui-spy USB-CDC devices) ───────────────────
try:
    import serial as _pyserial
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

# ── OpenCV ────────────────────────────────────────────────────────────────────
# Deferred - cv2 can segfault on Pi 5 if loaded at module level
cv2               = None
FACE_CASCADE      = None
HAS_CV2           = False
HAS_FACE_CASCADE  = False
CV2_BROKEN        = False   # True if cv2 imports but is missing core modules
COMPACT           = False   # set True at startup for small (e.g. 5") screens

def _try_load_cv2():
    """Imports cv2 and (separately) tries to load the face cascade.
    A missing/broken cascade file must NOT disable the whole camera —
    it only means face detection/tracking/recognition are unavailable;
    the raw video feed and object detection etc. work fine without it."""
    global cv2, FACE_CASCADE, HAS_CV2, HAS_FACE_CASCADE
    if HAS_CV2:
        return True
    try:
        import cv2 as _cv2
        cv2 = _cv2
        HAS_CV2 = True
    except Exception as e:
        print(f"[CAM] cv2 unavailable: {e}")
        HAS_CV2 = False
        return False

    # Try several known cascade locations — pip's opencv-python ships it
    # inside the package; some Debian/apt opencv builds put it elsewhere
    # or omit it if the install was partial.
    if not hasattr(cv2, "CascadeClassifier"):
        print("[CAM] cv2 is missing CascadeClassifier entirely — broken/"
              "conflicting opencv install (see below for the fix)")
    else:
        candidates = []
        try:
            candidates.append(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except Exception:
            pass
        candidates += [
            "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
            "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
        ]
        for path in candidates:
            if path and os.path.isfile(path):
                cascade = cv2.CascadeClassifier(path)
                if not cascade.empty():
                    FACE_CASCADE = cascade
                    HAS_FACE_CASCADE = True
                    break

    # Nothing found locally (this happens with some opencv-python-headless
    # builds, especially newer OpenCV 5.x wheels, that omit the data files)
    # — download the cascade directly from OpenCV's own repo as a last
    # resort. It's a small (~900KB), public, BSD-licensed XML file.
    #
    # NOTE: if cv2.CascadeClassifier itself doesn't exist, this isn't a
    # missing-file problem at all — it means the cv2 install is broken or
    # incomplete (the objdetect module didn't load), most commonly caused
    # by having BOTH opencv-python and opencv-python-headless installed
    # at once, which corrupts each other's shared files in site-packages.
    if not hasattr(cv2, "CascadeClassifier"):
        global CV2_BROKEN
        CV2_BROKEN = True
        print("[CAM] cv2 is missing CascadeClassifier entirely — this is a "
              "BROKEN/CONFLICTING install, not a missing file or network "
              "issue. Almost always caused by having both opencv-python "
              "and opencv-python-headless installed at once. Fix:\n"
              "  pip uninstall opencv-python opencv-python-headless "
              "opencv-contrib-python opencv-contrib-python-headless -y "
              "--break-system-packages\n"
              "  pip install opencv-python-headless --break-system-packages\n"
              "  python3 -c \"import cv2; print(cv2.__file__)\"  "
              "# sanity check it loads from one clean location")
        return True

    if not HAS_FACE_CASCADE:
        cache_path = os.path.join(
            os.path.expanduser("~"), ".sentry_rig_builder",
            "haarcascade_frontalface_default.xml")
        try:
            if not os.path.isfile(cache_path):
                os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                print("[CAM] cascade not found locally — downloading from "
                      "opencv/opencv (GitHub) ...")
                url = ("https://raw.githubusercontent.com/opencv/opencv/"
                       "4.x/data/haarcascades/haarcascade_frontalface_default.xml")
                urllib.request.urlretrieve(url, cache_path)
            cascade = cv2.CascadeClassifier(cache_path)
            if not cascade.empty():
                FACE_CASCADE = cascade
                HAS_FACE_CASCADE = True
                print(f"[CAM] cascade downloaded and loaded from {cache_path}")
        except Exception as e:
            print(f"[CAM] cascade auto-download failed: {e}")

    if not HAS_FACE_CASCADE:
        print("[CAM] face cascade file not found in any known location and "
              "could not be downloaded (check internet connection) — "
              "face detection/tracking/recognition disabled, but the camera "
              "feed itself will still work.")
    return True

# ── Local vision toolkit (object detection / classification / face   ──────────
#    recognition / flag recognition / pose/skeleton tracking) — all offline,
#    no LLM involved.
YOLO_DETECT_MODEL   = None   # ultralytics YOLO, general object detection
YOLO_CLASSIFY_MODEL = None   # ultralytics YOLO, whole-frame classification
YOLO_POSE_MODEL      = None  # ultralytics YOLO, human pose/skeleton keypoints
HAS_YOLO             = False
FACE_RECOGNIZER      = None  # cv2.face.LBPHFaceRecognizer
HAS_FACE_REC         = False
FACE_LABELS          = {}    # int label -> person name

# COCO 17-keypoint skeleton connections (0-indexed), the standard layout
# ultralytics YOLO-pose outputs: nose, eyes, ears, shoulders, elbows,
# wrists, hips, knees, ankles.
POSE_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]

def _draw_bracket_box(frame, x1, y1, x2, y2, color, label=None, thickness=2):
    """Draws a corner-bracket targeting box (sci-fi HUD style) instead of a
    plain rectangle — used consistently across face/object/pose overlays."""
    w, h = x2 - x1, y2 - y1
    clen = max(8, int(min(w, h) * 0.22))
    # top-left
    cv2.line(frame, (x1, y1), (x1 + clen, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + clen), color, thickness)
    # top-right
    cv2.line(frame, (x2, y1), (x2 - clen, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + clen), color, thickness)
    # bottom-left
    cv2.line(frame, (x1, y2), (x1 + clen, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - clen), color, thickness)
    # bottom-right
    cv2.line(frame, (x2, y2), (x2 - clen, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - clen), color, thickness)
    if label:
        cv2.putText(frame, label, (x1, max(12, y1 - 6)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

def _try_load_yolo(need_classify=False, need_pose=False):
    """Lazy-load ultralytics YOLOv8n (detection) / YOLOv8n-cls (classification)
    / YOLOv8n-pose (skeleton keypoints).
    Requires: pip install ultralytics --break-system-packages
    (pulls in torch — heavy; fine on Pi 4/5, painful on Pi Zero/3)."""
    global YOLO_DETECT_MODEL, YOLO_CLASSIFY_MODEL, YOLO_POSE_MODEL, HAS_YOLO
    try:
        from ultralytics import YOLO
        if YOLO_DETECT_MODEL is None:
            YOLO_DETECT_MODEL = YOLO("yolov8n.pt")
        if need_classify and YOLO_CLASSIFY_MODEL is None:
            YOLO_CLASSIFY_MODEL = YOLO("yolov8n-cls.pt")
        if need_pose and YOLO_POSE_MODEL is None:
            YOLO_POSE_MODEL = YOLO("yolov8n-pose.pt")
        HAS_YOLO = True
        return True
    except Exception as e:
        print(f"[VISION] ultralytics unavailable: {e}  "
              f"(pip install ultralytics --break-system-packages)")
        HAS_YOLO = False
        return False

def _try_load_face_recognizer():
    """LBPH face recognizer (needs opencv-contrib-python, not plain opencv)."""
    global FACE_RECOGNIZER, HAS_FACE_REC
    if not HAS_CV2:
        return False
    try:
        FACE_RECOGNIZER = cv2.face.LBPHFaceRecognizer_create()
        HAS_FACE_REC = True
        return True
    except AttributeError:
        print("[VISION] cv2.face unavailable — need opencv-contrib-python, "
              "not opencv-python-headless: "
              "pip install opencv-contrib-python --break-system-packages")
        HAS_FACE_REC = False
        return False

def _train_face_recognizer(known_faces_dir):
    """Scans known_faces_dir/<person_name>/*.jpg and trains the LBPH model."""
    global FACE_LABELS
    if not HAS_FACE_REC or not HAS_CV2 or not HAS_NUMPY:
        return False
    if not os.path.isdir(known_faces_dir):
        return False
    faces, labels, label_map = [], [], {}
    next_label = 0
    for person in sorted(os.listdir(known_faces_dir)):
        person_dir = os.path.join(known_faces_dir, person)
        if not os.path.isdir(person_dir):
            continue
        label_map[next_label] = person
        for fname in os.listdir(person_dir):
            path = os.path.join(person_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (200, 200))
            faces.append(img)
            labels.append(next_label)
        next_label += 1
    if not faces:
        return False
    FACE_RECOGNIZER.train(faces, np.array(labels))
    FACE_LABELS = label_map
    print(f"[VISION] trained face recognizer on {len(faces)} images, "
          f"{len(label_map)} known people")
    return True

REF_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif")

_ref_images = {}   # name -> {"hist": color histogram, "kp": keypoints, "des": ORB descriptors}

def _load_reference_images(images_dir):
    """Loads user-supplied reference images (any object — apple, tree, car,
    computer, a planet, whatever filename you give it) from images_dir and
    precomputes both a color histogram AND ORB keypoint descriptors for each.
    This is a classical-CV matcher (color + local features), not a trained
    deep classifier — it works well for a modest set of visually distinct
    reference objects, but won't generalize the way a trained model would
    (e.g. it matches against the exact images you provided, not the general
    concept of 'apple')."""
    global _ref_images
    _ref_images = {}
    if not HAS_CV2 or not os.path.isdir(images_dir):
        return
    orb = cv2.ORB_create(nfeatures=500)
    for fname in os.listdir(images_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in REF_IMAGE_EXTS:
            continue
        img = cv2.imread(os.path.join(images_dir, fname))
        if img is None:
            continue
        hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, des = orb.detectAndCompute(gray, None)
        h, w = gray.shape[:2]
        _ref_images[name] = {"hist": hist, "kp": kp, "des": des, "w": w, "h": h}
    if _ref_images:
        print(f"[VISION] loaded {len(_ref_images)} reference images: "
              f"{', '.join(_ref_images)}")

def _match_reference_image(frame, hist_threshold=0.5, min_good_matches=12):
    """Returns (name, score, corners) for the best-matching reference image,
    or None. corners is a 4x2 array of (x,y) points in frame space showing
    WHERE the object is — computed via homography from the matched ORB
    keypoints, the same classical technique used for planar object tracking
    (e.g. book-cover / logo detection). Combines a color-histogram check
    (fast pre-filter) with ORB feature matching (shape/texture-aware) for a
    meaningfully better generic-object match than color alone."""
    if not _ref_images:
        return None
    hsv  = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    orb  = cv2.ORB_create(nfeatures=500)
    kp, des = orb.detectAndCompute(gray, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_name, best_score, best_matches = None, 0.0, 0
    best_ref, best_good = None, None
    for name, ref in _ref_images.items():
        hist_score = cv2.compareHist(hist, ref["hist"], cv2.HISTCMP_CORREL)
        if hist_score < hist_threshold:
            continue  # fails the cheap pre-filter, skip the expensive part
        if des is None or ref["des"] is None:
            continue
        matches = bf.match(des, ref["des"])
        good = [m for m in matches if m.distance < 60]
        if len(good) > best_matches:
            best_matches = len(good)
            best_name    = name
            best_score   = min(1.0, len(good) / 40)  # rough confidence 0-1
            best_ref     = ref
            best_good    = good
    if not (best_name and best_matches >= min_good_matches):
        return None

    # Try to locate WHERE the object is via homography (needs >= 4 points).
    corners = None
    if HAS_NUMPY and len(best_good) >= 4:
        try:
            src_pts = np.float32(
                [best_ref["kp"][m.trainIdx].pt for m in best_good]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.queryIdx].pt for m in best_good]).reshape(-1, 1, 2)
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            if H is not None:
                rw, rh = best_ref["w"], best_ref["h"]
                ref_corners = np.float32(
                    [[0, 0], [rw, 0], [rw, rh], [0, rh]]).reshape(-1, 1, 2)
                projected = cv2.perspectiveTransform(ref_corners, H)
                corners = projected.reshape(-1, 2)
        except Exception:
            corners = None  # homography failed — still report the match, just no box

    return best_name, best_score, corners

# ═════════════════════════════════════════════════════════════════════════════
#  AUDIO AI TOOLKIT (mirrors the camera's Local Vision Toolkit) — animal
#  sound / alarm recognition, custom audio matching, speech-to-text,
#  translation. All optional, all lazy-loaded, all degrade gracefully.
# ═════════════════════════════════════════════════════════════════════════════
YAMNET_INTERPRETER = None
YAMNET_LABELS       = []   # index -> AudioSet class name
HAS_YAMNET          = False

# AudioSet class names (substring match, case-insensitive) that count as
# "animal" vs "alarm" for the two dedicated toggles — YAMNet itself just
# gives you 521 raw classes, this is our own curation on top of it.
ANIMAL_SOUND_KEYWORDS = (
    "dog", "bark", "cat", "meow", "bird", "chirp", "cow", "moo", "horse",
    "neigh", "pig", "oink", "sheep", "bleat", "goat", "rooster", "chicken",
    "cluck", "duck", "quack", "frog", "insect", "cricket", "owl", "lion",
    "roar", "elephant", "wolf", "howl", "growl", "purr", "livestock", "fowl",
)
ALARM_SOUND_KEYWORDS = (
    "alarm", "siren", "smoke detector", "fire alarm", "civil defense",
    "buzzer", "doorbell", "emergency vehicle", "ambulance", "police car",
    "fire engine", "foghorn", "air horn", "car alarm",
)

def _try_load_yamnet():
    """Lazy-loads Google's YAMNet (521-class audio event classifier) via
    tflite-runtime. Auto-downloads the small (~4MB) quantized model + label
    map from Google's public TF-Hub-hosted files on first use, cached under
    ~/.sentry_rig_builder/. Requires: pip install tflite-runtime numpy
    --break-system-packages (falls back to full tensorflow's tflite if
    tflite-runtime isn't available — either works)."""
    global YAMNET_INTERPRETER, YAMNET_LABELS, HAS_YAMNET
    if HAS_YAMNET:
        return True
    try:
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            from tensorflow.lite.python.interpreter import Interpreter
    except Exception as e:
        print(f"[AUDIO] tflite unavailable: {e}  "
              f"(pip install tflite-runtime --break-system-packages)")
        return False

    cache_dir = os.path.join(os.path.expanduser("~"), ".sentry_rig_builder")
    model_path = os.path.join(cache_dir, "yamnet.tflite")
    labels_path = os.path.join(cache_dir, "yamnet_labels.csv")
    try:
        os.makedirs(cache_dir, exist_ok=True)
        if not os.path.isfile(model_path):
            print("[AUDIO] downloading YAMNet model (~4MB, one-time) ...")
            urllib.request.urlretrieve(
                "https://tfhub.dev/google/lite-model/yamnet/classification/tflite/1?lite-format=tflite",
                model_path)
        if not os.path.isfile(labels_path):
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/tensorflow/models/master/"
                "research/audioset/yamnet/yamnet_class_map.csv", labels_path)
        with open(labels_path) as f:
            import csv as _csv
            YAMNET_LABELS = [row[2] for row in _csv.reader(f)][1:]  # skip header
        YAMNET_INTERPRETER = Interpreter(model_path=model_path)
        YAMNET_INTERPRETER.allocate_tensors()
        HAS_YAMNET = True
        print(f"[AUDIO] YAMNet loaded ({len(YAMNET_LABELS)} classes)")
        return True
    except Exception as e:
        print(f"[AUDIO] YAMNet setup failed: {e}")
        return False

def _classify_audio_chunk(samples_f32):
    """Runs one ~1s float32 mono 16kHz audio chunk through YAMNet.
    Returns a sorted list of (class_name, score) for the top few classes,
    or [] if unavailable."""
    if not HAS_YAMNET or not HAS_NUMPY:
        return []
    try:
        interp = YAMNET_INTERPRETER
        input_details = interp.get_input_details()
        interp.resize_tensor_input(input_details[0]['index'], [len(samples_f32)])
        interp.allocate_tensors()
        interp.set_tensor(input_details[0]['index'], samples_f32)
        interp.invoke()
        scores = interp.get_tensor(interp.get_output_details()[0]['index'])
        mean_scores = scores.mean(axis=0)
        top = np.argsort(mean_scores)[::-1][:5]
        return [(YAMNET_LABELS[i], float(mean_scores[i])) for i in top
                if i < len(YAMNET_LABELS)]
    except Exception as e:
        print(f"[AUDIO] classify error: {e}")
        return []

def _match_audio_event(top_classes, keywords, min_score=0.15):
    """Checks whether any of YAMNet's top predicted classes for this chunk
    matches one of our keyword lists (animal / alarm)."""
    for name, score in top_classes:
        if score < min_score:
            continue
        low = name.lower()
        if any(kw in low for kw in keywords):
            return name, score
    return None

# ── Custom audio recognition (your own reference clips) ────────────────────
REF_AUDIO_EXTS = (".wav",)   # arecord/wave module only reliably reads WAV
_ref_sounds = {}   # name -> mean MFCC-ish spectral fingerprint (numpy array)

def _audio_fingerprint(samples_f32, sr=16000, n_bands=32):
    """A lightweight, dependency-free spectral fingerprint: FFT magnitude
    binned into n_bands log-spaced frequency bands, averaged over the clip.
    Not as accurate as MFCCs (needs librosa) or a trained model, but works
    entirely with numpy — no extra install required beyond what YAMNet
    already needs. Good enough to tell apart distinct custom sounds
    (a specific siren vs a specific dog bark vs a doorbell), not fine
    enough to distinguish very similar sounds."""
    if not HAS_NUMPY or len(samples_f32) < 512:
        return None
    n = min(len(samples_f32), sr * 3)  # cap at 3s
    windowed = samples_f32[:n] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(windowed))
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    edges = np.logspace(np.log10(50), np.log10(sr / 2), n_bands + 1)
    bands = np.zeros(n_bands)
    for i in range(n_bands):
        mask = (freqs >= edges[i]) & (freqs < edges[i + 1])
        bands[i] = spectrum[mask].mean() if mask.any() else 0.0
    bands = bands / (bands.max() + 1e-9)
    return bands

def _load_reference_sounds(sounds_dir):
    """Loads user-supplied reference WAV clips from sounds_dir (filename =
    label, e.g. my_doorbell.wav) and precomputes their fingerprints."""
    global _ref_sounds
    _ref_sounds = {}
    if not HAS_NUMPY or not os.path.isdir(sounds_dir):
        return
    import wave
    for fname in os.listdir(sounds_dir):
        name, ext = os.path.splitext(fname)
        if ext.lower() not in REF_AUDIO_EXTS:
            continue
        try:
            with wave.open(os.path.join(sounds_dir, fname), "rb") as wf:
                sr = wf.getframerate()
                raw = wf.readframes(wf.getnframes())
                samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                fp = _audio_fingerprint(samples, sr=sr)
                if fp is not None:
                    _ref_sounds[name] = fp
        except Exception as e:
            print(f"[AUDIO] could not load reference sound '{fname}': {e}")
    if _ref_sounds:
        print(f"[AUDIO] loaded {len(_ref_sounds)} reference sounds: "
              f"{', '.join(_ref_sounds)}")

def _match_reference_sound(samples_f32, sr=16000, threshold=0.75):
    """Returns (name, score) for the best-matching reference sound, or None."""
    if not _ref_sounds:
        return None
    fp = _audio_fingerprint(samples_f32, sr=sr)
    if fp is None:
        return None
    best_name, best_score = None, 0.0
    for name, ref_fp in _ref_sounds.items():
        # Cosine similarity between fingerprints
        denom = (np.linalg.norm(fp) * np.linalg.norm(ref_fp)) + 1e-9
        score = float(np.dot(fp, ref_fp) / denom)
        if score > best_score:
            best_name, best_score = name, score
    if best_name and best_score >= threshold:
        return best_name, best_score
    return None

# ── Speech-to-text (Vosk, offline) ──────────────────────────────────────────
VOSK_MODEL      = None
HAS_VOSK        = False
VOSK_MODEL_URL  = ("https://alphacephei.com/vosk/models/"
                    "vosk-model-small-en-us-0.15.zip")

def _try_load_vosk():
    """Lazy-loads Vosk (offline speech-to-text). Auto-downloads the small
    (~40MB) English model on first use, cached under ~/.sentry_rig_builder/.
    Requires: pip install vosk --break-system-packages"""
    global VOSK_MODEL, HAS_VOSK
    if HAS_VOSK:
        return True
    try:
        import vosk
        vosk.SetLogLevel(-1)
    except Exception as e:
        print(f"[AUDIO] vosk unavailable: {e}  "
              f"(pip install vosk --break-system-packages)")
        return False

    cache_dir  = os.path.join(os.path.expanduser("~"), ".sentry_rig_builder")
    model_dir  = os.path.join(cache_dir, "vosk-model-small-en-us-0.15")
    zip_path   = os.path.join(cache_dir, "vosk-model.zip")
    try:
        if not os.path.isdir(model_dir):
            os.makedirs(cache_dir, exist_ok=True)
            print("[AUDIO] downloading Vosk speech model (~40MB, one-time) ...")
            urllib.request.urlretrieve(VOSK_MODEL_URL, zip_path)
            import zipfile
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(cache_dir)
            os.remove(zip_path)
        VOSK_MODEL = vosk.Model(model_dir)
        HAS_VOSK = True
        print("[AUDIO] Vosk speech-to-text model loaded")
        return True
    except Exception as e:
        print(f"[AUDIO] Vosk setup failed: {e}")
        return False

def _speech_to_text(samples_i16, sr=16000):
    """Runs one audio chunk (int16 PCM) through Vosk. Returns recognized
    text, or '' if nothing was said / recognized."""
    if not HAS_VOSK:
        return ""
    try:
        import vosk
        rec = vosk.KaldiRecognizer(VOSK_MODEL, sr)
        rec.AcceptWaveform(samples_i16.tobytes())
        result = json.loads(rec.FinalResult())
        return result.get("text", "").strip()
    except Exception as e:
        print(f"[AUDIO] speech-to-text error: {e}")
        return ""

def _translate_text(text, target_lang):
    """Translates text via the same local Ollama instance already used for
    the AI vision/chat panel — no separate translation API/dependency
    needed. Requires Ollama running with a text-capable model pulled."""
    if not text.strip():
        return ""
    models = ollama_list_models()
    if not models:
        return "ERROR: Ollama not reachable for translation"
    model = models[0]
    reply = ollama_chat(model, [
        {"role": "system", "content":
         f"Translate the user's message to {target_lang}. "
         f"Reply with ONLY the translation, nothing else."},
        {"role": "user", "content": text},
    ], timeout=20)
    return reply

# ── GPIO ──────────────────────────────────────────────────────────────────────
GPIO      = None
ON_PI     = False
_lgpio_h  = None   # lgpio chip handle

try:
    import lgpio as _lgpio

    class _LgpioPWM:
        """RPi.GPIO.PWM-compatible wrapper around lgpio's tx_pwm()."""
        def __init__(self, outer, pin, freq):
            self._outer = outer
            self._pin   = pin
            self._freq  = freq
            self._duty  = 0

        def start(self, duty_cycle):
            self._duty = duty_cycle
            _lgpio.tx_pwm(self._outer._get_chip(), self._pin,
                          self._freq, self._duty)

        def ChangeDutyCycle(self, duty_cycle):
            self._duty = duty_cycle
            _lgpio.tx_pwm(self._outer._get_chip(), self._pin,
                          self._freq, self._duty)

        def ChangeFrequency(self, freq):
            self._freq = freq
            _lgpio.tx_pwm(self._outer._get_chip(), self._pin,
                          self._freq, self._duty)

        def stop(self):
            _lgpio.tx_pwm(self._outer._get_chip(), self._pin, self._freq, 0)

    class _GPIOCompat:
        """Thin RPi.GPIO-compatible wrapper around lgpio (Pi 5 native)."""
        BCM  = OUT = IN = HIGH = LOW = 0
        PUD_DOWN = PUD_UP = PUD_OFF = 0
        _chip = None

        def _get_chip(self):
            global _lgpio_h
            if _lgpio_h is None:
                _lgpio_h = _lgpio.gpiochip_open(0)
            return _lgpio_h

        def setmode(self, _): pass
        def setwarnings(self, _): pass

        def setup(self, pin, direction, pull_up_down=None, initial=None):
            h = self._get_chip()
            if direction == 0:   # OUT
                _lgpio.gpio_claim_output(h, pin, 0 if initial is None else initial)
            else:                # IN
                _lgpio.gpio_claim_input(h, pin)

        def output(self, pin, value):
            _lgpio.gpio_write(self._get_chip(), pin, int(bool(value)))

        def input(self, pin):
            return _lgpio.gpio_read(self._get_chip(), pin)

        def PWM(self, pin, freq):
            return _LgpioPWM(self, pin, freq)

        def cleanup(self, pin=None):
            global _lgpio_h
            if _lgpio_h is not None:
                _lgpio.gpiochip_close(_lgpio_h)
                _lgpio_h = None

    GPIO  = _GPIOCompat()
    ON_PI = True
    print("[GPIO] using lgpio (Pi 5)")

except (ImportError, FileNotFoundError, OSError, PermissionError) as e:
    print(f"[GPIO] lgpio unavailable ({e}); trying RPi.GPIO")
    try:
        import RPi.GPIO as _RPIGPIO
        _RPIGPIO.setmode(_RPIGPIO.BCM)
        _RPIGPIO.setwarnings(False)

        class _RPICompat:
            BCM = _RPIGPIO.BCM
            OUT = _RPIGPIO.OUT
            IN  = _RPIGPIO.IN
            HIGH = _RPIGPIO.HIGH
            LOW  = _RPIGPIO.LOW
            PUD_DOWN = _RPIGPIO.PUD_DOWN
            PUD_UP   = _RPIGPIO.PUD_UP

            def setmode(self, m):       _RPIGPIO.setmode(m)
            def setwarnings(self, v):   _RPIGPIO.setwarnings(v)
            def setup(self, pin, direction, pull_up_down=None, initial=None):
                kwargs = {}
                if pull_up_down is not None: kwargs["pull_up_down"] = pull_up_down
                if initial is not None:      kwargs["initial"] = initial
                _RPIGPIO.setup(pin, direction, **kwargs)
            def output(self, pin, value): _RPIGPIO.output(pin, value)
            def input(self, pin):         return _RPIGPIO.input(pin)
            def PWM(self, pin, freq):     return _RPIGPIO.PWM(pin, freq)
            def cleanup(self, pin=None):
                if pin: _RPIGPIO.cleanup(pin)
                else:   _RPIGPIO.cleanup()

        GPIO  = _RPICompat()
        ON_PI = True
        print("[GPIO] using RPi.GPIO (Pi 1-4)")

    except Exception:
        ON_PI = False

# ─────────────────────────────────────────────────────────────────────────────
#  COLOURS / THEME  —  hazard-suit HUD palette (black / burnt orange / amber)
# ─────────────────────────────────────────────────────────────────────────────
BG       = "#050505"   # true near-black
PANEL    = "#0d0d0d"
CARD     = "#151515"
BORDER   = "#272727"   # neutral steel-gray border, used everywhere
ACCENT   = "#ff5a1f"   # HEV-orange, primary accent
GREEN    = "#5fd068"
RED      = "#ef4444"
ORANGE   = "#ff8a3d"
MUTED    = "#6e6e6e"
FG       = "#f4f4f2"   # clean near-white
SERVO_C  = "#d9922e"   # brass/amber
MOTOR_C  = "#4f8fae"   # muted steel-blue, kept for contrast
LASER_C  = "#ef4444"
CAM_C    = "#5fd068"
SOUND_C  = "#f5a623"
SENSOR_C = "#f5a623"
HAZARD_Y = "#e6b800"   # hazard-stripe accent (muted gold, subtler than before)
HAZARD_K = "#0d0d0d"   # hazard-stripe base — matches PANEL, recedes into bg
AI_BG    = "#170f05"   # AI panel header — dark amber-brown
AI_FG    = "#ffb04d"   # AI panel accent — bright amber
HOVER_OUTLINE = "#4a4a4a"


def _hazard_stripe(parent, height=3, stripe_w=14):
    """A slim, muted diagonal hazard-stripe divider (subtle industrial trim)."""
    bar = tk.Canvas(parent, height=height, bg=HAZARD_K, highlightthickness=0)
    bar.pack(fill="x")

    def _draw(event=None):
        bar.delete("all")
        w = bar.winfo_width() or 1200
        bar.create_rectangle(0, 0, w, height, fill=HAZARD_K, outline="")
        x = -height
        while x < w + height:
            bar.create_polygon(
                x, height, x + height, 0, x + height + stripe_w, 0, x + stripe_w, height,
                fill=HAZARD_Y, outline="")
            x += stripe_w * 3

    bar.bind("<Configure>", _draw)
    return bar


def _add_hover_outline(widget, thickness=1):
    """Gives a flat widget a subtle glowing border on mouse hover — the kind
    of small, deliberate interaction detail that separates a designed UI
    from unstyled default widgets."""
    widget.configure(highlightthickness=thickness, highlightbackground=BORDER,
                      highlightcolor=BORDER)
    widget.bind("<Enter>", lambda e: widget.configure(highlightbackground=FG))
    widget.bind("<Leave>", lambda e: widget.configure(highlightbackground=BORDER))

# ── pygame for audio ──────────────────────────────────────────────────────────
try:
    import pygame
    pygame.mixer.init()
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False

# ─────────────────────────────────────────────────────────────────────────────
#  HARDWARE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
_pwm_handles: dict = {}   # pin -> GPIO.PWM


def _servo_duty(angle, pulse_min=0.5, pulse_max=2.5, freq=50):
    pulse = pulse_min + (pulse_max - pulse_min) * (angle / 180.0)
    return (pulse / (1000.0 / freq)) * 100.0


def servo_set(pin: int, angle: float, pulse_min=0.5, pulse_max=2.5):
    if not ON_PI:
        print(f"[SIM] servo pin={pin} angle={angle:.1f}°")
        return
    try:
        import lgpio as _lgpio
        global _lgpio_h
        if _lgpio_h is None:
            _lgpio_h = _lgpio.gpiochip_open(0)
        pulse_us = int((pulse_min + (pulse_max - pulse_min) * (angle / 180.0)) * 1000)
        _lgpio.tx_servo(_lgpio_h, pin, pulse_us)
        def _stop():
            time.sleep(0.3)
            try: _lgpio.tx_servo(_lgpio_h, pin, 0)
            except Exception: pass
        threading.Thread(target=_stop, daemon=True).start()
    except (ImportError, AttributeError, Exception):
        if pin not in _pwm_handles:
            GPIO.setup(pin, GPIO.OUT)
            try:
                import RPi.GPIO as _RPIGPIO
                pwm = _RPIGPIO.PWM(pin, 50)
            except Exception:
                return
            pwm.start(0)
            _pwm_handles[pin] = pwm
        _pwm_handles[pin].ChangeDutyCycle(_servo_duty(angle, pulse_min, pulse_max))
        def _stop_pwm(p=pin):
            time.sleep(0.3)
            try: _pwm_handles[p].ChangeDutyCycle(0)
            except Exception: pass
        threading.Thread(target=_stop_pwm, daemon=True).start()


def laser_set(pin: int, state: bool):
    if not ON_PI:
        print(f"[SIM] laser pin={pin} {'ON' if state else 'OFF'}")
        return
    GPIO.setup(pin, GPIO.OUT)
    GPIO.output(pin, GPIO.HIGH if state else GPIO.LOW)


def motor_set(pin_fwd: int, pin_bwd: int, speed: int):
    """speed: -100..100"""
    if not ON_PI:
        print(f"[SIM] motor fwd={pin_fwd} bwd={pin_bwd} speed={speed}")
        return
    for p in (pin_fwd, pin_bwd):
        if p not in _pwm_handles:
            GPIO.setup(p, GPIO.OUT)
            pwm = GPIO.PWM(p, 1000)
            pwm.start(0)
            _pwm_handles[p] = pwm
    if speed >= 0:
        _pwm_handles[pin_fwd].ChangeDutyCycle(speed)
        _pwm_handles[pin_bwd].ChangeDutyCycle(0)
    else:
        _pwm_handles[pin_fwd].ChangeDutyCycle(0)
        _pwm_handles[pin_bwd].ChangeDutyCycle(-speed)



# ═════════════════════════════════════════════════════════════════════════════
#  POWER MONITORING — Pi health via vcgencmd (always available, no extra
#  hardware) + optional real current/power via an INA219 I2C sensor.
# ═════════════════════════════════════════════════════════════════════════════
# Bit meanings for `vcgencmd get_throttled` — straight from Raspberry Pi's
# own documentation. Bits 0-3 are the CURRENT state; bits 16-19 mean that
# condition has happened at some point since boot (even if it's fine now).
THROTTLE_FLAGS = {
    0:  "under-voltage detected",
    1:  "arm frequency capped",
    2:  "currently throttled",
    3:  "soft temp limit active",
    16: "under-voltage has occurred since boot",
    17: "frequency capping has occurred since boot",
    18: "throttling has occurred since boot",
    19: "soft temp limit has occurred since boot",
}

def read_pi_power():
    """Reads the Pi's own power/thermal health via vcgencmd — built into
    Raspberry Pi OS, no extra hardware needed. Returns a dict, or one with
    an 'error' key if vcgencmd isn't available (e.g. not actually on a Pi,
    or running the simulated/dev path)."""
    import subprocess
    result = {"volts_core": None, "temp_c": None,
             "throttled_raw": None, "flags": [], "ok": True}
    try:
        out = subprocess.check_output(["vcgencmd", "measure_volts", "core"],
                                      stderr=subprocess.DEVNULL, timeout=2).decode()
        result["volts_core"] = float(out.strip().split("=")[1].rstrip("V"))

        out = subprocess.check_output(["vcgencmd", "measure_temp"],
                                      stderr=subprocess.DEVNULL, timeout=2).decode()
        result["temp_c"] = float(out.strip().split("=")[1].rstrip("'C"))

        out = subprocess.check_output(["vcgencmd", "get_throttled"],
                                      stderr=subprocess.DEVNULL, timeout=2).decode()
        raw = int(out.strip().split("=")[1], 16)
        result["throttled_raw"] = raw
        result["flags"] = [msg for bit, msg in THROTTLE_FLAGS.items() if raw & (1 << bit)]
    except Exception as e:
        result["ok"] = False
        result["error"] = str(e)
    return result

_ina219_sensor = None
HAS_INA219 = False

def try_load_ina219(addr_str="0x40"):
    """Lazy-loads an INA219 current/power sensor over I2C. Requires real
    hardware wired in-line with whatever rail you want to measure, plus:
    pip install adafruit-circuitpython-ina219 --break-system-packages"""
    global _ina219_sensor, HAS_INA219
    if HAS_INA219:
        return True
    try:
        import board, busio
        from adafruit_ina219 import INA219
        addr = int(addr_str, 16) if isinstance(addr_str, str) else int(addr_str)
        i2c = busio.I2C(board.SCL, board.SDA)
        _ina219_sensor = INA219(i2c, addr=addr)
        HAS_INA219 = True
        return True
    except Exception as e:
        print(f"[POWER] INA219 unavailable: {e}  "
              f"(pip install adafruit-circuitpython-ina219 --break-system-packages, "
              f"and check it's actually wired to the I2C bus)")
        return False

def read_ina219():
    """Returns (voltage_v, current_ma, power_mw) from the INA219, or None
    if it's not loaded/available."""
    if not HAS_INA219 or _ina219_sensor is None:
        return None
    try:
        return (_ina219_sensor.bus_voltage, _ina219_sensor.current, _ina219_sensor.power)
    except Exception as e:
        print(f"[POWER] INA219 read error: {e}")
        return None


def cleanup_all():
    for pwm in _pwm_handles.values():
        try:
            pwm.stop()
        except Exception:
            pass
    if ON_PI:
        try:
            GPIO.cleanup()
        except Exception:
            pass


# ─────────────────────────────────────────────────────────────────────────────
#  COMPONENT DEFAULTS
# ─────────────────────────────────────────────────────────────────────────────
def default_servo():
    return {"type": "servo", "name": "Servo",
            "pin": 17, "min_deg": 0, "max_deg": 180,
            "pulse_min": 0.5, "pulse_max": 2.5, "step": 5,
            "key_left": "a", "key_right": "d", "key_reset": "s"}

def default_motor():
    return {"type": "motor", "name": "Motor",
            "pin_fwd": 20, "pin_bwd": 21,
            "key_fwd": "w", "key_bwd": "s", "key_stop": "x"}

def default_laser():
    return {"type": "laser", "name": "Laser", "pin": 24,
            "key_toggle": "f"}

def default_camera():
    return {"type": "camera", "name": "Camera",
            "index": 0,
            "tracking_servos": [],
            "tracking_enabled": False,
            "dead_zone": 30, "track_step": 2,
            "vision_enabled": False,
            "vision_model": "moondream",
            "vision_prompt": "Describe what you see and decide if any action is needed.",
            "vision_interval": 3.0,
            "vision_auto": False,
            # Local (offline, non-LLM) vision toolkit:
            "detect_faces": False,
            "recognize_faces": False,
            "detect_objects": False,
            "classify_scene": False,
            "detect_custom_images": False,
            "detect_pose": False,
            "known_faces_dir": "known_faces",
            "custom_images_dir": "reference_images"}

def default_sound():
    return {"type": "sound", "name": "Sounds",
            "folder": "", "files": [],
            "volume": 80}

def default_ultrasonic():
    return {"type": "sensor", "sensor_type": "ultrasonic", "name": "Ultrasonic",
            "pin_trig": 23, "pin_echo": 24,
            "alert_distance": 30, "alert_enabled": True}

def default_motion():
    return {"type": "sensor", "sensor_type": "motion", "name": "Motion",
            "pin": 25, "alert_enabled": True}

def default_temperature():
    return {"type": "sensor", "sensor_type": "temperature", "name": "Temperature",
            "pin": 4, "model": "DHT22",
            "alert_temp": 40.0, "alert_enabled": False}

def default_microphone():
    return {"type": "sensor", "sensor_type": "microphone", "name": "Microphone",
            "device_index": 0, "threshold": 500,
            "voice_commands": True,
            # Audio AI toolkit (mirrors the camera's Local Vision Toolkit):
            "detect_animal_sounds": False,
            "detect_alarms": False,
            "detect_custom_sounds": False,
            "speech_to_text": False,
            "translate_enabled": False,
            "translate_target_lang": "es",
            "custom_sounds_dir": "reference_sounds"}

def default_flock():
    # Companion sensor for a colonelpanichacks/flock-you (or oui-spy) ESP32
    # device: passively listens for Flock Safety camera WiFi probe/OUI
    # signatures and streams JSON detections over USB-CDC serial.
    # Repo: https://github.com/colonelpanichacks/flock-you
    return {"type": "sensor", "sensor_type": "flock", "name": "Flock Detector",
            "serial_port": "/dev/ttyUSB0", "baud_rate": 115200,
            "min_rssi": -90, "alert_enabled": True, "log_export": True}

def default_power():
    # Pi health (core voltage, under-voltage/throttle flags, temp) comes
    # free from `vcgencmd` — built into Raspberry Pi OS, no extra hardware.
    # Actual current draw of servos/motors/laser is NOT something the Pi
    # can see on its own — that needs a real INA219/INA226 current sensor
    # wired in-line with that rail, which is optional here.
    return {"type": "sensor", "sensor_type": "power", "name": "Power Monitor",
            "monitor_pi": True,
            "enable_ina219": False,
            "ina219_addr": "0x40",
            "ina219_label": "Servo Rail",
            "alert_on_undervoltage": True}

def default_flybrain():
    # A local spiking neural network controller, inspired by the viral
    # "fly brain plays Doom/Mario" projects that followed Google/Janelia's
    # MaleCNS v1.0 connectome release (Sept 2026). Runs entirely on-device —
    # no cloud, no internet at runtime.
    #
    # HONEST SCOPE: this does NOT run the actual 166,700-neuron MaleCNS
    # connectome — that dataset is gigabytes and those demos ran on much
    # more powerful hardware than a Pi. This is a real, working leaky-
    # integrate-and-fire spiking network sized to actually run in real time
    # here, defaulting to a randomly-wired synthetic network. If you obtain
    # a real (sampled/reduced) connectivity export — e.g. a CSV of
    # pre_id,post_id,weight from neuPrint or the FlyWire Codex — point
    # connectome_file at it and the real wiring gets used instead.
    return {"type": "flybrain", "name": "Fly Brain Controller",
            "connectome_file": "",       # pre/post/weight table: .csv or .feather
            "annotations_file": "",      # optional: cell-type/class table (.csv/.feather)
            "n_neurons": 500,            # synthetic network size if no file given
            "n_sensory": 64,             # first N neurons = visual input layer
            "n_motor": 8,                # last N neurons = motor output layer
            "sim_rate_hz": 10,           # simulation steps per second
            "target_servo_pan": "",
            "target_servo_tilt": "",
            "target_motor": "",
            "target_keys": "",           # e.g. "w,a,s,d" — one key per motor neuron
            "auto_reward_on_tracking": True}


# ═════════════════════════════════════════════════════════════════════════════
#  FLY BRAIN CONTROLLER — local spiking neural network engine
# ═════════════════════════════════════════════════════════════════════════════
class FlyBrainSim:
    """A leaky-integrate-and-fire spiking neural network, structured the same
    way the connectome-plays-games projects work: a sparse directed graph of
    neurons, sensory neurons driven by external input (here: a downsampled
    camera frame, standing in for the fly's optic lobe), spikes propagating
    along weighted synapses, and designated motor neurons whose firing rate
    becomes a control signal. A reward() call does simple Hebbian-style
    reinforcement of recently-active synapses feeding the rewarded neurons —
    the same crude "dopamine neuron" trick used in the Doom/trading demos.

    This is NOT a faithful biophysical simulation of real fly neurons, and
    with no file given it is NOT the actual MaleCNS connectome — it's a
    randomly-wired stand-in of the same size class, sized to actually run in
    real time on a Pi. Expect the same "mixed results" the real viral
    projects reported (undirected twitching more often than purposeful
    behavior) unless you spend time tuning it or feed it a real connectome.
    """

    def __init__(self, n_neurons=500, n_sensory=64, n_motor=8,
                connectome_file=None, annotations_file=None):
        self.n = n_neurons
        self.n_sensory = min(n_sensory, n_neurons // 4)
        self.n_motor   = min(n_motor, n_neurons // 8)
        self.tau  = 0.02     # membrane time constant (s)
        self.v_th = 1.0      # spike threshold
        self.v    = np.zeros(self.n, dtype=np.float32)
        self.refractory = np.zeros(self.n, dtype=np.float32)
        # Which row indices act as sensory/motor — overridden by real cell-type
        # annotations if an annotations_file is supplied; otherwise this stays
        # as an arbitrary first-N/last-N slice (honest fallback, not biology).
        self.sensory_idx = list(range(self.n_sensory))
        self.motor_idx   = list(range(self.n - self.n_motor, self.n))

        if connectome_file and os.path.isfile(connectome_file):
            self._load_connectome(connectome_file, annotations_file)
        else:
            self._generate_synthetic_network()

        self.motor_rate = np.zeros(self.n_motor, dtype=np.float32)
        self._recent_spikes = np.zeros(self.n, dtype=np.float32)  # for reward()

    def _generate_synthetic_network(self):
        """Random sparse directed graph — NOT the real fly connectome, just
        a stand-in of similar sparsity so the sim behaves architecturally
        the same way (sparse, mostly local, some long-range projections)."""
        rng = np.random.default_rng()
        density = min(0.02, 2000.0 / (self.n * self.n))  # keep it sparse
        mask = rng.random((self.n, self.n)) < density
        np.fill_diagonal(mask, False)
        weights = rng.normal(0.3, 0.15, size=(self.n, self.n)).astype(np.float32)
        weights[rng.random((self.n, self.n)) < 0.2] *= -1  # ~20% inhibitory
        self.W = np.where(mask, weights, 0.0).astype(np.float32)
        print(f"[FLYBRAIN] generated synthetic network: {self.n} neurons, "
              f"{int(mask.sum())} synapses (no real connectome file given)")

    @staticmethod
    def _read_table(path):
        """Reads a connectome/annotations table from either .feather (needs
        pandas + pyarrow) or .csv (works with just Python's stdlib as a
        fallback). Returns a pandas DataFrame either way if pandas is
        available, else a plain csv.reader iterator for the .csv path."""
        ext = os.path.splitext(path)[1].lower()
        if ext == ".feather":
            if not HAS_PANDAS:
                raise RuntimeError(
                    "reading .feather files needs pandas + pyarrow "
                    "(pip install pandas pyarrow --break-system-packages)")
            return pd.read_feather(path)
        if HAS_PANDAS:
            return pd.read_csv(path)
        import csv as _csv
        with open(path) as f:
            return list(_csv.DictReader(f))

    @staticmethod
    def _find_col(columns, candidates):
        """Column names shift between connectome dataset releases (e.g.
        body_pre vs bodyId_pre vs pre). Rather than hard-fail on a name
        mismatch, try each known candidate in turn."""
        cols_lower = {c.lower(): c for c in columns}
        for cand in candidates:
            if cand in cols_lower:
                return cols_lower[cand]
        return None

    def _load_connectome(self, path, annotations_path=None):
        """Loads a real pre/post/weight connectivity table — .csv or the
        official MaleCNS .feather export both work. Only the first
        n_neurons unique body IDs encountered are kept (real connectomes
        have far more neurons than can fit in a dense sim), so raise
        n_neurons in the editor if you want more of the graph used.
        Uses scipy sparse storage when available — required for anything
        beyond a few thousand neurons; a dense matrix would need gigabytes
        of RAM at real connectome scale.

        IMPORTANT: this uses vectorized pandas/numpy operations, not a
        Python-level row loop. The real MaleCNS connectome-weights file has
        millions of rows — looping it in pure Python would take an
        extremely long time on a Pi (looks exactly like "won't load", not
        an error), even though only the first n_neurons IDs actually get
        used in the end."""
        try:
            df = self._read_table(path)
            if not HAS_PANDAS:
                raise RuntimeError(
                    "pandas is required to load a real connectome file "
                    "(pip install pandas pyarrow --break-system-packages)")
            columns = df.columns
            pre_col = self._find_col(columns, ["body_pre", "bodyid_pre", "pre_bodyid", "pre"])
            post_col = self._find_col(columns, ["body_post", "bodyid_post", "post_bodyid", "post"])
            weight_col = self._find_col(columns, ["weight", "weightpaths", "count", "syn_count"])
            if not (pre_col and post_col):
                raise RuntimeError(
                    f"couldn't find pre/post neuron-ID columns among {list(columns)} "
                    f"— check the file matches a neuPrint/FlyWire connection export")

            # First n_neurons unique IDs, in order of first appearance —
            # done with pd.unique on the combined pre+post series (vectorized,
            # handles millions of rows in well under a second).
            all_ids = pd.concat([df[pre_col], df[post_col]], ignore_index=True)
            unique_ids = pd.unique(all_ids)[: self.n]
            id_map = {bid: i for i, bid in enumerate(unique_ids)}

            # Keep only edges where BOTH endpoints survived the cap —
            # vectorized boolean mask, not a per-row Python check.
            mask = df[pre_col].isin(id_map) & df[post_col].isin(id_map)
            kept = df.loc[mask]
            rows = kept[pre_col].map(id_map).to_numpy()
            cols = kept[post_col].map(id_map).to_numpy()
            weights = (kept[weight_col].to_numpy(dtype=np.float32)
                      if weight_col else np.ones(len(kept), dtype=np.float32))

            if HAS_SCIPY:
                self.W = sp.csr_matrix((weights, (rows, cols)), shape=(self.n, self.n),
                                       dtype=np.float32)
            else:
                self.W = np.zeros((self.n, self.n), dtype=np.float32)
                self.W[rows, cols] = weights
                print("[FLYBRAIN] scipy not installed — using a dense matrix. "
                      "Fine for a few hundred neurons; install scipy for "
                      "larger real-connectome slices "
                      "(pip install scipy --break-system-packages)")

            print(f"[FLYBRAIN] loaded connectome from {path}: "
                  f"{len(id_map)} neurons (of {self.n} slots), {len(rows)} synapses"
                  f"{' [sparse]' if HAS_SCIPY else ' [dense]'}")

            if annotations_path and os.path.isfile(annotations_path):
                self._apply_annotations(annotations_path, id_map)
        except Exception as e:
            print(f"[FLYBRAIN] connectome load failed ({e}), falling back "
                  f"to synthetic network")
            self._generate_synthetic_network()

    def _apply_annotations(self, path, id_map):
        """Uses real cell-type/class annotations (bodyId/type/class columns
        in the MaleCNS release) to pick ACTUAL visual/optic-lobe neurons as
        sensory inputs and ACTUAL descending neurons as motor outputs,
        instead of an arbitrary first-N/last-N slice of whatever order the
        connectivity table happened to list neurons in. This is the single
        biggest accuracy upgrade available without writing a custom
        biophysical model — it means the 'sensory' and 'motor' neurons are
        the ones the real fly actually uses for vision and movement.

        Uses vectorized pandas string matching, not a Python row loop —
        also sidesteps a real bug the row-loop version had: the real
        MaleCNS annotations file has a column literally named 'class',
        which is a reserved Python keyword and can't be read as a
        namedtuple attribute via itertuples() (df.itertuples() silently
        can't expose it that way), so the per-row version always failed
        on the exact file this feature exists for."""
        try:
            df = self._read_table(path)
            if not HAS_PANDAS:
                print("[FLYBRAIN] pandas required to read annotations — skipping "
                      "(pip install pandas pyarrow --break-system-packages)")
                return
            columns = df.columns
            id_col    = self._find_col(columns, ["bodyid", "body_id", "id"])
            class_col = self._find_col(columns, ["class", "cell_type", "type", "primary_type"])
            if not (id_col and class_col):
                print(f"[FLYBRAIN] annotations file missing expected columns "
                      f"(found {list(columns)}) — keeping first-N/last-N fallback")
                return

            # Only look at rows whose body ID actually made it into the
            # loaded connectome slice — no point classifying neurons that
            # got cut off by n_neurons anyway.
            df = df[df[id_col].isin(id_map)]
            cls_lower = df[class_col].astype(str).str.lower()

            sensory_mask = cls_lower.str.contains(
                "visual|optic|photoreceptor|ol_", regex=True, na=False)
            motor_mask = (cls_lower.str.contains("descending", na=False)
                         | cls_lower.str.startswith("dn"))

            sensory_ids = [id_map[b] for b in df.loc[sensory_mask, id_col]]
            motor_ids   = [id_map[b] for b in df.loc[motor_mask, id_col]]

            if sensory_ids:
                self.sensory_idx = sensory_ids[:self.n_sensory]
                self.n_sensory = len(self.sensory_idx)
            if motor_ids:
                self.motor_idx = motor_ids[:self.n_motor]
                self.n_motor = len(self.motor_idx)
            print(f"[FLYBRAIN] applied annotations: {len(self.sensory_idx)} real "
                  f"visual/optic neurons as sensory input, {len(self.motor_idx)} "
                  f"real descending neurons as motor output")
        except Exception as e:
            print(f"[FLYBRAIN] annotations load failed ({e}) — keeping "
                  f"first-N/last-N fallback")

    def set_sensory_input(self, values):
        """values: array-like matching n_sensory, roughly 0..1 (e.g. per-cell
        average brightness from a downsampled camera frame). Applied to the
        actual designated sensory neuron indices (self.sensory_idx), which
        are real visual/optic neurons if annotations were loaded, otherwise
        the first-N fallback slice."""
        n = min(len(values), len(self.sensory_idx))
        idx = self.sensory_idx[:n]
        self.v[idx] += np.asarray(values[:n], dtype=np.float32) * 0.5

    def step(self, dt=0.1):
        """Advances the simulation by dt seconds. Returns the motor neurons'
        current firing state (bool array, length n_motor)."""
        # Leaky integrate-and-fire update. self.W may be a dense numpy array
        # or a scipy sparse matrix (real connectome slices need sparse —
        # a dense matrix at real connectome scale would need gigabytes).
        raw_current = self.W.T @ self._recent_spikes
        input_current = np.asarray(raw_current).flatten()
        self.v += (-self.v / self.tau + input_current) * dt
        self.refractory = np.maximum(0, self.refractory - dt)

        spiking = (self.v >= self.v_th) & (self.refractory <= 0)
        self._recent_spikes = spiking.astype(np.float32)
        self.v[spiking] = 0.0
        self.refractory[spiking] = 0.005

        motor_spikes = spiking[self.motor_idx]
        # Exponential moving average firing rate, for a smoother control
        # signal than raw per-step spikes (which would be very jittery).
        self.motor_rate = 0.8 * self.motor_rate + 0.2 * motor_spikes.astype(np.float32)
        return motor_spikes

    def reward(self, magnitude=1.0):
        """Crude dopamine-style reinforcement: strengthens synapses that
        were active into the currently-most-active motor neuron, the same
        trick the viral Doom/trading demos used ('stimulate dopamine
        neurons on reward'). Not a real learning rule — just enough
        plasticity to let good-by-accident behavior get reinforced."""
        if self.motor_rate.max() < 0.01:
            return
        best_motor_idx = self.motor_idx[int(np.argmax(self.motor_rate))]
        active_pre = np.where(self._recent_spikes > 0)[0]
        if len(active_pre) == 0:
            return
        if HAS_SCIPY and sp.issparse(self.W):
            # scipy's LIL matrix doesn't support fancy-index += for this
            # exact shape (many rows, one scalar column) — it tries to
            # broadcast-add across the WHOLE matrix and raises
            # NotImplementedError. active_pre is small (just the neurons
            # that spiked this instant), so a plain per-row loop is both
            # correct and plenty fast here.
            w_lil = self.W.tolil()
            for r in active_pre:
                w_lil[r, best_motor_idx] = w_lil[r, best_motor_idx] + magnitude * 0.05
            self.W = w_lil.tocsr()
            self.W.data = np.clip(self.W.data, -2.0, 2.0)
        else:
            self.W[active_pre, best_motor_idx] += magnitude * 0.05
            np.clip(self.W, -2.0, 2.0, out=self.W)


# ═════════════════════════════════════════════════════════════════════════════
#  COMPONENT EDITOR DIALOG
# ═════════════════════════════════════════════════════════════════════════════
class ComponentEditor(tk.Toplevel):
    def __init__(self, parent, comp: dict, servo_names: list):
        super().__init__(parent)
        self.configure(bg=PANEL)
        self.resizable(False, True)
        self.geometry("440x620")
        self.grab_set()
        self.result = None
        self._comp = copy.deepcopy(comp)
        self._vars = {}
        self._servo_names = servo_names

        t = comp["type"]
        self.title(f"Edit {t.capitalize()}")

        f_h = tkfont.Font(family="Courier", size=12, weight="bold")
        f_n = tkfont.Font(family="Courier", size=11)
        f_s = tkfont.Font(family="Courier", size=10)

        # Save/Cancel reserved at the bottom FIRST so the scroll area above
        # it always shrinks to fit rather than pushing these off-screen.
        btn_row = tk.Frame(self, bg=PANEL)
        btn_row.pack(side="bottom", pady=12)
        tk.Button(btn_row, text="SAVE", bg=GREEN, fg="#fff",
                  font=f_n, relief="flat", width=10,
                  command=self._save).pack(side="left", padx=8)
        tk.Button(btn_row, text="CANCEL", bg=BORDER, fg=MUTED,
                  font=f_n, relief="flat", width=10,
                  command=self.destroy).pack(side="left", padx=8)

        # ── Scrollable body (mouse-wheel enabled) ───────────────────────────
        body_outer = tk.Frame(self, bg=PANEL)
        body_outer.pack(side="top", fill="both", expand=True)

        canvas  = tk.Canvas(body_outer, bg=PANEL, highlightthickness=0)
        vscroll = tk.Scrollbar(body_outer, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self._scroll_frame = tk.Frame(canvas, bg=PANEL)
        _window = canvas.create_window((0, 0), window=self._scroll_frame, anchor="nw")

        def _on_frame_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        self._scroll_frame.bind("<Configure>", _on_frame_configure)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(_window, width=e.width))

        def _on_mousewheel(event):
            delta = -1 if event.num == 4 else 1 if event.num == 5 else -event.delta // 120
            canvas.yview_scroll(delta, "units")
        canvas.bind("<Enter>", lambda e: (
            canvas.bind_all("<MouseWheel>", _on_mousewheel),
            canvas.bind_all("<Button-4>", _on_mousewheel),
            canvas.bind_all("<Button-5>", _on_mousewheel)))
        canvas.bind("<Leave>", lambda e: (
            canvas.unbind_all("<MouseWheel>"),
            canvas.unbind_all("<Button-4>"),
            canvas.unbind_all("<Button-5>")))

        tk.Label(self._scroll_frame, text=f"── {t.upper()} SETTINGS ──",
                 bg=PANEL, fg=ACCENT, font=f_h).pack(pady=(14, 6))

        if t == "camera":
            self._build_camera_ui(f_h, f_n, f_s)
        elif t == "sound":
            self._build_sound_ui(f_n, f_s)
        elif t == "sensor":
            self._build_sensor_ui(f_n, f_s)
        elif t == "flybrain":
            self._build_flybrain_ui(f_n, f_s)
        else:
            self._build_generic_ui(t, f_n)

    # ── Generic editor (servo / motor / laser) ────────────────────────────
    def _build_generic_ui(self, t, f_n):
        form = tk.Frame(self._scroll_frame, bg=PANEL)
        form.pack(padx=20, pady=6)
        fields = self._fields_for(t)
        for row, (label, key, kind) in enumerate(fields):
            tk.Label(form, text=label, bg=PANEL, fg=MUTED,
                     font=f_n, anchor="w", width=18).grid(
                row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(self._comp.get(key, "")))
            self._vars[key] = var
            tk.Entry(form, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_n,
                     relief="flat", width=18).grid(row=row, column=1, padx=8)

    def _fields_for(self, t):
        if t == "servo":
            return [
                ("Name",        "name",      "entry"),
                ("GPIO Pin",    "pin",       "entry"),
                ("Min Angle",   "min_deg",   "entry"),
                ("Max Angle",   "max_deg",   "entry"),
                ("Step deg",    "step",      "entry"),
                ("Pulse Min ms","pulse_min", "entry"),
                ("Pulse Max ms","pulse_max", "entry"),
                ("Key Left",    "key_left",  "entry"),
                ("Key Right",   "key_right", "entry"),
                ("Key Reset",   "key_reset", "entry"),
            ]
        elif t == "motor":
            return [
                ("Name",         "name",    "entry"),
                ("GPIO Fwd Pin", "pin_fwd", "entry"),
                ("GPIO Bwd Pin", "pin_bwd", "entry"),
                ("Key Forward",  "key_fwd", "entry"),
                ("Key Backward", "key_bwd", "entry"),
                ("Key Stop",     "key_stop","entry"),
            ]
        elif t == "laser":
            return [
                ("Name",       "name",       "entry"),
                ("GPIO Pin",   "pin",        "entry"),
                ("Key Toggle", "key_toggle", "entry"),
            ]
        return []

    # ── Camera-specific editor ────────────────────────────────────────────
    def _build_camera_ui(self, f_h, f_n, f_s):
        outer = tk.Frame(self._scroll_frame, bg=PANEL)
        outer.pack(padx=20, pady=4, fill="both")

        # Basic fields
        form = tk.Frame(outer, bg=PANEL)
        form.pack(fill="x")
        cam_fields = [("Name", "name"), ("Camera Index", "index"),
                      ("Dead Zone px", "dead_zone"), ("Track Step °", "track_step")]
        for row, (label, key) in enumerate(cam_fields):
            tk.Label(form, text=label, bg=PANEL, fg=MUTED,
                     font=f_n, anchor="w", width=18).grid(
                row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(self._comp.get(key, "")))
            self._vars[key] = var
            tk.Entry(form, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_n,
                     relief="flat", width=18).grid(
                row=row, column=1, padx=8)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=10)

        # ── Tracking enable/disable toggle ────────────────────────────────
        tog_row = tk.Frame(outer, bg=PANEL)
        tog_row.pack(fill="x", pady=4)
        tk.Label(tog_row, text="Face Tracking",
                 bg=PANEL, fg=MUTED, font=f_n, width=18, anchor="w").pack(side="left")

        self._track_enabled = tk.BooleanVar(
            value=self._comp.get("tracking_enabled", False))

        self._tog_btn = tk.Button(
            tog_row,
            text="* ENABLED" if self._track_enabled.get() else "o DISABLED",
            bg=GREEN if self._track_enabled.get() else BORDER,
            fg="#fff", font=f_n, relief="flat", cursor="hand2",
            width=12, command=self._toggle_tracking)
        self._tog_btn.pack(side="left", padx=8)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=10)

        # ── Servo list ────────────────────────────────────────────────────
        tk.Label(outer, text="TRACKING SERVOS",
                 bg=PANEL, fg=ACCENT, font=f_s).pack(anchor="w")
        tk.Label(outer, text="Add servos that will follow the face",
                 bg=PANEL, fg=MUTED, font=f_s).pack(anchor="w", pady=(0, 6))

        # Current tracking servos list
        list_frame = tk.Frame(outer, bg=CARD,
                              highlightthickness=1, highlightbackground=BORDER)
        list_frame.pack(fill="x", pady=4)

        self._servo_list_frame = tk.Frame(list_frame, bg=CARD)
        self._servo_list_frame.pack(fill="x", padx=8, pady=6)

        self._tracking_servos = list(self._comp.get("tracking_servos", []))
        self._refresh_servo_list(f_s)

        # Add servo dropdown
        if self._servo_names:
            add_row = tk.Frame(outer, bg=PANEL)
            add_row.pack(fill="x", pady=6)
            tk.Label(add_row, text="Add servo:",
                     bg=PANEL, fg=MUTED, font=f_s).pack(side="left")
            self._add_var = tk.StringVar(value=self._servo_names[0])
            cb = ttk.Combobox(add_row, textvariable=self._add_var,
                              values=self._servo_names, width=14, font=f_s)
            cb.pack(side="left", padx=8)
            tk.Button(add_row, text="+ ADD", bg=SERVO_C, fg="#fff",
                      font=f_s, relief="flat", cursor="hand2",
                      command=lambda: self._add_servo(f_s)).pack(side="left")
        else:
            tk.Label(outer, text="⚠ No servos in build - add servos first!",
                     bg=PANEL, fg=ORANGE, font=f_s).pack(pady=4)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=10)

        # ── AI Vision section ─────────────────────────────────────────────
        tk.Label(outer, text="AI VISION  (Ollama)",
                 bg=PANEL, fg=ACCENT, font=f_s).pack(anchor="w")

        vis_tog_row = tk.Frame(outer, bg=PANEL)
        vis_tog_row.pack(fill="x", pady=4)
        tk.Label(vis_tog_row, text="Vision Mode",
                 bg=PANEL, fg=MUTED, font=f_n, width=18, anchor="w").pack(side="left")

        self._vision_enabled = tk.BooleanVar(
            value=self._comp.get("vision_enabled", False))
        self._vis_tog_btn = tk.Button(
            vis_tog_row,
            text="* ENABLED" if self._vision_enabled.get() else "o DISABLED",
            bg=CAM_C if self._vision_enabled.get() else BORDER,
            fg="#fff", font=f_n, relief="flat", cursor="hand2",
            width=12, command=self._toggle_vision)
        self._vis_tog_btn.pack(side="left", padx=8)

        vis_form = tk.Frame(outer, bg=PANEL)
        vis_form.pack(fill="x")
        for row, (label, key, default) in enumerate([
            ("Vision Model",  "vision_model",    "moondream"),
            ("Interval (sec)","vision_interval",  "3.0"),
        ]):
            tk.Label(vis_form, text=label, bg=PANEL, fg=MUTED,
                     font=f_s, anchor="w", width=18).grid(
                row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(self._comp.get(key, default)))
            self._vars[key] = var
            tk.Entry(vis_form, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_s,
                     relief="flat", width=18).grid(row=row, column=1, padx=8)

        tk.Label(outer, text="Vision Prompt:", bg=PANEL, fg=MUTED,
                 font=f_s).pack(anchor="w", pady=(6, 2))
        self._vision_prompt_txt = tk.Text(outer, bg=CARD, fg=FG,
                                           insertbackground=FG, font=f_s,
                                           relief="flat", height=3, width=40,
                                           wrap="word")
        self._vision_prompt_txt.insert("1.0", self._comp.get(
            "vision_prompt",
            "Describe what you see and decide if any action is needed."))
        self._vision_prompt_txt.pack(fill="x", pady=2)

        auto_row = tk.Frame(outer, bg=PANEL)
        auto_row.pack(fill="x", pady=4)
        tk.Label(auto_row, text="Auto Analyze",
                 bg=PANEL, fg=MUTED, font=f_s, width=18, anchor="w").pack(side="left")
        self._vision_auto = tk.BooleanVar(value=self._comp.get("vision_auto", False))
        tk.Checkbutton(auto_row, variable=self._vision_auto,
                       bg=PANEL, fg=FG, selectcolor=CARD,
                       activebackground=PANEL, font=f_s,
                       text="Analyze every N seconds automatically"
                       ).pack(side="left")

        # ── Local vision toolkit (offline, no LLM) ─────────────────────────
        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=8)
        tk.Label(outer, text="LOCAL VISION TOOLKIT  (offline)",
                 bg=PANEL, fg=ACCENT, font=f_s).pack(anchor="w")

        self._local_vision_vars = {}
        for label, key, note in [
            ("Face Detection",   "detect_faces",    "boxes only (START TRACKING adds servo lock)"),
            ("Face Recognition", "recognize_faces", "names faces; needs opencv-contrib-python"),
            ("Object Detection", "detect_objects",  "needs ultralytics"),
            ("Scene Classification", "classify_scene", "needs ultralytics"),
            ("Custom Image Recognition", "detect_custom_images",
             "match against your own photos of anything"),
            ("Pose / Skeleton Tracking", "detect_pose", "needs ultralytics"),
        ]:
            row = tk.Frame(outer, bg=PANEL)
            row.pack(fill="x", pady=2)
            var = tk.BooleanVar(value=self._comp.get(key, False))
            self._local_vision_vars[key] = var
            tk.Checkbutton(row, variable=var, bg=PANEL, fg=FG,
                           selectcolor=CARD, activebackground=PANEL,
                           font=f_s, text=label, width=18, anchor="w"
                           ).pack(side="left")
            tk.Label(row, text=note, bg=PANEL, fg=MUTED, font=f_s
                     ).pack(side="left", padx=4)

        dir_row = tk.Frame(outer, bg=PANEL)
        dir_row.pack(fill="x", pady=(6, 2))
        for label, key, default in [
            ("Known Faces Dir", "known_faces_dir", "known_faces"),
            ("Reference Images Dir", "custom_images_dir", "reference_images"),
        ]:
            r = tk.Frame(outer, bg=PANEL)
            r.pack(fill="x", pady=2)
            tk.Label(r, text=label, bg=PANEL, fg=MUTED,
                     font=f_s, anchor="w", width=18).pack(side="left")
            var = tk.StringVar(value=str(self._comp.get(key, default)))
            self._vars[key] = var
            tk.Entry(r, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_s,
                     relief="flat", width=18).pack(side="left", padx=8)

    def _toggle_vision(self):
        self._vision_enabled.set(not self._vision_enabled.get())
        if self._vision_enabled.get():
            self._vis_tog_btn.config(text="* ENABLED", bg=CAM_C)
        else:
            self._vis_tog_btn.config(text="o DISABLED", bg=BORDER)

    # ── Sound editor ──────────────────────────────────────────────────────
    def _build_sound_ui(self, f_n, f_s):
        outer = tk.Frame(self._scroll_frame, bg=PANEL)
        outer.pack(padx=20, pady=4, fill="both")

        # Name & volume
        form = tk.Frame(outer, bg=PANEL)
        form.pack(fill="x")
        for row, (label, key, default) in enumerate([
            ("Name",     "name",   "Sounds"),
            ("Volume %", "volume", "80"),
        ]):
            tk.Label(form, text=label, bg=PANEL, fg=MUTED,
                     font=f_n, anchor="w", width=18).grid(
                row=row, column=0, sticky="w", pady=3)
            var = tk.StringVar(value=str(self._comp.get(key, default)))
            self._vars[key] = var
            tk.Entry(form, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_n,
                     relief="flat", width=18).grid(row=row, column=1, padx=8)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=10)

        # Folder picker
        folder_row = tk.Frame(outer, bg=PANEL)
        folder_row.pack(fill="x", pady=4)
        tk.Label(folder_row, text="Folder:", bg=PANEL, fg=MUTED,
                 font=f_s).pack(side="left")
        self._folder_var = tk.StringVar(value=self._comp.get("folder", ""))
        tk.Entry(folder_row, textvariable=self._folder_var, bg=CARD, fg=FG,
                 insertbackground=FG, font=f_s, relief="flat", width=22
                 ).pack(side="left", padx=6)
        tk.Button(folder_row, text="Browse", bg=BORDER, fg=FG,
                  font=f_s, relief="flat", cursor="hand2",
                  command=self._browse_folder).pack(side="left")

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=8)

        # File list
        tk.Label(outer, text="MP3 FILES", bg=PANEL, fg=ACCENT, font=f_s).pack(anchor="w")

        list_frame = tk.Frame(outer, bg=CARD,
                              highlightthickness=1, highlightbackground=BORDER)
        list_frame.pack(fill="x", pady=4)
        self._sound_list_frame = tk.Frame(list_frame, bg=CARD)
        self._sound_list_frame.pack(fill="x", padx=8, pady=6)

        self._sound_files = list(self._comp.get("files", []))
        self._refresh_sound_list(f_s)

        # Add file buttons
        add_row = tk.Frame(outer, bg=PANEL)
        add_row.pack(fill="x", pady=6)
        tk.Button(add_row, text="+ Add MP3 File", bg=SOUND_C, fg="#000",
                  font=f_s, relief="flat", cursor="hand2",
                  command=lambda: self._add_sound_file(f_s)).pack(side="left", padx=4)
        tk.Button(add_row, text="+ Add from Folder", bg=BORDER, fg=FG,
                  font=f_s, relief="flat", cursor="hand2",
                  command=lambda: self._add_from_folder(f_s)).pack(side="left", padx=4)

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select Sound Folder")
        if folder:
            self._folder_var.set(folder)

    def _refresh_sound_list(self, f_s):
        for w in self._sound_list_frame.winfo_children():
            w.destroy()
        if not self._sound_files:
            tk.Label(self._sound_list_frame, text="No files added yet",
                     bg=CARD, fg=MUTED, font=f_s).pack()
            return
        for path in self._sound_files:
            row = tk.Frame(self._sound_list_frame, bg=CARD)
            row.pack(fill="x", pady=2)
            name = os.path.basename(path)
            tk.Label(row, text=f"[A]  {name}", bg=CARD, fg=FG,
                     font=f_s, anchor="w").pack(side="left", fill="x", expand=True)
            tk.Button(row, text="✕", bg=RED, fg="#fff",
                      font=f_s, relief="flat", cursor="hand2",
                      command=lambda p=path: self._remove_sound(p, f_s)
                      ).pack(side="right")

    def _add_sound_file(self, f_s):
        path = filedialog.askopenfilename(
            title="Select MP3 / WAV file",
            filetypes=[("Audio", "*.mp3 *.wav *.ogg"), ("All", "*.*")])
        if path and path not in self._sound_files:
            self._sound_files.append(path)
            self._refresh_sound_list(f_s)

    def _add_from_folder(self, f_s):
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if folder:
            self._folder_var.set(folder)
            for f in sorted(os.listdir(folder)):
                if f.lower().endswith((".mp3", ".wav", ".ogg")):
                    full = os.path.join(folder, f)
                    if full not in self._sound_files:
                        self._sound_files.append(full)
            self._refresh_sound_list(f_s)

    def _remove_sound(self, path, f_s):
        if path in self._sound_files:
            self._sound_files.remove(path)
            self._refresh_sound_list(f_s)

    # ── Sensor editor ─────────────────────────────────────────────────────
    def _build_sensor_ui(self, f_n, f_s):
        outer = tk.Frame(self._scroll_frame, bg=PANEL)
        outer.pack(padx=20, pady=4, fill="both")

        st = self._comp.get("sensor_type", "ultrasonic")

        # Common: name
        name_row = tk.Frame(outer, bg=PANEL)
        name_row.pack(fill="x", pady=4)
        tk.Label(name_row, text="Name", bg=PANEL, fg=MUTED,
                 font=f_n, width=18, anchor="w").pack(side="left")
        var = tk.StringVar(value=self._comp.get("name", "Sensor"))
        self._vars["name"] = var
        tk.Entry(name_row, textvariable=var, bg=CARD, fg=FG,
                 insertbackground=FG, font=f_n, relief="flat", width=18).pack(side="left", padx=8)

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=8)

        # Sensor-specific fields
        if st == "ultrasonic":
            fields = [
                ("Trigger Pin", "pin_trig", "23"),
                ("Echo Pin",    "pin_echo", "24"),
                ("Alert Distance cm", "alert_distance", "30"),
            ]
            bool_fields = [("Alert Enabled", "alert_enabled")]

        elif st == "motion":
            fields = [("GPIO Pin", "pin", "25")]
            bool_fields = [("Alert Enabled", "alert_enabled")]

        elif st == "temperature":
            fields = [
                ("GPIO Pin",   "pin",        "4"),
                ("Model",      "model",      "DHT22"),
                ("Alert Temp C", "alert_temp", "40"),
            ]
            bool_fields = [("Alert Enabled", "alert_enabled")]

        elif st == "microphone":
            fields = [
                ("Device Index", "device_index", "0"),
                ("Noise Threshold", "threshold", "500"),
                ("Custom Sounds Dir", "custom_sounds_dir", "reference_sounds"),
                ("Translate To (lang)", "translate_target_lang", "es"),
            ]
            bool_fields = [
                ("Voice Commands", "voice_commands"),
                ("Animal Sound Recognition", "detect_animal_sounds"),
                ("Alarm / Siren Recognition", "detect_alarms"),
                ("Custom Sound Recognition", "detect_custom_sounds"),
                ("Speech-to-Text", "speech_to_text"),
                ("Translate Speech", "translate_enabled"),
            ]

        elif st == "flock":
            fields = [
                ("Serial Port",  "serial_port",  "/dev/ttyUSB0"),
                ("Baud Rate",    "baud_rate",    "115200"),
                ("Min RSSI dBm", "min_rssi",     "-90"),
            ]
            bool_fields = [
                ("Alert Enabled", "alert_enabled"),
                ("Log to KML/CSV", "log_export"),
            ]
        elif st == "power":
            fields = [
                ("INA219 I2C Addr", "ina219_addr", "0x40"),
                ("INA219 Rail Label", "ina219_label", "Servo Rail"),
            ]
            bool_fields = [
                ("Monitor Pi Health (vcgencmd)", "monitor_pi"),
                ("Enable INA219 Current Sensor", "enable_ina219"),
                ("Alert on Under-Voltage", "alert_on_undervoltage"),
            ]
        else:
            fields, bool_fields = [], []

        for label, key, default in fields:
            row = tk.Frame(outer, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                     font=f_n, width=18, anchor="w").pack(side="left")
            var = tk.StringVar(value=str(self._comp.get(key, default)))
            self._vars[key] = var
            tk.Entry(row, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_n, relief="flat", width=18).pack(side="left", padx=8)

        for label, key in bool_fields:
            row = tk.Frame(outer, bg=PANEL)
            row.pack(fill="x", pady=3)
            tk.Label(row, text=label, bg=PANEL, fg=MUTED,
                     font=f_n, width=18, anchor="w").pack(side="left")
            var = tk.BooleanVar(value=self._comp.get(key, True))
            self._vars[key] = var
            tk.Checkbutton(row, variable=var, bg=PANEL, fg=FG,
                           selectcolor=CARD, activebackground=PANEL,
                           font=f_n).pack(side="left")

        if st == "power":
            tk.Label(outer,
                     text="Pi health (voltage/throttle/temp) needs no extra\n"
                          "hardware — it's built into Raspberry Pi OS.\n"
                          "The INA219 current sensor is optional and needs:\n"
                          "pip install adafruit-circuitpython-ina219\n"
                          "--break-system-packages, wired in-line with\n"
                          "whatever rail you want real current/power numbers for.",
                     bg=PANEL, fg=MUTED, font=("Courier", 8), justify="left",
                     wraplength=380).pack(anchor="w", pady=(8, 0))

    def _toggle_tracking(self):
        self._track_enabled.set(not self._track_enabled.get())
        if self._track_enabled.get():
            self._tog_btn.config(text="* ENABLED", bg=GREEN)
        else:
            self._tog_btn.config(text="o DISABLED", bg=BORDER)

    def _refresh_servo_list(self, f_s):
        for w in self._servo_list_frame.winfo_children():
            w.destroy()
        if not self._tracking_servos:
            tk.Label(self._servo_list_frame, text="No tracking servos added yet",
                     bg=CARD, fg=MUTED, font=f_s).pack()
        for name in self._tracking_servos:
            row = tk.Frame(self._servo_list_frame, bg=CARD)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"[S]  {name}", bg=CARD, fg=FG,
                     font=f_s, width=20, anchor="w").pack(side="left")
            tk.Button(row, text="✕", bg=RED, fg="#fff",
                      font=f_s, relief="flat", cursor="hand2",
                      command=lambda n=name: self._remove_servo(n, f_s)
                      ).pack(side="right")

    def _add_servo(self, f_s):
        name = self._add_var.get()
        if name and name not in self._tracking_servos:
            self._tracking_servos.append(name)
            self._refresh_servo_list(f_s)

    def _remove_servo(self, name, f_s):
        if name in self._tracking_servos:
            self._tracking_servos.remove(name)
            self._refresh_servo_list(f_s)

    def _build_flybrain_ui(self, f_n, f_s):
        outer = tk.Frame(self._scroll_frame, bg=PANEL)
        outer.pack(padx=16, pady=6, fill="x")

        tk.Label(outer,
                 text="Local spiking neural network controller.\n"
                      "No connectome file = random synthetic network\n"
                      "(same size class, NOT the real fly connectome).",
                 bg=PANEL, fg=MUTED, font=f_s, justify="left",
                 wraplength=380).pack(anchor="w", pady=(0, 10))

        def _file_field(label_text, key, note=None):
            """Stacked label-above-field layout — avoids the truncation
            that happens cramming a wide label + entry + browse button onto
            one line in a narrow dialog."""
            tk.Label(outer, text=label_text, bg=PANEL, fg=MUTED,
                     font=f_s, anchor="w").pack(fill="x", pady=(4, 1))
            row = tk.Frame(outer, bg=PANEL)
            row.pack(fill="x")
            var = tk.StringVar(value=str(self._comp.get(key, "")))
            self._vars[key] = var
            tk.Entry(row, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_s, relief="flat"
                     ).pack(side="left", fill="x", expand=True, ipady=3)
            tk.Button(row, text="...", bg=BORDER, fg=FG, font=f_s,
                      relief="flat", cursor="hand2", padx=8,
                      command=lambda: var.set(
                          filedialog.askopenfilename(
                              filetypes=[("Connectome/annotation data",
                                        "*.feather *.csv"), ("All", "*.*")])
                          or var.get())
                      ).pack(side="left", padx=(4, 0))
            if note:
                tk.Label(outer, text=note, bg=PANEL, fg=MUTED,
                         font=("Courier", 8), anchor="w",
                         wraplength=380, justify="left").pack(fill="x", pady=(1, 4))

        _file_field("Connectome File", "connectome_file")
        if not HAS_PANDAS:
            tk.Label(outer, text="⚠ .feather needs: pip install pandas pyarrow "
                                 "--break-system-packages",
                     bg=PANEL, fg=ORANGE, font=("Courier", 8), anchor="w",
                     wraplength=380, justify="left").pack(fill="x", pady=(0, 4))

        _file_field("Annotations File", "annotations_file",
                   note="optional — real cell types → real sensory/motor neurons")

        def _text_field(label_text, key, default):
            tk.Label(outer, text=label_text, bg=PANEL, fg=MUTED,
                     font=f_s, anchor="w").pack(fill="x", pady=(6, 1))
            var = tk.StringVar(value=str(self._comp.get(key, default)))
            self._vars[key] = var
            tk.Entry(outer, textvariable=var, bg=CARD, fg=FG,
                     insertbackground=FG, font=f_s, relief="flat"
                     ).pack(fill="x", ipady=3)
            return var

        _text_field("Neuron Count",    "n_neurons",    "500")
        _text_field("Sensory Neurons", "n_sensory",    "64")
        _text_field("Motor Neurons",   "n_motor",      "8")
        _text_field("Sim Rate (Hz)",   "sim_rate_hz",  "10")

        tk.Frame(outer, bg=BORDER, height=1).pack(fill="x", pady=10)
        tk.Label(outer, text="OUTPUTS  (all optional — leave blank to skip)",
                 bg=PANEL, fg=ACCENT, font=f_s, anchor="w").pack(fill="x", pady=(0, 4))

        NONE_LABEL = "(none)"

        def _optional_servo_field(label_text, key):
            tk.Label(outer, text=label_text, bg=PANEL, fg=MUTED,
                     font=f_s, anchor="w").pack(fill="x", pady=(4, 1))
            current = self._comp.get(key, "") or NONE_LABEL
            var = tk.StringVar(value=current)
            options = [NONE_LABEL] + self._servo_names
            if self._servo_names:
                cb = ttk.Combobox(outer, textvariable=var, values=options,
                                  font=f_s, state="readonly")
                cb.pack(fill="x")
            else:
                tk.Label(outer, text="(no servos in this build yet)",
                         bg=PANEL, fg=MUTED, font=("Courier", 8),
                         anchor="w").pack(fill="x")
            self._vars[key] = var   # saved specially below — see _save()

        _optional_servo_field("Target Servo (pan)",  "target_servo_pan")
        _optional_servo_field("Target Servo (tilt)", "target_servo_tilt")
        _text_field("Target Motor (leave blank to skip)", "target_motor", "")

        _text_field("Target Keys (comma-separated, e.g. w,a,s,d)", "target_keys", "")
        tk.Label(outer,
                 text="Each key gets 'pressed' — triggering whatever action\n"
                      "is already bound to that key on the Control page —\n"
                      "whenever that motor neuron fires strongly. 1st key =\n"
                      "1st motor neuron, 2nd key = 2nd, and so on.",
                 bg=PANEL, fg=MUTED, font=("Courier", 8), justify="left",
                 wraplength=380).pack(anchor="w", pady=(1, 4))

        self._flybrain_auto_reward = tk.BooleanVar(
            value=self._comp.get("auto_reward_on_tracking", True))
        tk.Checkbutton(outer, variable=self._flybrain_auto_reward,
                       bg=PANEL, fg=FG, selectcolor=CARD, activebackground=PANEL,
                       font=f_s, text="Auto-reward when camera has a locked target"
                       ).pack(anchor="w", pady=(8, 0))

    def _save(self):
        t = self._comp["type"]
        # Save all entry vars
        for key, var in self._vars.items():
            val = var.get()
            try:    val = int(val)
            except ValueError:
                try:    val = float(val)
                except ValueError: pass
            self._comp[key] = val

        if t == "camera":
            self._comp["tracking_servos"]  = self._tracking_servos
            self._comp["tracking_enabled"] = self._track_enabled.get()
            self._comp["vision_enabled"]   = self._vision_enabled.get()
            self._comp["vision_auto"]      = self._vision_auto.get()
            self._comp["vision_prompt"]    = self._vision_prompt_txt.get("1.0", "end").strip()
            for key, var in self._local_vision_vars.items():
                self._comp[key] = var.get()
        elif t == "sound":
            self._comp["files"]  = self._sound_files
            self._comp["folder"] = self._folder_var.get()
        elif t == "flybrain":
            self._comp["auto_reward_on_tracking"] = self._flybrain_auto_reward.get()
            for key in ("target_servo_pan", "target_servo_tilt"):
                if self._comp.get(key) == "(none)":
                    self._comp[key] = ""
        # sensor: all fields already saved via self._vars loop above

        self.result = self._comp
        self.destroy()


# ═════════════════════════════════════════════════════════════════════════════
#  BUILDER PAGE
# ═════════════════════════════════════════════════════════════════════════════
class BuilderPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app = app
        self._build_ui()

    def _build_ui(self):
        f_title = tkfont.Font(family="Courier", size=(13 if COMPACT else 20), weight="bold")
        f_sub   = tkfont.Font(family="Courier", size=(9 if COMPACT else 11))
        f_med   = tkfont.Font(family="Courier", size=(9 if COMPACT else 12))
        f_small = tkfont.Font(family="Courier", size=(8 if COMPACT else 10))

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, height=(40 if COMPACT else 60))
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text=("// SRB" if COMPACT else "// SENTRY RIG BUILDER"),
                 bg=PANEL, fg=ACCENT, font=f_title).pack(side="left", padx=(8 if COMPACT else 20))

        pi_text  = "* LIVE" if ON_PI else "* SIM"
        pi_color = GREEN if ON_PI else ORANGE
        if not COMPACT:
            pi_text = "* PI LIVE" if ON_PI else "* SIMULATION"
        tk.Label(hdr, text=pi_text, bg=PANEL, fg=pi_color,
                 font=f_sub).pack(side="left", padx=(4 if COMPACT else 10))

        # Header buttons
        for txt, cmd, color in [
            ("SAVE",        self.app.save_build,             BORDER),
            ("LOAD",        self.app.load_build,             BORDER),
            ("LAUNCH",      self.app.launch,                  GREEN),
        ]:
            b = tk.Button(hdr, text=txt, bg=color, fg=FG,
                      activebackground=ACCENT, activeforeground=BG,
                      font=f_med, relief="flat", cursor="hand2", bd=0,
                      padx=(8 if COMPACT else 16), pady=(3 if COMPACT else 6), command=cmd)
            b.pack(side="right", padx=(3 if COMPACT else 6), pady=(6 if COMPACT else 10))
            _add_hover_outline(b)

        # ── Hazard stripe divider ────────────────────────────────────────
        _hazard_stripe(self, height=(2 if COMPACT else 3))

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=(6 if COMPACT else 16), pady=(6 if COMPACT else 12))

        # Left - add component panel
        left = tk.Frame(body, bg=PANEL, width=(140 if COMPACT else 200),
                        highlightthickness=1, highlightbackground=BORDER)
        left.pack(side="left", fill="y", padx=(0, 12))
        left.pack_propagate(False)

        tk.Label(left, text="ADD COMPONENT",
                 bg=PANEL, fg=MUTED, font=f_small).pack(pady=(16, 8), anchor="w", padx=14)

        # Bottom info block first, so it reserves its space before the
        # scrollable button area claims the rest.
        info_block = tk.Frame(left, bg=PANEL)
        info_block.pack(side="bottom", fill="x")
        tk.Frame(info_block, bg=BORDER, height=1).pack(fill="x", padx=10, pady=12)
        tk.Label(info_block, text="BUILD INFO", bg=PANEL, fg=MUTED,
                 font=f_small).pack()
        self.info_lbl = tk.Label(info_block, text="", bg=PANEL, fg=FG,
                                  font=f_small, justify="left")
        self.info_lbl.pack(padx=10, pady=6)

        # Scrollable button area (mouse-wheel enabled) ──────────────────────
        btn_outer = tk.Frame(left, bg=PANEL)
        btn_outer.pack(side="top", fill="both", expand=True)

        btn_canvas = tk.Canvas(btn_outer, bg=PANEL, highlightthickness=0)
        btn_scroll = tk.Scrollbar(btn_outer, orient="vertical",
                                   command=btn_canvas.yview)
        btn_canvas.configure(yscrollcommand=btn_scroll.set)
        btn_scroll.pack(side="right", fill="y")
        btn_canvas.pack(side="left", fill="both", expand=True)

        btn_frame = tk.Frame(btn_canvas, bg=PANEL)
        btn_window = btn_canvas.create_window((0, 0), window=btn_frame, anchor="nw")

        def _btn_frame_configure(e):
            btn_canvas.configure(scrollregion=btn_canvas.bbox("all"))
        btn_frame.bind("<Configure>", _btn_frame_configure)
        btn_canvas.bind("<Configure>",
                         lambda e: btn_canvas.itemconfig(btn_window, width=e.width))

        def _on_mousewheel(event):
            delta = -1 if event.num == 4 else 1 if event.num == 5 else -event.delta // 120
            btn_canvas.yview_scroll(delta, "units")
        btn_canvas.bind("<Enter>", lambda e: (
            btn_canvas.bind_all("<MouseWheel>", _on_mousewheel),
            btn_canvas.bind_all("<Button-4>", _on_mousewheel),
            btn_canvas.bind_all("<Button-5>", _on_mousewheel)))
        btn_canvas.bind("<Leave>", lambda e: (
            btn_canvas.unbind_all("<MouseWheel>"),
            btn_canvas.unbind_all("<Button-4>"),
            btn_canvas.unbind_all("<Button-5>")))

        # Distinct accent color per category — same black HUD panel for all,
        # differentiated by a slim colored strip (like HL's ammo/health icons)
        for label, factory, color in [
            ("SERVO",       default_servo,       "#d9922e"),
            ("MOTOR",       default_motor,       "#4f8fae"),
            ("LASER",       default_laser,       "#ef4444"),
            ("CAMERA",      default_camera,      "#5fd068"),
            ("SOUND",       default_sound,       "#f5c542"),
            ("ULTRASONIC",  default_ultrasonic,  "#2fb8a6"),
            ("MOTION",      default_motion,      "#4fc3d9"),
            ("TEMPERATURE", default_temperature, "#ff7a3d"),
            ("MICROPHONE",  default_microphone,  "#e8935a"),
            ("FLOCK DETECT", default_flock,       "#ff5a1f"),
            ("FLY BRAIN",    default_flybrain,    "#9b59ff"),
            ("POWER MONITOR", default_power,      "#ffd23f"),
        ]:
            row = tk.Frame(btn_frame, bg=CARD, bd=0,
                            highlightthickness=1, highlightbackground=BORDER)
            row.pack(pady=3, padx=12, fill="x")

            strip = tk.Frame(row, bg=color, width=4)
            strip.pack(side="left", fill="y")

            lbl = tk.Label(row, text=f"+  {label}", bg=CARD, fg=FG,
                            font=f_med, anchor="w", padx=10, pady=9,
                            cursor="hand2")
            lbl.pack(side="left", fill="x", expand=True)

            def _add_cmd(f=factory):
                self._add(f())

            def _on_enter(e, row=row, lbl=lbl, color=color):
                row.configure(highlightbackground=color)
                lbl.configure(fg=color)

            def _on_leave(e, row=row, lbl=lbl):
                row.configure(highlightbackground=BORDER)
                lbl.configure(fg=FG)

            for w in (row, lbl, strip):
                w.bind("<Button-1>", lambda e, f=_add_cmd: f())
                w.bind("<Enter>", _on_enter)
                w.bind("<Leave>", _on_leave)

        # Right - component list
        right = tk.Frame(body, bg=BG)
        right.pack(side="right", fill="both", expand=True)

        tk.Label(right, text="COMPONENTS", bg=BG, fg=MUTED,
                 font=f_small).pack(anchor="w", pady=(0, 6))

        # Scrollable canvas
        scroll_frame = tk.Frame(right, bg=BG)
        scroll_frame.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(scroll_frame, bg=BG,
                                highlightthickness=0)
        scrollbar = tk.Scrollbar(scroll_frame, orient="vertical",
                                 command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.list_frame = tk.Frame(self.canvas, bg=BG)
        self.canvas_window = self.canvas.create_window(
            (0, 0), window=self.list_frame, anchor="nw")

        self.list_frame.bind("<Configure>", lambda e: self.canvas.configure(
            scrollregion=self.canvas.bbox("all")))
        self.canvas.bind("<Configure>", lambda e: self.canvas.itemconfig(
            self.canvas_window, width=e.width))

        self.refresh()

    def _add(self, comp):
        self.app.components.append(comp)
        self.refresh()

    def refresh(self):
        for w in self.list_frame.winfo_children():
            w.destroy()

        f_name  = tkfont.Font(family="Courier", size=12, weight="bold")
        f_small = tkfont.Font(family="Courier", size=10)

        type_colors = {"servo": SERVO_C, "motor": MOTOR_C,
                       "laser": LASER_C, "camera": CAM_C,
                       "sound": SOUND_C, "sensor": SENSOR_C,
                       "flybrain": "#9b59ff"}
        type_icons  = {"servo": "[S]", "motor": "[M]", "laser": "[L]",
                       "camera": "[C]", "sound": "[A]", "sensor": "[~]",
                       "flybrain": "[B]"}

        for i, comp in enumerate(self.app.components):
            color = type_colors.get(comp["type"], ACCENT)
            card = tk.Frame(self.list_frame, bg=CARD,
                            highlightthickness=1, highlightbackground=color)
            card.pack(fill="x", pady=5, padx=4)

            # Color stripe
            tk.Frame(card, bg=color, width=5).pack(side="left", fill="y")

            info = tk.Frame(card, bg=CARD)
            info.pack(side="left", fill="both", expand=True, padx=10, pady=8)

            icon = type_icons.get(comp["type"], "?")
            tk.Label(info,
                     text=f"{icon}  {comp['name']}",
                     bg=CARD, fg=FG, font=f_name).pack(anchor="w")

            # Summary line
            if comp["type"] == "servo":
                detail = f"pin={comp['pin']}  range={comp['min_deg']}°–{comp['max_deg']}°  step={comp['step']}°"
            elif comp["type"] == "motor":
                detail = f"fwd={comp['pin_fwd']}  bwd={comp['pin_bwd']}"
            elif comp["type"] == "laser":
                detail = f"pin={comp['pin']}"
            elif comp["type"] == "camera":
                servos = comp.get("tracking_servos", [])
                enabled = comp.get("tracking_enabled", False)
                vision  = comp.get("vision_enabled", False)
                detail = f"index={comp['index']}  tracking={'ON' if enabled else 'OFF'}  vision={'ON' if vision else 'OFF'}"
            elif comp["type"] == "sound":
                files = comp.get("files", [])
                detail = f"folder={comp.get('folder','?')}  {len(files)} file(s)  vol={comp.get('volume',80)}%"
            elif comp["type"] == "sensor":
                st = comp.get("sensor_type", "?")
                if st == "ultrasonic":
                    detail = f"trig={comp.get('pin_trig')}  echo={comp.get('pin_echo')}  alert<{comp.get('alert_distance')}cm"
                elif st == "motion":
                    detail = f"pin={comp.get('pin')}  alert={'ON' if comp.get('alert_enabled') else 'OFF'}"
                elif st == "temperature":
                    detail = f"pin={comp.get('pin')}  model={comp.get('model')}  alert>{comp.get('alert_temp')}C"
                elif st == "microphone":
                    detail = f"device={comp.get('device_index')}  threshold={comp.get('threshold')}  voice={'ON' if comp.get('voice_commands') else 'OFF'}"
                elif st == "flock":
                    detail = f"port={comp.get('serial_port')}  baud={comp.get('baud_rate')}  alert={'ON' if comp.get('alert_enabled') else 'OFF'}"
                elif st == "power":
                    ina = comp.get("enable_ina219", False)
                    detail = (f"pi_health={'ON' if comp.get('monitor_pi', True) else 'OFF'}  "
                             f"ina219={'ON @'+comp.get('ina219_addr','0x40') if ina else 'OFF'}  "
                             f"alert={'ON' if comp.get('alert_on_undervoltage', True) else 'OFF'}")
                else:
                    detail = st
            else:
                detail = ""
            tk.Label(info, text=detail, bg=CARD, fg=MUTED,
                     font=f_small).pack(anchor="w")

            # Buttons
            btns = tk.Frame(card, bg=CARD)
            btns.pack(side="right", padx=10)

            tk.Button(btns, text="EDIT", bg=BORDER, fg=FG,
                      activebackground=ACCENT, activeforeground=BG,
                      font=f_small, relief="flat", cursor="hand2",
                      padx=8, pady=4,
                      command=lambda idx=i: self._edit(idx)).pack(pady=2)
            tk.Button(btns, text="DEL", bg=RED, fg="#fff",
                      activebackground="#ff6b6b", activeforeground="#fff",
                      font=f_small, relief="flat", cursor="hand2",
                      padx=8, pady=4,
                      command=lambda idx=i: self._delete(idx)).pack(pady=2)

        # Update info label
        counts = {}
        for c in self.app.components:
            counts[c["type"]] = counts.get(c["type"], 0) + 1
        lines = [f"{v}x {k}" for k, v in counts.items()] or ["No components"]
        self.info_lbl.config(text="\n".join(lines))

    def _edit(self, idx):
        servo_names = [c["name"] for c in self.app.components
                       if c["type"] == "servo"]
        dlg = ComponentEditor(self, self.app.components[idx], servo_names)
        self.wait_window(dlg)
        if dlg.result:
            self.app.components[idx] = dlg.result
            self.refresh()

    def _delete(self, idx):
        self.app.components.pop(idx)
        self.refresh()


# ═════════════════════════════════════════════════════════════════════════════
#  CONTROL PAGE
# ═════════════════════════════════════════════════════════════════════════════
class ControlPage(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent, bg=BG)
        self.app        = app
        self._running   = True
        self._tracking  = False
        self._servo_angles = {}   # name -> float
        self._laser_states = {}   # name -> bool
        self._motor_speeds = {}   # name -> int
        self._cam_thread = None
        self._latest_frame = None
        self._face_offset  = 0
        self._face_found   = False
        self._frame_lock   = threading.Lock()
        self._track_servo_comps = []   # list of servo component dicts

        self._build_ui()
        self._setup_keyboard()
        self._start_camera()
        self._ui_loop()

    def _build_ui(self):
        f_title = tkfont.Font(family="Courier", size=(12 if COMPACT else 18), weight="bold")
        f_med   = tkfont.Font(family="Courier", size=(9 if COMPACT else 12))
        f_small = tkfont.Font(family="Courier", size=(8 if COMPACT else 10))
        f_big   = tkfont.Font(family="Courier", size=(14 if COMPACT else 20), weight="bold")

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=PANEL, height=(38 if COMPACT else 55))
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text=("# CONTROL" if COMPACT else "# SENTRY CONTROL"),
                 bg=PANEL, fg=ACCENT, font=f_title).pack(side="left", padx=(8 if COMPACT else 20))

        tk.Button(hdr, text=("<" if COMPACT else "< BUILDER"), bg=BORDER, fg=MUTED,
                  activebackground=ACCENT, activeforeground=BG,
                  font=f_small, relief="flat", cursor="hand2",
                  padx=(6 if COMPACT else 12), pady=(3 if COMPACT else 6),
                  command=self._back).pack(side="right", padx=(4 if COMPACT else 12), pady=(6 if COMPACT else 10))

        tk.Button(hdr, text="[AI]", bg=SERVO_C, fg="#fff",
                  activebackground=ACCENT,
                  font=f_small, relief="flat", cursor="hand2",
                  padx=(6 if COMPACT else 12), pady=(3 if COMPACT else 6),
                  command=self._toggle_ai).pack(side="right", pady=(6 if COMPACT else 10))

        self._kb_active = tk.BooleanVar(value=True)
        self._kb_btn = tk.Button(hdr,
                  text="[KB]",
                  bg=GREEN, fg="#fff",
                  font=f_small, relief="flat", cursor="hand2",
                  padx=(6 if COMPACT else 12), pady=(3 if COMPACT else 6),
                  command=self._toggle_keyboard)
        self._kb_btn.pack(side="right", pady=(6 if COMPACT else 10), padx=4)

        # Active key indicator
        self._key_indicator = tk.StringVar(value="")
        tk.Label(hdr, textvariable=self._key_indicator,
                 bg=PANEL, fg=SENSOR_C, font=f_small).pack(side="right", padx=8)

        # ── Hazard stripe divider ────────────────────────────────────────
        _hazard_stripe(self, height=(2 if COMPACT else 3))

        # ── Body - horizontal layout ───────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True)

        # ── LEFT: Camera panel ────────────────────────────────────────────
        self.cam_comps  = [c for c in self.app.components if c["type"] == "camera"]
        self.cam_canvas = None
        self._vision_auto_on = tk.BooleanVar(value=False)
        self._vision_busy    = False

        if self.cam_comps:
            _try_load_cv2()  # safe deferred load
            print(f"[CAM] cam_comps={len(self.cam_comps)}  HAS_CV2={HAS_CV2}  "
                  f"HAS_PIL={HAS_PIL}  HAS_FACE_CASCADE={HAS_FACE_CASCADE}")
            cam_comp  = self.cam_comps[0]
            cam_w = 260 if COMPACT else 340
            cam_panel = tk.Frame(body, bg=PANEL, width=cam_w,
                                 highlightthickness=1, highlightbackground=CAM_C)
            cam_panel.pack(side="left", fill="y", padx=(10, 6), pady=10)
            cam_panel.pack_propagate(False)

            tk.Label(cam_panel, text=f"[C] {cam_comp['name']}",
                     bg=PANEL, fg=CAM_C, font=f_med).pack(pady=(8, 4))

            cam_res = (240, 180) if COMPACT else (320, 240)
            self.cam_canvas = tk.Canvas(cam_panel, width=cam_res[0], height=cam_res[1],
                                        bg="#111", highlightthickness=0)
            self.cam_canvas.pack()
            cx, cy = cam_res[0] // 2, cam_res[1] // 2

            if not HAS_CV2:
                self.cam_canvas.create_text(cx, cy,
                    text="opencv not installed\npip install opencv-python-headless",
                    fill=RED, font=("Courier", 10), justify="center")
            elif not HAS_PIL:
                err_txt = (PIL_IMPORT_ERROR or "unknown error")[:60]
                hint = ("sudo apt install python3-pil.imagetk"
                        if "ImageTk" in (PIL_IMPORT_ERROR or "")
                        else "pip install pillow --break-system-packages")
                self.cam_canvas.create_text(cx, cy,
                    text=f"Pillow import failed:\n{err_txt}\n\nfix: {hint}",
                    fill=RED, font=("Courier", 9), justify="center", width=cam_res[0]-20)
            else:
                self.cam_canvas.create_text(cx, cy, text="Starting camera...",
                                            fill=MUTED, font=("Courier", 10))

            self.track_status      = tk.StringVar(value="* TRACKING  OFF")
            self._track_status_lbl = tk.Label(cam_panel,
                                               textvariable=self.track_status,
                                               bg=PANEL, fg=MUTED, font=f_small)
            self._track_status_lbl.pack(pady=(4, 0))

            track_names = cam_comp.get("tracking_servos", [])
            servo_map   = {c["name"]: c for c in self.app.components
                           if c["type"] == "servo"}
            self._track_servo_comps = [servo_map[n] for n in track_names
                                       if n in servo_map]
            self._tracking = cam_comp.get("tracking_enabled", False)

            self.btn_track = tk.Button(
                cam_panel,
                text="#  STOP TRACKING" if self._tracking else "#  START TRACKING",
                bg=RED if self._tracking else GREEN,
                fg="#fff", bd=0, font=f_small, relief="flat", cursor="hand2",
                width=20, pady=5, command=self._toggle_tracking)
            self.btn_track.pack(pady=6, padx=8)
            _add_hover_outline(self.btn_track)

            if HAS_CV2 and CV2_BROKEN:
                tk.Label(cam_panel,
                         text="⚠ cv2 install is broken (missing\n"
                              "CascadeClassifier) — not a file or\n"
                              "network issue. Likely both opencv-python\n"
                              "and opencv-python-headless are installed\n"
                              "at once. See terminal for the fix.",
                         bg=PANEL, fg=RED, font=("Courier", 8),
                         justify="center").pack(pady=(0, 6))
            elif HAS_CV2 and not HAS_FACE_CASCADE:
                tk.Label(cam_panel,
                         text="⚠ face cascade unavailable — tracking/face\n"
                              "features disabled. Auto-download failed;\n"
                              "check internet connection and restart,\n"
                              "or see terminal for details.",
                         bg=PANEL, fg=RED, font=("Courier", 8),
                         justify="center").pack(pady=(0, 6))

            # Vision section
            if cam_comp.get("vision_enabled", False):
                tk.Frame(cam_panel, bg=BORDER, height=1).pack(fill="x", padx=8, pady=2)
                tk.Label(cam_panel, text="🧠 AI VISION",
                         bg=PANEL, fg=ACCENT, font=f_small).pack()

                self._vision_result = tk.StringVar(value="Press Analyze to start")
                tk.Label(cam_panel, textvariable=self._vision_result,
                         bg=PANEL, fg=FG, font=f_small,
                         wraplength=300, justify="left").pack(padx=8, pady=2)

                vis_row = tk.Frame(cam_panel, bg=PANEL)
                vis_row.pack(pady=4)

                tk.Button(vis_row, text="[?] ANALYZE",
                          bg=CAM_C, fg="#fff",
                          font=f_small, relief="flat", cursor="hand2",
                          padx=8, pady=4,
                          command=self._vision_analyze_once).pack(side="left", padx=4)

                self._vision_auto_on = tk.BooleanVar(
                    value=cam_comp.get("vision_auto", False))
                self._vis_auto_btn = tk.Button(
                    vis_row,
                    text="AUTO ON" if self._vision_auto_on.get() else "AUTO OFF",
                    bg=GREEN if self._vision_auto_on.get() else BORDER,
                    fg="#fff", font=f_small, relief="flat", cursor="hand2",
                    padx=8, pady=4, command=self._toggle_vision_auto)
                self._vis_auto_btn.pack(side="left", padx=4)

                self._vision_interval = float(cam_comp.get("vision_interval", 3.0))
                self._vision_model    = cam_comp.get("vision_model", "moondream")
                self._vision_prompt   = cam_comp.get("vision_prompt", "Describe what you see.")

                if self._vision_auto_on.get():
                    self._schedule_vision()

            # ── Local vision toolkit panel ───────────────────────────────
            local_flags = ("detect_faces", "recognize_faces", "detect_objects",
                           "classify_scene", "detect_custom_images", "detect_pose")
            if any(cam_comp.get(k, False) for k in local_flags):
                tk.Frame(cam_panel, bg=BORDER, height=1).pack(fill="x", padx=8, pady=2)
                tk.Label(cam_panel, text="[V] LOCAL VISION",
                         bg=PANEL, fg=CAM_C, font=f_small).pack()

                self._vision_log = tk.Text(cam_panel, height=6, bg=BG, fg=FG,
                                           insertbackground=FG, font=f_small,
                                           relief="flat", wrap="word")
                self._vision_log.pack(fill="x", padx=8, pady=(2, 4))
                self._vision_log.tag_config("hit", foreground=CAM_C)
                self._vision_log.configure(state="disabled")

                if cam_comp.get("recognize_faces", False):
                    tk.Button(cam_panel, text="+ ADD KNOWN FACE",
                              bg=BORDER, fg=FG, font=f_small, relief="flat",
                              cursor="hand2", pady=4,
                              command=self._add_known_face).pack(fill="x", padx=8, pady=(0, 6))

                self._start_local_vision(cam_comp)

        # ── MIDDLE: Component controls ────────────────────────────────────
        mid = tk.Frame(body, bg=BG)
        mid.pack(side="left", fill="both", expand=True, padx=6, pady=10)

        rcanvas = tk.Canvas(mid, bg=BG, highlightthickness=0)
        rsb     = tk.Scrollbar(mid, orient="vertical", command=rcanvas.yview)
        rcanvas.configure(yscrollcommand=rsb.set)
        rsb.pack(side="right", fill="y")
        rcanvas.pack(side="left", fill="both", expand=True)

        self.ctrl_frame = tk.Frame(rcanvas, bg=BG)
        rcanvas_win = rcanvas.create_window((0, 0), window=self.ctrl_frame, anchor="nw")
        self.ctrl_frame.bind("<Configure>", lambda e: rcanvas.configure(
            scrollregion=rcanvas.bbox("all")))
        rcanvas.bind("<Configure>", lambda e: rcanvas.itemconfig(
            rcanvas_win, width=e.width))

        for comp in self.app.components:
            if comp["type"] == "camera":
                continue
            try:
                self._make_control(comp, f_med, f_small, f_big)
            except Exception as e:
                print(f"[ERROR] {comp.get('name')}: {e}")

        # ── RIGHT: AI panel (hidden until toggled) ────────────────────────
        self._ai_panel = AIChatPanel(body, self)

    def _make_control(self, comp, f_med, f_small, f_big):
        t = comp["type"]
        type_colors = {"servo": SERVO_C, "motor": MOTOR_C, "laser": LASER_C,
                       "camera": CAM_C, "sound": SOUND_C, "sensor": SENSOR_C,
                       "flybrain": "#9b59ff"}
        type_icons  = {"servo": "[S]", "motor": "[M]", "laser": "[L]",
                       "camera": "[C]", "sound": "[A]", "sensor": "[~]",
                       "flybrain": "[B]"}
        color = type_colors.get(t, ACCENT)
        icon  = type_icons.get(t, "?")

        card = tk.Frame(self.ctrl_frame, bg=CARD,
                        highlightthickness=1, highlightbackground=color)
        card.pack(fill="x", pady=6, padx=4)
        tk.Frame(card, bg=color, width=5).pack(side="left", fill="y")

        inner = tk.Frame(card, bg=CARD)
        inner.pack(fill="both", expand=True, padx=12, pady=10)

        tk.Label(inner, text=f"{icon}  {comp['name']}",
                 bg=CARD, fg=FG, font=f_med).pack(anchor="w")

        if t == "servo":
            self._servo_angles[comp["name"]] = 90.0
            # Don't call servo_set here - let the user move it first

            angle_var = tk.DoubleVar(value=90.0)

            angle_lbl = tk.Label(inner, text="90.0°",
                                  bg=CARD, fg=FG, font=f_big)
            angle_lbl.pack()

            # Arc canvas
            arc_cv = tk.Canvas(inner, width=200, height=100,
                               bg=CARD, highlightthickness=0)
            arc_cv.pack()

            def draw_arc(a, cv=arc_cv, c=color):
                cv.delete("all")
                cx, cy, r = 100, 90, 70
                cv.create_arc(cx-r, cy-r, cx+r, cy+r,
                              start=0, extent=180,
                              style="arc", outline=BORDER, width=8)
                cv.create_arc(cx-r, cy-r, cx+r, cy+r,
                              start=0, extent=a,
                              style="arc", outline=c, width=8)
                rad = math.radians(180 - a)
                nx, ny = cx + r*math.cos(rad), cy - r*math.sin(rad)
                cv.create_line(cx, cy, nx, ny, fill=ACCENT, width=3)
                cv.create_oval(cx-4, cy-4, cx+4, cy+4, fill=ACCENT, outline="")

            draw_arc(90.0)

            def move(direction, name=comp["name"], c=comp,
                     lbl=angle_lbl, draw=draw_arc):
                cur = self._servo_angles[name]
                new = max(c["min_deg"], min(c["max_deg"],
                          cur + direction * c["step"]))
                self._servo_angles[name] = new
                servo_set(c["pin"], new, c["pulse_min"], c["pulse_max"])
                lbl.config(text=f"{new:.1f}°")
                draw(new)

            def reset_servo(name=comp["name"], c=comp,
                            lbl=angle_lbl, draw=draw_arc):
                self._servo_angles[name] = 90.0
                servo_set(c["pin"], 90.0, c["pulse_min"], c["pulse_max"])
                lbl.config(text="90.0°")
                draw(90.0)

            btn_row = tk.Frame(inner, bg=CARD)
            btn_row.pack(pady=6)

            bl = tk.Button(btn_row, text="< LEFT", bg=SERVO_C, fg="#fff", bd=0,
                           font=f_small, relief="flat", cursor="hand2",
                           width=9, height=2)
            bl.grid(row=0, column=0, padx=6)
            _add_hover_outline(bl)

            br = tk.Button(btn_row, text="RIGHT >", bg=SERVO_C, fg="#fff", bd=0,
                           font=f_small, relief="flat", cursor="hand2",
                           width=9, height=2)
            br.grid(row=0, column=1, padx=6)
            _add_hover_outline(br)

            # hold-to-move
            _hold = {"job": None, "dir": 0}

            def start_hold(d, _h=_hold, mv=move):
                _h["dir"] = d
                mv(d)
                _schedule(_h, mv)

            def _schedule(_h, mv):
                _h["job"] = self.after(60, lambda: _step(_h, mv))

            def _step(_h, mv):
                if _h["dir"] != 0:
                    mv(_h["dir"])
                    _schedule(_h, mv)

            def stop_hold(_h=_hold):
                _h["dir"] = 0
                if _h["job"]:
                    self.after_cancel(_h["job"])
                    _h["job"] = None

            bl.bind("<ButtonPress-1>",   lambda e, d=-1: start_hold(d))
            bl.bind("<ButtonRelease-1>", lambda e: stop_hold())
            br.bind("<ButtonPress-1>",   lambda e, d=+1: start_hold(d))
            br.bind("<ButtonRelease-1>", lambda e: stop_hold())

            reset_btn = tk.Button(inner, text="RESET 90°", bg=BORDER, fg=MUTED, bd=0,
                      font=f_small, relief="flat", cursor="hand2",
                      command=reset_servo)
            reset_btn.pack()
            _add_hover_outline(reset_btn)

            # Keybind display
            kb_frame = tk.Frame(inner, bg=CARD)
            kb_frame.pack(pady=(8,0), fill="x")
            tk.Label(kb_frame, text="KEYS:", bg=CARD, fg=MUTED, font=f_small).pack(side="left")
            for kb_label, kb_key in [("Left", "key_left"), ("Right", "key_right"), ("Reset", "key_reset")]:
                tk.Label(kb_frame, text=f"{kb_label}:", bg=CARD, fg=MUTED, font=f_small).pack(side="left", padx=(8,2))
                kv = tk.StringVar(value=comp.get(kb_key, ""))
                ke = tk.Entry(kb_frame, textvariable=kv, width=3, bg=BORDER, fg=ACCENT,
                              insertbackground=ACCENT, font=f_small, relief="flat")
                ke.pack(side="left")
                def update_key(e, c=comp, k=kb_key, v=kv):
                    c[k] = v.get().strip().lower()
                    self._setup_keyboard()
                ke.bind("<FocusOut>", update_key)
                ke.bind("<Return>",   update_key)

        elif t == "motor":
            self._motor_speeds[comp["name"]] = 0

            speed_var = tk.IntVar(value=0)
            spd_lbl = tk.Label(inner, text="SPEED: 0%",
                                bg=CARD, fg=FG, font=f_big)
            spd_lbl.pack()

            def _set_motor_speed(v, c=comp, lbl=spd_lbl):
                # UI updates first, always — a GPIO/hardware error must never
                # block the slider/label from reflecting what was requested.
                v = int(v)
                lbl.config(text=f"SPEED: {v}%")
                try:
                    motor_set(c["pin_fwd"], c["pin_bwd"], v)
                except Exception as e:
                    print(f"[MOTOR] error setting speed on {c['name']}: {e}")

            slider = tk.Scale(inner, from_=-100, to=100,
                              orient="horizontal", variable=speed_var,
                              bg=CARD, fg=FG, troughcolor=BORDER,
                              highlightthickness=0, sliderrelief="flat",
                              length=260,
                              command=_set_motor_speed)
            slider.pack(pady=6)

            btn_row = tk.Frame(inner, bg=CARD)
            btn_row.pack()
            for txt, val in [("⏪ REV", -80), ("⏹ STOP", 0), ("⏩ FWD", 80)]:
                mb = tk.Button(btn_row, text=txt, bg=MOTOR_C, fg="#fff", bd=0,
                          font=f_small, relief="flat", cursor="hand2",
                          width=8, pady=6,
                          command=lambda v=val, s=slider: (
                              s.set(v), _set_motor_speed(v)
                          ))
                mb.pack(side="left", padx=6)
                _add_hover_outline(mb)

            # Motor keybind display
            kb_frame = tk.Frame(inner, bg=CARD)
            kb_frame.pack(pady=(8, 0), fill="x")
            tk.Label(kb_frame, text="KEYS:", bg=CARD, fg=MUTED, font=f_small).pack(side="left")
            for kb_label, kb_key in [("Fwd", "key_fwd"), ("Bwd", "key_bwd"), ("Stop", "key_stop")]:
                tk.Label(kb_frame, text=f"{kb_label}:", bg=CARD, fg=MUTED, font=f_small).pack(side="left", padx=(8, 2))
                kv = tk.StringVar(value=comp.get(kb_key, ""))
                ke = tk.Entry(kb_frame, textvariable=kv, width=3, bg=BORDER, fg=ACCENT,
                              insertbackground=ACCENT, font=f_small, relief="flat")
                ke.pack(side="left")
                def update_motor_key(e, c=comp, k=kb_key, v=kv):
                    c[k] = v.get().strip().lower()
                    self._setup_keyboard()
                ke.bind("<FocusOut>", update_motor_key)
                ke.bind("<Return>",   update_motor_key)

        elif t == "laser":
            self._laser_states[comp["name"]] = False
            state_var = tk.StringVar(value="OFF")

            lbl = tk.Label(inner, text="◉  OFF",
                           bg=CARD, fg=MUTED, font=f_big)
            lbl.pack(pady=4)

            def toggle(c=comp, l=lbl):
                cur = self._laser_states[c["name"]]
                new = not cur
                self._laser_states[c["name"]] = new
                laser_set(c["pin"], new)
                if new:
                    l.config(text="◉  ON", fg=RED)
                    fire_btn.config(bg=RED, text="[L] LASER ON")
                else:
                    l.config(text="◉  OFF", fg=MUTED)
                    fire_btn.config(bg=BORDER, text="[L] FIRE LASER")

            fire_btn = tk.Button(inner, text="[L] FIRE LASER",
                                  bg=BORDER, fg=FG, bd=0,
                                  activebackground=RED,
                                  font=f_med, relief="flat", cursor="hand2",
                                  width=16, height=2,
                                  command=toggle)
            fire_btn.pack(pady=6)
            _add_hover_outline(fire_btn)

        elif t == "sound":
            if not HAS_PYGAME:
                tk.Label(inner, text="⚠ pygame not installed\npip install pygame --break-system-packages",
                         bg=CARD, fg=ORANGE, font=f_small, justify="left").pack()
                return

            files = comp.get("files", [])
            vol   = comp.get("volume", 80)
            pygame.mixer.music.set_volume(vol / 100.0)

            # Volume slider
            vol_row = tk.Frame(inner, bg=CARD)
            vol_row.pack(fill="x", pady=4)
            tk.Label(vol_row, text="Vol:", bg=CARD, fg=MUTED, font=f_small).pack(side="left")
            vol_var = tk.IntVar(value=vol)
            tk.Scale(vol_row, from_=0, to=100, orient="horizontal",
                     variable=vol_var, bg=CARD, fg=FG,
                     troughcolor=BORDER, highlightthickness=0,
                     sliderrelief="flat", length=160,
                     command=lambda v: pygame.mixer.music.set_volume(int(v)/100.0)
                     ).pack(side="left", padx=6)

            # Stop button
            sound_stop_btn = tk.Button(vol_row, text="⏹ STOP", bg=BORDER, fg=MUTED, bd=0,
                      font=f_small, relief="flat", cursor="hand2",
                      command=pygame.mixer.music.stop)
            sound_stop_btn.pack(side="right", padx=4)
            _add_hover_outline(sound_stop_btn)

            if not files:
                tk.Label(inner, text="No files added - edit component to add MP3s",
                         bg=CARD, fg=MUTED, font=f_small).pack(pady=6)
                return

            # Playlist
            tk.Label(inner, text="PLAYLIST", bg=CARD, fg=MUTED,
                     font=f_small).pack(anchor="w", pady=(6, 2))

            playlist_frame = tk.Frame(inner, bg=CARD)
            playlist_frame.pack(fill="x")

            self._now_playing = tk.StringVar(value="")
            tk.Label(inner, textvariable=self._now_playing,
                     bg=CARD, fg=SOUND_C, font=f_small).pack(anchor="w")

            for path in files:
                fname = os.path.basename(path)
                row   = tk.Frame(playlist_frame, bg=CARD)
                row.pack(fill="x", pady=2)

                def play(p=path, n=fname, np=self._now_playing):
                    try:
                        pygame.mixer.music.load(p)
                        pygame.mixer.music.play()
                        np.set(f"> {n}")
                    except Exception as e:
                        np.set(f"Error: {e}")

                tk.Button(row, text=">", bg=SOUND_C, fg="#000",
                          font=f_small, relief="flat", cursor="hand2",
                          width=3, command=play).pack(side="left", padx=4)
                tk.Label(row, text=fname, bg=CARD, fg=FG,
                         font=f_small, anchor="w").pack(side="left")

        elif t == "flybrain":
            # ── Fly Brain Controller: local spiking neural network ───────
            status_var  = tk.StringVar(value="○ initializing...")
            motor_var   = tk.StringVar(value="motor rates: --")

            top_row = tk.Frame(inner, bg=CARD)
            top_row.pack(fill="x", pady=(2, 6))
            tk.Label(top_row, textvariable=status_var, bg=CARD, fg=MUTED,
                     font=f_small).pack(side="left")

            # Live neuron activity graph — clustered layout (visual / central
            # / motor, same grouping the reference dashboards use), dots
            # brighten when that neuron actually spikes this frame.
            graph_canvas = tk.Canvas(inner, width=260, height=170,
                                     bg="#050508", highlightthickness=0)
            graph_canvas.pack(pady=(0, 4))

            motor_canvas = tk.Canvas(inner, height=70, bg=CARD, highlightthickness=0)
            motor_canvas.pack(fill="x", padx=2, pady=(0, 4))

            tk.Label(inner, textvariable=motor_var, bg=CARD, fg="#9b59ff",
                     font=("Courier", 7)).pack(pady=(0, 4))

            log = tk.Text(inner, height=6, bg=BG, fg=FG, insertbackground=FG,
                          font=f_small, relief="flat", wrap="word")
            log.pack(fill="x", padx=2, pady=(0, 4))
            log.tag_config("hit", foreground="#9b59ff")
            log.configure(state="disabled")

            tk.Button(inner, text="⚡ REWARD (reinforce current activity)",
                      bg="#9b59ff", fg="#fff", font=f_small, relief="flat",
                      cursor="hand2", pady=5,
                      command=lambda c=comp: self._flybrain_reward(c)
                      ).pack(fill="x", padx=2, pady=(0, 6))

            key = f"flybrain_{comp['name']}"
            if not hasattr(self, "_flybrain_state"):
                self._flybrain_state = {}
            self._flybrain_state[key] = {
                "comp": comp, "status_var": status_var,
                "motor_var": motor_var, "log": log, "sim": None,
                "graph_canvas": graph_canvas, "motor_canvas": motor_canvas,
                "dot_ids": [], "positions": [], "roles": [],
            }
            self._start_flybrain(key)
            return

        elif t == "sensor" and comp.get("sensor_type") == "power":
            # ── Power Monitor: Pi health (always) + optional INA219 rail ────
            status_var = tk.StringVar(value="○ reading...")
            volt_var   = tk.StringVar(value="core: -- V")
            temp_var   = tk.StringVar(value="temp: -- °C")
            rail_var   = tk.StringVar(value="")

            tk.Label(inner, textvariable=status_var, bg=CARD, fg=MUTED,
                     font=f_small).pack(pady=(2, 4))
            tk.Label(inner, textvariable=volt_var, bg=CARD, fg=FG,
                     font=f_med).pack()
            tk.Label(inner, textvariable=temp_var, bg=CARD, fg=FG,
                     font=f_med).pack(pady=(0, 4))

            flags_log = tk.Text(inner, height=4, bg=BG, fg=FG,
                                insertbackground=FG, font=f_small,
                                relief="flat", wrap="word")
            flags_log.pack(fill="x", padx=2, pady=(0, 4))
            flags_log.tag_config("warn", foreground=RED)
            flags_log.configure(state="disabled")

            if comp.get("enable_ina219", False):
                tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=8, pady=2)
                tk.Label(inner, textvariable=rail_var, bg=CARD, fg="#ffd23f",
                         font=f_small).pack(pady=(2, 4))

            key = f"sensor_{comp['name']}"
            if not hasattr(self, "_power_state"):
                self._power_state = {}
            self._power_state[key] = {
                "comp": comp, "status_var": status_var, "volt_var": volt_var,
                "temp_var": temp_var, "rail_var": rail_var, "flags_log": flags_log,
                "last_flags": set(),
            }
            self._start_power_monitor(key)
            return

        elif t == "sensor" and comp.get("sensor_type") == "flock":
            # ── Flock Detector: live JSON detection feed over USB serial ────
            # Compatible with colonelpanichacks/flock-you and oui-spy firmware
            # (github.com/colonelpanichacks/flock-you) — reads newline-delimited
            # JSON detections like {"ssid":..,"mac_address":..,"rssi":..,...}
            status_var = tk.StringVar(value="○ not connected")
            count_var  = tk.StringVar(value="0 detections")

            top_row = tk.Frame(inner, bg=CARD)
            top_row.pack(fill="x", pady=(2, 6))
            tk.Label(top_row, textvariable=status_var, bg=CARD, fg=MUTED,
                     font=f_small).pack(side="left")
            tk.Label(top_row, textvariable=count_var, bg=CARD, fg=ACCENT,
                     font=f_small).pack(side="right")

            log = tk.Text(inner, height=7, bg=BG, fg=FG, insertbackground=FG,
                          font=f_small, relief="flat", wrap="none")
            log.pack(fill="x", padx=2, pady=(0, 4))
            log.tag_config("hit", foreground=RED)
            log.configure(state="disabled")

            key = f"sensor_{comp['name']}"
            if not hasattr(self, "_flock_state"):
                self._flock_state = {}
            self._flock_state[key] = {
                "comp": comp, "status_var": status_var,
                "count_var": count_var, "log": log, "count": 0,
            }
            self._start_flock_listener(key)
            return

        elif t == "sensor":
            st = comp.get("sensor_type", "?")
            val_var   = tk.StringVar(value="-- reading --")
            alert_var = tk.StringVar(value="")

            # Big reading display
            tk.Label(inner, textvariable=val_var,
                     bg=CARD, fg=SENSOR_C,
                     font=f_big).pack(pady=4)

            alert_lbl = tk.Label(inner, textvariable=alert_var,
                                  bg=CARD, fg=RED, font=f_small)
            alert_lbl.pack()

            # Store ref for sensor loop
            key = f"sensor_{comp['name']}"
            if not hasattr(self, "_sensor_vars"):
                self._sensor_vars = {}
            self._sensor_vars[key] = {
                "comp": comp, "val_var": val_var, "alert_var": alert_var
            }

            # Manual refresh button
            tk.Button(inner, text="REFRESH", bg=SENSOR_C, fg="#000",
                      font=f_small, relief="flat", cursor="hand2",
                      command=lambda k=key: self._read_sensor(k)
                      ).pack(pady=4)

            # Load/configure the Audio AI toolkit (if this mic has any of
            # its 5 toggles on) BEFORE scheduling the mic reader below —
            # the reader spawns a background thread almost immediately,
            # so this order guarantees _aa_* attributes exist before that
            # thread could ever touch them, closing the race at its root
            # rather than only defending against it downstream.
            if st == "microphone":
                self._start_audio_ai(comp)

            # Start auto-reading (DHT sensors need ≥2s between reads)
            poll_interval = 2500 if st == "temperature" else 200 if st == "microphone" else 1000
            self._schedule_sensor(key, poll_interval)

            # ── Record / Playback (microphone only) ──────────────────────
            if st == "microphone":
                rec_status = tk.StringVar(value="")
                rec_frame  = tk.Frame(inner, bg=CARD)
                rec_frame.pack(pady=(6, 0))

                rec_btn  = tk.Button(rec_frame, text="⏺ RECORD", bg=RED,     fg="#fff",
                                     font=f_small, relief="flat", cursor="hand2", width=10)
                play_btn = tk.Button(rec_frame, text="▶ PLAY",   bg=SOUND_C, fg="#000",
                                     font=f_small, relief="flat", cursor="hand2", width=10)
                stop_btn = tk.Button(rec_frame, text="⏹ STOP",   bg=BORDER,  fg=MUTED,
                                     font=f_small, relief="flat", cursor="hand2", width=10)
                rec_btn.pack(side="left",  padx=4)
                play_btn.pack(side="left", padx=4)
                stop_btn.pack(side="left", padx=4)

                tk.Label(inner, textvariable=rec_status,
                         bg=CARD, fg=MUTED, font=f_small).pack()

                self._mic_recording = False
                self._mic_stop_flag = threading.Event()
                self._mic_last_file = None

                def _do_record(c=comp, status=rec_status):
                    import subprocess, tempfile
                    self._mic_stop_flag.clear()
                    self._mic_recording = True
                    # Stop monitor stream so device is free
                    try:
                        self._mic_proc.terminate()
                        self._mic_monitor_running = False
                    except Exception: pass
                    time.sleep(0.3)

                    device_str = getattr(self, "_mic_device_str", "default")
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                    tmp.close()
                    cmd = ["arecord", "-D", device_str,
                           "-f", "S16_LE", "-r", "44100", "-c", "1",
                           "-t", "wav", tmp.name]
                    try:
                        proc = subprocess.Popen(cmd, stderr=subprocess.DEVNULL)
                        self._rec_proc = proc
                        self.after(0, lambda: status.set("● Recording..."))
                        self._mic_stop_flag.wait()
                        proc.terminate()
                        proc.wait()
                        self._mic_last_file = tmp.name
                        self.after(0, lambda: status.set("Saved ✓"))
                    except Exception as e:
                        self.after(0, lambda err=str(e): status.set(f"err: {err[:30]}"))
                    finally:
                        self._mic_recording = False
                        # Restart monitor
                        self._mic_monitor_running = False
                        self._start_mic_monitor(comp)

                def _do_play(status=rec_status):
                    if not self._mic_last_file:
                        status.set("Nothing recorded yet"); return
                    try:
                        if not pygame.mixer.get_init():
                            pygame.mixer.init(frequency=44100)
                        sound = pygame.mixer.Sound(self._mic_last_file)
                        sound.play()
                        status.set("▶ Playing...")
                    except Exception as e:
                        # Fallback to aplay
                        try:
                            import subprocess
                            subprocess.Popen(["aplay", self._mic_last_file],
                                             stderr=subprocess.DEVNULL)
                            status.set("▶ Playing...")
                        except Exception as e2:
                            status.set(f"Play error: {e2}")

                def _do_stop(status=rec_status):
                    self._mic_stop_flag.set()
                    pygame.mixer.stop()
                    status.set("Stopped")

                rec_btn.config( command=lambda: threading.Thread(target=_do_record, daemon=True).start())
                play_btn.config(command=_do_play)
                stop_btn.config(command=_do_stop)

                # ── Audio AI Toolkit panel (mirrors the camera's local
                #    vision toolkit) — only shown if at least one is on.
                audio_ai_flags = ("detect_animal_sounds", "detect_alarms",
                                  "detect_custom_sounds", "speech_to_text",
                                  "translate_enabled")
                if any(comp.get(k, False) for k in audio_ai_flags):
                    tk.Frame(inner, bg=BORDER, height=1).pack(fill="x", padx=8, pady=(8, 2))
                    tk.Label(inner, text="[A] AUDIO AI",
                             bg=CARD, fg=SOUND_C, font=f_small).pack()

                    self._audio_log = tk.Text(inner, height=6, bg=BG, fg=FG,
                                              insertbackground=FG, font=f_small,
                                              relief="flat", wrap="word")
                    self._audio_log.pack(fill="x", padx=8, pady=(2, 6))
                    self._audio_log.tag_config("hit", foreground=SOUND_C)
                    self._audio_log.configure(state="disabled")

    # ── Sensor reading ────────────────────────────────────────────────────
    def _schedule_sensor(self, key, interval=1000):
        if self._running:
            self._read_sensor(key)
            self.after(interval, lambda: self._schedule_sensor(key, interval))

    def _flybrain_log_line(self, key, text, hit=False):
        state = getattr(self, "_flybrain_state", {}).get(key)
        if not state:
            return
        def _do():
            log = state["log"]
            log.configure(state="normal")
            log.insert("end", text + "\n", ("hit",) if hit else ())
            log.see("end")
            if int(log.index("end-1c").split(".")[0]) > 200:
                log.delete("1.0", "2.0")
            log.configure(state="disabled")
        self.after(0, _do)

    def _flybrain_reward(self, comp):
        key = f"flybrain_{comp['name']}"
        state = getattr(self, "_flybrain_state", {}).get(key)
        if state and state["sim"]:
            state["sim"].reward()
            self._flybrain_log_line(key, "[REWARD] reinforced current activity", hit=True)

    def _setup_brain_graph(self, key, sim):
        """Precomputes a clustered 2D layout (visual / central-brain / motor
        neurons, same grouping the reference connectome dashboards use) and
        creates one canvas dot per neuron up-front. Updates afterward only
        recolor existing dots (canvas.itemconfig) rather than recreating
        them — recreating hundreds of canvas items every frame would be
        far too slow on a Pi."""
        state  = self._flybrain_state[key]
        canvas = state["graph_canvas"]
        canvas.delete("all")

        w, h = 260, 170
        rng  = np.random.default_rng(42) if HAS_NUMPY else None
        sensory_set = set(sim.sensory_idx)
        motor_set_  = set(sim.motor_idx)

        # Cap how many dots actually get drawn for large real-connectome
        # slices — thousands of canvas items would make the whole UI
        # sluggish. Sample representative neurons from each role instead.
        MAX_DOTS = 400
        all_idx = list(range(sim.n))
        if sim.n > MAX_DOTS and HAS_NUMPY:
            all_idx = sorted(rng.choice(sim.n, MAX_DOTS, replace=False).tolist())

        positions, roles, dot_ids = [], [], []
        for i in all_idx:
            if i in sensory_set:
                role = "sensory"; cx, cy = w * 0.18, h * 0.5
            elif i in motor_set_:
                role = "motor"; cx, cy = w * 0.82, h * 0.5
            else:
                role = "other"; cx, cy = w * 0.5, h * 0.5
            jitter_x = (rng.normal(0, w * 0.11) if rng is not None else 0)
            jitter_y = (rng.normal(0, h * 0.32) if rng is not None else 0)
            x = max(6, min(w - 6, cx + jitter_x))
            y = max(6, min(h - 6, cy + jitter_y))
            color = {"sensory": "#3fa9f5", "motor": "#ff5a5a", "other": "#6b6b78"}[role]
            dot = canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")
            positions.append((x, y)); roles.append(role); dot_ids.append(dot)

        # Cluster labels, matching the reference dashboards' legend style
        canvas.create_text(w * 0.18, 12, text="VISUAL", fill="#3fa9f5", font=("Courier", 7))
        canvas.create_text(w * 0.5, 12, text="CENTRAL", fill="#6b6b78", font=("Courier", 7))
        canvas.create_text(w * 0.82, 12, text="MOTOR", fill="#ff5a5a", font=("Courier", 7))

        state["dot_ids"]   = dot_ids
        state["positions"] = positions
        state["roles"]     = roles
        state["idx_map"]   = all_idx   # which real neuron index each dot represents

    def _update_brain_graph(self, key, spikes, motor_rates):
        """Recolors the precomputed dots: bright = spiked this frame, dim =
        idle. Also redraws the motor-rate bar readout, same idea as the
        'READOUT → ACTION' gauges in the reference dashboards."""
        state = self._flybrain_state.get(key)
        if not state or not state["dot_ids"]:
            return
        canvas = state["graph_canvas"]
        bright = {"sensory": "#8ecbff", "motor": "#ff9b9b", "other": "#c9c9d6"}
        dim    = {"sensory": "#1c4f73", "motor": "#732626", "other": "#2c2c33"}
        for dot, role, idx in zip(state["dot_ids"], state["roles"], state["idx_map"]):
            is_spiking = idx < len(spikes) and spikes[idx] > 0
            canvas.itemconfig(dot, fill=(bright if is_spiking else dim)[role])

        mcanvas = state["motor_canvas"]
        mcanvas.delete("all")
        mw = mcanvas.winfo_width()
        if mw <= 1:   # not yet mapped/rendered on the very first call
            mw = 260
        bar_h = 70 // max(1, len(motor_rates))
        for i, rate in enumerate(motor_rates):
            y = i * bar_h
            bar_w = int(max(0, min(1, rate)) * (mw - 60))
            mcanvas.create_text(4, y + bar_h // 2, text=f"M{i}", fill=MUTED,
                                font=("Courier", 7), anchor="w")
            mcanvas.create_rectangle(24, y + 2, 24 + bar_w, y + bar_h - 2,
                                     fill="#9b59ff", outline="")

    def _start_flybrain(self, key):
        state = self._flybrain_state[key]
        comp  = state["comp"]

        sim = FlyBrainSim(
            n_neurons=int(comp.get("n_neurons", 500)),
            n_sensory=int(comp.get("n_sensory", 64)),
            n_motor=int(comp.get("n_motor", 8)),
            connectome_file=comp.get("connectome_file") or None,
            annotations_file=comp.get("annotations_file") or None)
        state["sim"] = sim
        self._setup_brain_graph(key, sim)

        pan_name    = comp.get("target_servo_pan", "")
        tilt_name   = comp.get("target_servo_tilt", "")
        motor_name  = comp.get("target_motor", "")
        auto_reward = comp.get("auto_reward_on_tracking", True)
        rate_hz     = max(1, int(comp.get("sim_rate_hz", 10)))
        dt          = 1.0 / rate_hz

        def _find(ctype, name):
            if not name:
                return None
            return next((c for c in self.app.components
                        if c["type"] == ctype and c["name"] == name), None)

        pan_comp   = _find("servo", pan_name)
        tilt_comp  = _find("servo", tilt_name)
        motor_comp = _find("motor", motor_name)

        # Optional: map motor neurons straight onto whatever's already bound
        # to a keyboard key on the Control page — reuses the exact same
        # dispatch table _on_key_press() uses, so a fly-brain-driven "w"
        # does exactly what a human pressing "w" would do. This is the
        # same trick the viral Doom/Mario demos used (neurons → keypresses).
        target_keys = [k.strip().lower() for k in
                       comp.get("target_keys", "").split(",") if k.strip()]
        KEY_FIRE_THRESHOLD = 0.3
        key_was_firing = {k: False for k in target_keys}

        def _get_sensory_input():
            """Downsamples the live camera frame (if any) into a grid of
            average-brightness values — a coarse stand-in for the fly's
            compound-eye ommatidia. No camera = flat/idle input."""
            n = sim.n_sensory
            frame = getattr(self, "_latest_frame", None)
            if frame is None or not HAS_CV2 or not HAS_NUMPY:
                return np.zeros(n, dtype=np.float32)
            try:
                side = max(1, int(n ** 0.5))
                small = cv2.resize(frame, (side, side))
                gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
                flat  = gray.flatten()
                if len(flat) < n:
                    flat = np.pad(flat, (0, n - len(flat)))
                return flat[:n]
            except Exception:
                return np.zeros(n, dtype=np.float32)

        def _loop():
            self.after(0, lambda: state["status_var"].set(
                f"● running ({sim.n} neurons, {rate_hz} Hz)"))
            step_count = 0
            while self._running:
                try:
                    sim.set_sensory_input(_get_sensory_input())
                    sim.step(dt)
                    step_count += 1

                    rates = sim.motor_rate
                    if len(rates) >= 2 and (pan_comp or tilt_comp):
                        pan_signal  = float(rates[0]) - float(rates[len(rates)//2 % len(rates)])
                        tilt_signal = float(rates[1 % len(rates)]) - float(rates[-1])
                        if pan_comp:
                            cur = self._servo_angles.get(pan_comp["name"], 90.0)
                            new = max(pan_comp["min_deg"], min(pan_comp["max_deg"],
                                     cur + pan_signal * pan_comp.get("step", 2)))
                            self._servo_angles[pan_comp["name"]] = new
                            servo_set(pan_comp["pin"], new,
                                     pan_comp["pulse_min"], pan_comp["pulse_max"])
                        if tilt_comp:
                            cur = self._servo_angles.get(tilt_comp["name"], 90.0)
                            new = max(tilt_comp["min_deg"], min(tilt_comp["max_deg"],
                                     cur + tilt_signal * tilt_comp.get("step", 2)))
                            self._servo_angles[tilt_comp["name"]] = new
                            servo_set(tilt_comp["pin"], new,
                                     tilt_comp["pulse_min"], tilt_comp["pulse_max"])
                    if motor_comp and len(rates):
                        speed = int(max(-100, min(100, (float(rates.mean()) - 0.1) * 400)))
                        motor_set(motor_comp["pin_fwd"], motor_comp["pin_bwd"], speed)

                    if target_keys and len(rates):
                        for i, k in enumerate(target_keys):
                            if i >= len(rates):
                                break
                            firing = float(rates[i]) > KEY_FIRE_THRESHOLD
                            # Only dispatch on the rising edge (like a real
                            # keypress event, not a held-down repeat every
                            # single sim step) — fire once per activation.
                            if firing and not key_was_firing.get(k, False):
                                actions = getattr(self, "_keymap", {}).get(k, [])
                                for fn, _label in actions:
                                    try: fn()
                                    except Exception as e:
                                        print(f"[FLYBRAIN] key '{k}' action error: {e}")
                            key_was_firing[k] = firing

                    if auto_reward and getattr(self, "_face_found", False):
                        sim.reward(magnitude=0.3)

                    # Redraw the graph at ~5Hz regardless of sim_rate_hz —
                    # no point re-rendering hundreds of canvas items faster
                    # than the eye (or the Pi) can keep up with.
                    redraw_every = max(1, rate_hz // 5)
                    if step_count % redraw_every == 0:
                        spikes_snapshot = sim._recent_spikes.copy()
                        rates_snapshot  = rates.copy()
                        self.after(0, lambda s=spikes_snapshot, r=rates_snapshot:
                                  self._update_brain_graph(key, s, r))

                    if step_count % (rate_hz * 2) == 0:  # ~every 2s
                        active = int((sim._recent_spikes > 0).sum())
                        self.after(0, lambda a=active: state["status_var"].set(
                            f"● running ({a}/{sim.n} neurons active)"))
                        self.after(0, lambda r=rates.copy(): state["motor_var"].set(
                            "motor rates: " + " ".join(f"{v:.2f}" for v in r)))
                except Exception as e:
                    print(f"[FLYBRAIN] step error: {e}")
                time.sleep(dt)
            print("[FLYBRAIN] stopped")

        if not HAS_NUMPY:
            state["status_var"].set("⚠ numpy required — not running")
            self._flybrain_log_line(key,
                "[ERROR] numpy is required for the Fly Brain Controller")
            return

        threading.Thread(target=_loop, daemon=True).start()

    def _start_power_monitor(self, key):
        """Polls Pi health (vcgencmd) every ~2s, plus an optional INA219
        rail reading if enabled. Runs in a background thread since vcgencmd
        is a subprocess call — cheap, but no reason to block the UI loop."""
        state = self._power_state[key]
        comp  = state["comp"]

        if comp.get("enable_ina219", False):
            try_load_ina219(comp.get("ina219_addr", "0x40"))

        def _poll():
            while self._running:
                if comp.get("monitor_pi", True):
                    info = read_pi_power()
                    if info["ok"]:
                        flags = set(info["flags"])
                        warn = any("under-voltage detected" in f or "currently throttled" in f
                                  for f in flags)
                        self.after(0, lambda w=warn: state["status_var"].set(
                            "⚠ POWER ISSUE" if w else "● OK"))
                        self.after(0, lambda i=info: state["volt_var"].set(
                            f"core: {i['volts_core']:.3f} V" if i['volts_core'] is not None else "core: -- V"))
                        self.after(0, lambda i=info: state["temp_var"].set(
                            f"temp: {i['temp_c']:.1f} °C" if i['temp_c'] is not None else "temp: -- °C"))

                        new_flags = flags - state["last_flags"]
                        recovered = bool(state["last_flags"]) and not flags
                        if new_flags and comp.get("alert_on_undervoltage", True):
                            for f in new_flags:
                                is_current = not f.endswith("since boot")
                                self._power_log_line(key, f"[{'NOW' if is_current else 'HIST'}] {f}",
                                                    warn=is_current)
                        elif recovered:
                            self._power_log_line(key, "[OK] issue cleared")
                        elif not flags and "logged_initial_ok" not in state:
                            self._power_log_line(key, "[OK] no throttle/voltage issues")
                            state["logged_initial_ok"] = True
                        state["last_flags"] = flags
                    else:
                        self.after(0, lambda i=info: state["status_var"].set(
                            f"⚠ vcgencmd unavailable: {i.get('error', '?')[:40]}"))

                if comp.get("enable_ina219", False) and HAS_INA219:
                    rail = read_ina219()
                    if rail:
                        v, ma, mw = rail
                        label = comp.get("ina219_label", "Rail")
                        self.after(0, lambda l=label, v=v, ma=ma, mw=mw:
                                  state["rail_var"].set(
                                      f"{l}: {v:.2f}V  {ma:.0f}mA  {mw:.0f}mW"))
                time.sleep(2.0)

        threading.Thread(target=_poll, daemon=True).start()

    def _power_log_line(self, key, text, warn=False):
        state = getattr(self, "_power_state", {}).get(key)
        if not state:
            return
        def _do():
            log = state["flags_log"]
            log.configure(state="normal")
            log.insert("end", text + "\n", ("warn",) if warn else ())
            log.see("end")
            if int(log.index("end-1c").split(".")[0]) > 100:
                log.delete("1.0", "2.0")
            log.configure(state="disabled")
        self.after(0, _do)

    def _start_flock_listener(self, key):
        """Opens a persistent background reader for a Flock Detector sensor.
        Real hardware: reads newline-delimited JSON over USB serial from a
        colonelpanichacks/flock-you or oui-spy ESP32 device.
        No hardware / no pyserial: emits occasional simulated detections so
        the panel is still demoable off-Pi."""
        if not hasattr(self, "_flock_state") or key not in self._flock_state:
            return
        state = self._flock_state[key]
        comp  = state["comp"]

        def _log_line(text, hit=False):
            def _do():
                log = state["log"]
                log.configure(state="normal")
                log.insert("end", text + "\n", ("hit",) if hit else ())
                log.see("end")
                # keep the log short
                if int(log.index("end-1c").split(".")[0]) > 200:
                    log.delete("1.0", "2.0")
                log.configure(state="disabled")
            self.after(0, _do)

        def _handle_detection(det):
            state["count"] += 1
            self.after(0, lambda: state["count_var"].set(f"{state['count']} detections"))
            ssid = det.get("ssid", "?")
            mac  = det.get("mac_address", "?")
            rssi = det.get("rssi", "?")
            method = det.get("detection_method", "?")
            min_rssi = comp.get("min_rssi", -90)
            try:
                if rssi != "?" and int(rssi) < int(min_rssi):
                    return  # too weak / out of configured range
            except (TypeError, ValueError):
                pass
            _log_line(f"[{method}] {ssid}  {mac}  rssi={rssi}", hit=True)
            if comp.get("alert_enabled", True):
                self.after(0, lambda: state["status_var"].set(
                    f"● FLOCK CAMERA DETECTED — {ssid}"))
                self.after(4000, lambda: state["status_var"].set("● listening..."))

        def _run_real():
            port = comp.get("serial_port", "/dev/ttyUSB0")
            baud = int(comp.get("baud_rate", 115200))
            while self._running:
                try:
                    with _pyserial.Serial(port, baud, timeout=1) as ser:
                        self.after(0, lambda: state["status_var"].set("● listening..."))
                        while self._running:
                            line = ser.readline().decode("utf-8", "ignore").strip()
                            if not line:
                                continue
                            try:
                                det = json.loads(line)
                            except json.JSONDecodeError:
                                continue
                            _handle_detection(det)
                except Exception as e:
                    self.after(0, lambda err=str(e): state["status_var"].set(
                        f"○ {port} unavailable: {err[:40]}"))
                    time.sleep(3)

        def _run_sim():
            import random
            self.after(0, lambda: state["status_var"].set("● listening (simulated - no serial device)"))
            ssids = ["Flock_Camera_014", "Flock_Camera_221", "Flock_Camera_003"]
            while self._running:
                time.sleep(random.uniform(8, 20))
                if not self._running:
                    break
                _handle_detection({
                    "ssid": random.choice(ssids),
                    "mac_address": ":".join(f"{random.randint(0,255):02x}" for _ in range(6)),
                    "rssi": random.randint(-85, -40),
                    "detection_method": random.choice(["probe_request", "oui_match"]),
                })

        target = _run_real if (HAS_SERIAL and ON_PI) else _run_sim
        threading.Thread(target=target, daemon=True).start()

    def _read_sensor(self, key):
        if not hasattr(self, "_sensor_vars") or key not in self._sensor_vars:
            return
        info = self._sensor_vars[key]
        comp = info["comp"]
        st   = comp.get("sensor_type", "?")

        def _read():
            try:
                if st == "ultrasonic":
                    dist = self._read_ultrasonic(comp)
                    text = f"{dist:.1f} cm" if dist is not None else "no echo / timeout"
                    alert = ""
                    if dist is not None and comp.get("alert_enabled") and dist < comp.get("alert_distance", 30):
                        alert = f"! OBJECT DETECTED at {dist:.1f}cm!"
                    self.after(0, lambda t=text, a=alert: (
                        info["val_var"].set(t), info["alert_var"].set(a)))

                elif st == "motion":
                    detected = self._read_motion(comp)
                    if detected:
                        text  = "No motion"
                        alert = ""
                        self.after(0, lambda t=text, a=alert: (
                            info["val_var"].set(t), info["alert_var"].set(a)))
                    else:
                        text  = "MOTION DETECTED!"
                        alert = "! ALERT!" if comp.get("alert_enabled") else ""
                        self.after(0, lambda t=text, a=alert: (
                            info["val_var"].set(t), info["alert_var"].set(a)))
                        self.after(3000, lambda: (
                            info["val_var"].set("No motion"),
                            info["alert_var"].set("")
                        ) if self._read_motion(comp) else None)

                elif st == "temperature":
                    temp, hum = self._read_temperature(comp)
                    if temp is not None:
                        text  = f"{temp:.1f}C  {hum:.1f}%"
                        alert = f"! HIGH TEMP: {temp:.1f}C!" if comp.get("alert_enabled") and temp > comp.get("alert_temp", 40) else ""
                    else:
                        text  = f"err: {str(hum)[:28]}" if hum else "sensor error"
                        alert = ""
                    self.after(0, lambda t=text, a=alert: (
                        info["val_var"].set(t), info["alert_var"].set(a)))

                elif st == "microphone":
                    vol, triggered = self._read_microphone(comp)
                    if vol is None:
                        text  = f"err: {triggered[:30]}" if triggered else "mic error"
                        alert = ""
                    else:
                        bars  = int(vol / 10)
                        bar   = "█" * bars + "░" * (10 - bars)
                        text  = f"{bar} {vol:.0f}"
                        alert = "! NOISE DETECTED!" if triggered and comp.get("alert_enabled") else ""
                    self.after(0, lambda t=text, a=alert: (
                        info["val_var"].set(t), info["alert_var"].set(a)))

            except Exception as e:
                self.after(0, lambda err=str(e): info["val_var"].set(f"err: {err[:30]}"))

        threading.Thread(target=_read, daemon=True).start()

    def _read_ultrasonic(self, comp):
        if not ON_PI:
            import random
            return random.uniform(10, 100)
        try:
            trig = int(comp["pin_trig"])
            echo = int(comp["pin_echo"])
            GPIO.setup(trig, GPIO.OUT)
            GPIO.setup(echo, GPIO.IN)
            GPIO.output(trig, False)
            time.sleep(0.02)
            GPIO.output(trig, True)
            time.sleep(0.00001)
            GPIO.output(trig, False)
            pulse_start = time.time()
            while GPIO.input(echo) == 0:
                if time.time() - pulse_start > 0.1: return None
            pulse_start = time.time()
            pulse_end = pulse_start
            while GPIO.input(echo) == 1:
                pulse_end = time.time()
                if pulse_end - pulse_start > 0.1: return None
            return (pulse_end - pulse_start) * 17150
        except Exception:
            return None

    def _read_motion(self, comp):
        if not ON_PI:
            import random
            return random.random() > 0.8
        try:
            if not hasattr(self, "_motion_pins_setup"):
                self._motion_pins_setup = set()
            pin = int(comp["pin"])
            if pin not in self._motion_pins_setup:
                GPIO.setup(pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                self._motion_pins_setup.add(pin)
            return GPIO.input(pin) == GPIO.HIGH
        except Exception as e:
            print(f"[MOTION] error: {e}")
            return False

    def _get_pyaudio(self):
        pass  # kept for compatibility, unused

    def _audio_log_line(self, text, hit=False):
        if not hasattr(self, "_audio_log"):
            return
        def _do():
            log = self._audio_log
            log.configure(state="normal")
            log.insert("end", text + "\n", ("hit",) if hit else ())
            log.see("end")
            if int(log.index("end-1c").split(".")[0]) > 300:
                log.delete("1.0", "2.0")
            log.configure(state="disabled")
        self.after(0, _do)

    def _start_audio_ai(self, comp):
        """Lazily loads whichever Audio AI models are enabled for this mic,
        and stores config on self for the mic monitor loop to use. Same
        shape as _start_local_vision for the camera."""
        self._aa_detect_animals  = comp.get("detect_animal_sounds", False)
        self._aa_detect_alarms   = comp.get("detect_alarms", False)
        self._aa_detect_custom   = comp.get("detect_custom_sounds", False)
        self._aa_speech_to_text  = comp.get("speech_to_text", False)
        self._aa_translate       = comp.get("translate_enabled", False)
        self._aa_target_lang     = comp.get("translate_target_lang", "es")
        self._aa_custom_dir      = comp.get("custom_sounds_dir", "reference_sounds")
        self._aa_last_event      = 0.0
        self._aa_last_speech     = 0.0
        self._aa_last_animal_logged = None
        self._aa_last_alarm_logged  = None

        if self._aa_detect_animals or self._aa_detect_alarms:
            _try_load_yamnet()

        if self._aa_detect_custom:
            _load_reference_sounds(self._aa_custom_dir)
            if not _ref_sounds:
                self._audio_log_line(
                    f"no reference sounds yet in '{self._aa_custom_dir}/' "
                    f"— add labelled WAV clips there (e.g. my_alarm.wav)")

        if self._aa_speech_to_text or self._aa_translate:
            _try_load_vosk()


        if getattr(self, "_mic_monitor_running", False):
            return
        self._mic_monitor_running = True
        self._mic_vol  = 0.0
        self._mic_trig = False
        self._mic_err  = None

        def _find_device():
            import subprocess, re
            devices = []
            # Honour the configured Device Index FIRST — previously this
            # setting was collected in the editor and then never used,
            # so picking a device had no effect at all.
            try:
                idx = int(comp.get("device_index", 0))
                devices.append(f"plughw:{idx},0")
            except (TypeError, ValueError):
                pass
            try:
                out = subprocess.check_output(["arecord", "-l"],
                                              stderr=subprocess.DEVNULL).decode()
                for m in re.finditer(r"card (\d+):.*?device (\d+):", out):
                    devices.append(f"plughw:{m.group(1)},{m.group(2)}")
            except Exception:
                pass
            devices += ["default", "plughw:0,0", "plughw:1,0", "plughw:2,0"]
            seen = set()
            for dev in devices:
                if dev in seen:
                    continue
                seen.add(dev)
                proc = None
                try:
                    proc = subprocess.Popen(
                        ["arecord", "-D", dev, "-f", "S16_LE", "-r", "16000", "-c", "1", "-"],
                        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
                    time.sleep(0.4)
                    if proc.poll() is None:
                        return dev, proc
                    proc.terminate()
                except Exception:
                    # Make sure a half-started arecord never survives us.
                    if proc is not None:
                        try:
                            proc.terminate()
                        except Exception:
                            pass
                    continue
            return None, None

        def _monitor():
            import struct, math
            device_str, proc = _find_device()
            if proc is None:
                self._mic_vol = -1
                self._mic_err = "no mic found - check USB connection"
                self._mic_monitor_running = False
                return
            self._mic_device_str = device_str
            self._mic_proc = proc
            print(f"[MIC] using: {device_str}")
            CHUNK = 1024
            SR    = 16000   # 16kHz mono — what YAMNet/Vosk both expect,
                             # and plenty for level metering too
            audio_buffer = []     # rolling window of raw int16 samples
            BUFFER_TARGET = SR * 2   # ~2 seconds per AI analysis pass
            # Also watch self._running so leaving the Control page actually
            # stops us — otherwise arecord kept holding the mic device open
            # forever and the NEXT launch could not open it.
            while self._mic_monitor_running and self._running:
                if getattr(self, "_mic_recording", False):
                    time.sleep(0.05); continue
                try:
                    raw = proc.stdout.read(CHUNK * 2)
                    if not raw or len(raw) < 4: break
                    count = len(raw) // 2
                    samples = struct.unpack(f"{count}h", raw[:count * 2])
                    rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                    # Audio levels are logarithmic, so a linear percentage
                    # pegged at 100 during normal speech. Map RMS to dBFS
                    # (-60..0 dB) instead, which spreads quiet-to-loud across
                    # the whole meter the way an audio level meter should.
                    if rms > 1.0:
                        dbfs = 20.0 * math.log10(rms / 32768.0)
                    else:
                        dbfs = -60.0
                    self._mic_vol = max(0.0, min(100.0, (dbfs + 60.0) / 60.0 * 100.0))
                    # Re-read the threshold each loop so editing it takes
                    # effect immediately instead of only on restart.
                    try:
                        threshold = int(comp.get("threshold", 500))
                    except (TypeError, ValueError):
                        threshold = 500
                    self._mic_trig = rms > threshold

                    # ── Audio AI toolkit — buffer samples, process in ~2s
                    #    chunks so YAMNet/Vosk aren't invoked on tiny slices.
                    any_ai = (getattr(self, "_aa_detect_animals", False)
                             or getattr(self, "_aa_detect_alarms", False)
                             or getattr(self, "_aa_detect_custom", False)
                             or getattr(self, "_aa_speech_to_text", False))
                    if any_ai and HAS_NUMPY:
                        audio_buffer.extend(samples)
                        if len(audio_buffer) >= BUFFER_TARGET:
                            chunk_i16 = np.array(audio_buffer[:BUFFER_TARGET], dtype=np.int16)
                            audio_buffer = audio_buffer[BUFFER_TARGET:]
                            # Dispatched to its OWN thread, not called inline —
                            # translation alone can take up to 120s (Ollama's
                            # timeout), and YAMNet/Vosk aren't instant either.
                            # Running any of that in THIS loop would freeze
                            # the level meter (which lives in this same loop)
                            # for however long the AI call takes. Guarded so
                            # a slow chunk never piles up multiple overlapping
                            # analysis threads — just skips a chunk instead.
                            if not getattr(self, "_aa_busy", False):
                                self._aa_busy = True
                                def _run_ai(chunk=chunk_i16, sr=SR):
                                    try:
                                        self._process_audio_ai_chunk(chunk, sr)
                                    finally:
                                        self._aa_busy = False
                                threading.Thread(target=_run_ai, daemon=True).start()
                except Exception as e:
                    print(f"[MIC] {e}"); break
            try: proc.terminate()
            except Exception: pass
            self._mic_proc = None
            self._mic_monitor_running = False
            print("[MIC] monitor stopped, device released")

        threading.Thread(target=_monitor, daemon=True).start()

    def _process_audio_ai_chunk(self, chunk_i16, sr):
        """Runs whichever Audio AI features are enabled on one ~2s chunk of
        int16 mono audio. Called from the mic monitor thread — dispatches
        each analysis and logs hits, same pattern as the camera's vision
        toolkit. Each heavy step is wrapped so one failing engine (e.g.
        YAMNet not installed) never blocks the others."""
        chunk_f32 = chunk_i16.astype(np.float32) / 32768.0
        # Skip near-silence — running these models on silence wastes CPU
        # and Vosk in particular tends to hallucinate short junk words.
        if float(np.abs(chunk_f32).mean()) < 0.003:
            return

        if (getattr(self, "_aa_detect_animals", False)
                or getattr(self, "_aa_detect_alarms", False)) and HAS_YAMNET:
            try:
                top = _classify_audio_chunk(chunk_f32)
                if getattr(self, "_aa_detect_animals", False):
                    m = _match_audio_event(top, ANIMAL_SOUND_KEYWORDS)
                    if m and m[0] != getattr(self, "_aa_last_animal_logged", None):
                        self._audio_log_line(f"[ANIMAL] {m[0]} ({m[1]:.0%})", hit=True)
                        self._aa_last_animal_logged = m[0]
                if getattr(self, "_aa_detect_alarms", False):
                    m = _match_audio_event(top, ALARM_SOUND_KEYWORDS)
                    if m and m[0] != getattr(self, "_aa_last_alarm_logged", None):
                        self._audio_log_line(f"[ALARM] {m[0]} ({m[1]:.0%})", hit=True)
                        self._aa_last_alarm_logged = m[0]
            except Exception as e:
                print(f"[AUDIO] classification error: {e}")

        if getattr(self, "_aa_detect_custom", False):
            try:
                match = _match_reference_sound(chunk_f32, sr=sr)
                if match:
                    self._audio_log_line(f"[MATCH] {match[0]} ({match[1]:.0%})", hit=True)
            except Exception as e:
                print(f"[AUDIO] custom sound match error: {e}")

        if (getattr(self, "_aa_speech_to_text", False)
                or getattr(self, "_aa_translate", False)) and HAS_VOSK:
            try:
                text = _speech_to_text(chunk_i16, sr=sr)
                if text:
                    self._audio_log_line(f"[SPEECH] {text}", hit=True)
                    if getattr(self, "_aa_translate", False):
                        target_lang = getattr(self, "_aa_target_lang", "es")
                        translated = _translate_text(text, target_lang)
                        if translated:
                            self._audio_log_line(
                                f"[{target_lang.upper()}] {translated}", hit=True)
            except Exception as e:
                print(f"[AUDIO] speech-to-text error: {e}")

    def _read_microphone(self, comp):
        if not ON_PI:
            import random
            return random.uniform(0, 40), False
        self._start_mic_monitor(comp)
        vol = getattr(self, "_mic_vol", 0.0)
        if vol == -1:
            return None, getattr(self, "_mic_err", "mic error")
        return vol, getattr(self, "_mic_trig", False)

    def _read_temperature(self, comp):
        if not ON_PI:
            import random
            return round(random.uniform(20, 45), 1), round(random.uniform(30, 80), 1)
        try:
            import adafruit_dht, board
            if not hasattr(self, "_dht_handles"):
                self._dht_handles = {}
            pin_num = int(comp["pin"])
            key = f"dht_{pin_num}"
            if key not in self._dht_handles:
                pin = getattr(board, f"D{pin_num}")
                self._dht_handles[key] = (
                    adafruit_dht.DHT22(pin) if comp.get("model") == "DHT22"
                    else adafruit_dht.DHT11(pin)
                )
            dht = self._dht_handles[key]

            # DHT sensors fail randomly - retry up to 5 times with 0.5s gap
            last_err = ""
            for attempt in range(5):
                try:
                    temp = dht.temperature
                    hum  = dht.humidity
                    if temp is not None and hum is not None:
                        return temp, hum
                except Exception as e:
                    last_err = str(e)
                time.sleep(0.5)

            # All retries failed - reset the object so next poll starts fresh
            try:
                dht.exit()
            except Exception:
                pass
            del self._dht_handles[key]
            return None, f"5 retries failed: {last_err[:40]}"

        except Exception as e:
            return None, str(e)

    # ── Keyboard ──────────────────────────────────────────────────────────
    def _setup_keyboard(self):
        self._keymap   = {}
        self._held_keys = set()

        for comp in self.app.components:
            t    = comp["type"]
            name = comp["name"]

            if t == "servo":
                for key, direction, label in [
                    (comp.get("key_left",  ""), -1, f"{name} LEFT"),
                    (comp.get("key_right", ""), +1, f"{name} RIGHT"),
                ]:
                    if key:
                        self._keymap.setdefault(key.lower(), []).append(
                            (lambda c=comp, d=direction: self._kb_servo(c, d), label))
                reset_key = comp.get("key_reset", "")
                if reset_key:
                    self._keymap.setdefault(reset_key.lower(), []).append(
                        (lambda c=comp: self._kb_servo_reset(c), f"{name} RESET"))

            elif t == "motor":
                for key, speed, label in [
                    (comp.get("key_fwd",  ""),  80, f"{name} FWD"),
                    (comp.get("key_bwd",  ""), -80, f"{name} BWD"),
                    (comp.get("key_stop", ""),   0, f"{name} STOP"),
                ]:
                    if key:
                        self._keymap.setdefault(key.lower(), []).append(
                            (lambda c=comp, s=speed: self._kb_motor(c, s), label))

            elif t == "laser":
                key = comp.get("key_toggle", "")
                if key:
                    self._keymap.setdefault(key.lower(), []).append(
                        (lambda c=comp: self._kb_laser(c), f"{name} TOGGLE"))

        # T always toggles tracking if camera exists
        if self.cam_comps:
            self._keymap.setdefault("t", []).append(
                (self._toggle_tracking, "TRACKING"))

        self.app.bind("<KeyPress>",   self._on_key_press)
        self.app.bind("<KeyRelease>", self._on_key_release)
        self._start_gamepad()

    # ── Gamepad (evdev) ───────────────────────────────────────────────────────
    def _start_gamepad(self):
        if getattr(self, "_gamepad_running", False):
            return
        self._gamepad_running = True
        threading.Thread(target=self._gamepad_loop, daemon=True).start()

    def _gamepad_loop(self):
        try:
            from evdev import InputDevice, ecodes, list_devices
        except ImportError:
            print("[GAMEPAD] evdev not installed: pip install evdev")
            self._gamepad_running = False
            return

        # Find first gamepad/joystick
        device = None
        for path in list_devices():
            try:
                d = InputDevice(path)
                caps = d.capabilities()
                if ecodes.EV_ABS in caps or ecodes.EV_KEY in caps:
                    name = d.name.lower()
                    if any(w in name for w in ("gamepad","joystick","controller",
                                               "xbox","ps4","ps5","ds4","dualshock",
                                               "logitech","8bitdo","pro controller")):
                        device = d
                        break
            except Exception:
                continue

        # If no named controller found, take first device with ABS axes
        if device is None:
            for path in list_devices():
                try:
                    d = InputDevice(path)
                    if ecodes.EV_ABS in d.capabilities():
                        device = d; break
                except Exception:
                    continue

        if device is None:
            print("[GAMEPAD] no controller found")
            self._gamepad_running = False
            return

        print(f"[GAMEPAD] connected: {device.name}")
        self.after(0, lambda n=device.name: self._key_indicator.set(f"🎮 {n}"))

        # Get servos and motors in order
        servos = [c for c in self.app.components if c["type"] == "servo"]
        motors = [c for c in self.app.components if c["type"] == "motor"]
        lasers = [c for c in self.app.components if c["type"] == "laser"]

        # Axis dead zone
        DEAD = 5000
        MAX  = 32767

        axis_state = {}

        try:
            for event in device.read_loop():
                if not self._gamepad_running:
                    break

                # ── Analog axes → servos and motors ──────────────────────────
                if event.type == ecodes.EV_ABS:
                    val  = event.value
                    code = event.code
                    axis_state[code] = val

                    # Left stick X (ABS_X=0) → first servo
                    if code == ecodes.ABS_X and servos:
                        if abs(val - MAX // 2) > DEAD:
                            norm = (val - MAX // 2) / (MAX // 2)  # -1 to +1
                            direction = 1 if norm > 0 else -1
                            self.after(0, lambda c=servos[0], d=direction:
                                       self._kb_servo(c, d))

                    # Left stick Y (ABS_Y=1) → first motor forward/back
                    elif code == ecodes.ABS_Y and motors:
                        if abs(val - MAX // 2) > DEAD:
                            norm  = (val - MAX // 2) / (MAX // 2)
                            speed = int(-norm * 100)  # push up = forward
                            self.after(0, lambda c=motors[0], s=speed:
                                       self._kb_motor(c, s))
                        else:
                            self.after(0, lambda c=motors[0]:
                                       self._kb_motor(c, 0))

                    # Right stick X (ABS_RX=3) → second servo if exists
                    elif code == ecodes.ABS_RX and len(servos) > 1:
                        if abs(val - MAX // 2) > DEAD:
                            norm = (val - MAX // 2) / (MAX // 2)
                            direction = 1 if norm > 0 else -1
                            self.after(0, lambda c=servos[1], d=direction:
                                       self._kb_servo(c, d))

                    # Right stick Y (ABS_RY=4) → second motor if exists
                    elif code == ecodes.ABS_RY and len(motors) > 1:
                        if abs(val - MAX // 2) > DEAD:
                            norm  = (val - MAX // 2) / (MAX // 2)
                            speed = int(-norm * 100)
                            self.after(0, lambda c=motors[1], s=speed:
                                       self._kb_motor(c, s))
                        else:
                            self.after(0, lambda c=motors[1]:
                                       self._kb_motor(c, 0))

                    # Right trigger (ABS_RZ=5) → laser 0 toggle
                    elif code == ecodes.ABS_RZ and lasers:
                        if val > 200 and not axis_state.get("rz_held"):
                            axis_state["rz_held"] = True
                            self.after(0, lambda c=lasers[0]: self._kb_laser(c))
                        elif val <= 200:
                            axis_state["rz_held"] = False

                # ── Buttons ───────────────────────────────────────────────────
                elif event.type == ecodes.EV_KEY and event.value == 1:
                    code = event.code

                    # South button (A/Cross) → reset all servos
                    if code == ecodes.BTN_SOUTH:
                        for s in servos:
                            self.after(0, lambda c=s: self._kb_servo_reset(c))

                    # North button (Y/Triangle) → toggle tracking
                    elif code == ecodes.BTN_NORTH and self.cam_comps:
                        self.after(0, self._toggle_tracking)

                    # East (B/Circle) → stop all motors
                    elif code == ecodes.BTN_EAST:
                        for m in motors:
                            self.after(0, lambda c=m: self._kb_motor(c, 0))

                    # West (X/Square) → toggle laser 1
                    elif code == ecodes.BTN_WEST and len(lasers) > 1:
                        self.after(0, lambda c=lasers[1]: self._kb_laser(c))

                    # Start → toggle keyboard
                    elif code == ecodes.BTN_START:
                        self.after(0, self._toggle_keyboard)

        except Exception as e:
            print(f"[GAMEPAD] disconnected: {e}")
        self._gamepad_running = False

    def _toggle_keyboard(self):
        self._kb_active.set(not self._kb_active.get())
        if self._kb_active.get():
            self._kb_btn.config(text="[KB] ON", bg=GREEN)
        else:
            self._kb_btn.config(text="[KB] OFF", bg=BORDER)
            self._key_indicator.set("")

    def _on_key_press(self, event):
        if not self._kb_active.get():
            return
        if isinstance(event.widget, (tk.Entry, tk.Text)):
            return
        key = event.keysym.lower()
        if key in self._held_keys:
            return
        self._held_keys.add(key)
        if key in self._keymap:
            label = " | ".join(a[1] for a in self._keymap[key])
            self._key_indicator.set(f"[{key.upper()}] {label}")
            for fn, _ in self._keymap[key]:
                try:    fn()
                except Exception as e: print(f"[KB] {e}")

    def _on_key_release(self, event):
        key = event.keysym.lower()
        self._held_keys.discard(key)
        # Auto-stop motors on key release
        for comp in self.app.components:
            if comp["type"] == "motor":
                if key in (comp.get("key_fwd","").lower(),
                           comp.get("key_bwd","").lower()):
                    self._kb_motor(comp, 0)
        if not self._held_keys:
            self.after(300, lambda: self._key_indicator.set("") if not self._held_keys else None)

    def _kb_servo(self, comp, direction):
        name = comp["name"]
        cur  = self._servo_angles.get(name, 90.0)
        new  = max(comp["min_deg"], min(comp["max_deg"],
                   cur + direction * comp["step"]))
        self._servo_angles[name] = new
        servo_set(comp["pin"], new, comp["pulse_min"], comp["pulse_max"])

    def _kb_servo_reset(self, comp):
        self._servo_angles[comp["name"]] = 90.0
        servo_set(comp["pin"], 90.0, comp["pulse_min"], comp["pulse_max"])

    def _kb_motor(self, comp, speed):
        self._motor_speeds[comp["name"]] = speed
        motor_set(comp["pin_fwd"], comp["pin_bwd"], speed)

    def _kb_laser(self, comp):
        name = comp["name"]
        new  = not self._laser_states.get(name, False)
        self._laser_states[name] = new
        laser_set(comp["pin"], new)

    # ── Camera thread ─────────────────────────────────────────────────────
    def _start_local_vision(self, cam_comp):
        """Lazily loads whichever local vision models are enabled for this
        camera, and stores config on self for _cam_loop to use."""
        self._lv_detect_faces    = cam_comp.get("detect_faces", False)
        self._lv_recognize_faces = cam_comp.get("recognize_faces", False)
        self._lv_detect_objects  = cam_comp.get("detect_objects", False)
        self._lv_classify_scene  = cam_comp.get("classify_scene", False)
        self._lv_detect_custom_images = cam_comp.get("detect_custom_images", False)
        self._lv_detect_pose     = cam_comp.get("detect_pose", False)
        self._lv_known_faces_dir = cam_comp.get("known_faces_dir", "known_faces")
        self._lv_custom_images_dir = cam_comp.get("custom_images_dir", "reference_images")
        self._lv_last_heavy      = 0.0   # throttle timestamp for YOLO/matching

        if self._lv_detect_objects or self._lv_classify_scene or self._lv_detect_pose:
            _try_load_yolo(need_classify=self._lv_classify_scene,
                           need_pose=self._lv_detect_pose)

        if self._lv_recognize_faces:
            if _try_load_face_recognizer():
                if not _train_face_recognizer(self._lv_known_faces_dir):
                    self._vision_log_line(
                        f"no known faces yet in '{self._lv_known_faces_dir}/' "
                        f"— use ADD KNOWN FACE")

        if self._lv_detect_custom_images:
            _load_reference_images(self._lv_custom_images_dir)
            if not _ref_images:
                self._vision_log_line(
                    f"no reference images in '{self._lv_custom_images_dir}/' "
                    f"— add named photos there (e.g. apple.jpg, my_car.png)")

        self._lv_last_obj  = 0.0
        self._lv_last_cls  = 0.0
        self._lv_last_custom = 0.0
        self._lv_last_pose = 0.0
        self._lv_last_face_logged = None

    def _vision_log_line(self, text, hit=False):
        if not hasattr(self, "_vision_log"):
            return
        def _do():
            log = self._vision_log
            log.configure(state="normal")
            log.insert("end", text + "\n", ("hit",) if hit else ())
            log.see("end")
            if int(log.index("end-1c").split(".")[0]) > 300:
                log.delete("1.0", "2.0")
            log.configure(state="disabled")
        self.after(0, _do)

    def _add_known_face(self):
        if not hasattr(self, "_latest_frame") or self._latest_frame is None:
            messagebox.showwarning("No Frame", "No camera frame available yet.")
            return
        name = simpledialog.askstring("Add Known Face",
                                       "Name for this person:", parent=self)
        if not name:
            return
        with self._frame_lock:
            frame = self._latest_frame.copy() if self._latest_frame is not None else None
        if frame is None:
            return

        crop = frame
        if HAS_CV2 and HAS_FACE_CASCADE:
            gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
            if len(faces):
                x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
                crop = frame[y:y+h, x:x+w]

        person_dir = os.path.join(self._lv_known_faces_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        path = os.path.join(person_dir, f"img_{int(time.time())}.jpg")
        cv2.imwrite(path, crop)

        if _try_load_face_recognizer():
            _train_face_recognizer(self._lv_known_faces_dir)
        self._vision_log_line(f"[FACE] saved sample for '{name}' — recognizer retrained", hit=True)

    def _start_camera(self):
        _try_load_cv2()
        if not self.cam_comps or not HAS_CV2 or not HAS_PIL:
            return
        self._cam_thread = threading.Thread(target=self._cam_loop, daemon=True)
        self._cam_thread.start()

    def _cam_loop(self):
        idx = self.cam_comps[0]["index"]

        # Try to open camera with retries
        cap = None
        for attempt in range(5):
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                break
            cap.release()
            time.sleep(1.0)

        if not cap or not cap.isOpened():
            # Try index 0 as fallback
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print(f"[CAM] ERROR: Could not open camera index {idx}")
                return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce latency

        print(f"[CAM] Camera opened on index {idx}")

        while self._running:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.05)
                continue

            want_faces = (self._tracking or getattr(self, "_lv_detect_faces", False)
                          or getattr(self, "_lv_recognize_faces", False))

            # Face detection / tracking / recognition ────────────────────
            # ONE detection pass feeds all three features. Boxes are always
            # drawn when detection runs (no separate "draw" toggle), and the
            # tracked target is highlighted distinctly instead of getting a
            # second box drawn on top of it.
            best, offset = None, 0
            if want_faces and HAS_CV2 and HAS_FACE_CASCADE:
                gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
                best_area = 0
                for (x, y, w, h) in faces:
                    if w*h > best_area:
                        best_area, best = w*h, (x, y, w, h)

                    label, color = None, (0, 255, 80)
                    if getattr(self, "_lv_recognize_faces", False) and HAS_FACE_REC and FACE_LABELS:
                        face_crop = cv2.resize(gray[y:y+h, x:x+w], (200, 200))
                        try:
                            lbl_id, conf = FACE_RECOGNIZER.predict(face_crop)
                            if conf < 80:  # lower = better match for LBPH
                                label = FACE_LABELS.get(lbl_id, "?")
                            else:
                                label, color = "unknown", (0, 120, 255)
                        except Exception:
                            pass
                        if label and label != self._lv_last_face_logged:
                            self._vision_log_line(f"[FACE] {label}", hit=(label != "unknown"))
                            self._lv_last_face_logged = label

                    _draw_bracket_box(frame, x, y, x+w, y+h, color, label=label)

                if best:
                    bx, by, bw, bh = best
                    cx = bx + bw//2
                    # Centre from the ACTUAL frame width, not a hardcoded 160 —
                    # cameras don't always honour the requested capture size,
                    # which would silently skew servo aim.
                    frame_cx = frame.shape[1] // 2
                    offset = cx - frame_cx
                    if self._tracking:
                        # Highlight the locked target: outer cyan bracket +
                        # centre dot, layered around the normal box above.
                        _draw_bracket_box(frame, bx - 5, by - 5, bx + bw + 5,
                                          by + bh + 5, (0, 255, 255),
                                          label="LOCKED")
                        cv2.circle(frame, (cx, by + bh//2), 4, (0, 255, 255), -1)

            # Object detection (throttled — YOLO is heavy) ─────────────────
            now = time.time()
            if not hasattr(self, "_lv_obj_cache"):
                self._lv_obj_cache = []   # persists between throttled runs
            if getattr(self, "_lv_detect_objects", False) and HAS_YOLO \
                    and now - self._lv_last_obj > 1.0:
                self._lv_last_obj = now
                try:
                    results = YOLO_DETECT_MODEL(frame, verbose=False)[0]
                    seen, cache = set(), []
                    for box in results.boxes:
                        cls_id = int(box.cls[0])
                        name   = results.names.get(cls_id, str(cls_id))
                        conf   = float(box.conf[0])
                        if conf < 0.4:
                            continue
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cache.append((x1, y1, x2, y2, name, conf))
                        seen.add(name)
                    self._lv_obj_cache = cache
                    if seen:
                        self._vision_log_line(f"[OBJ] {', '.join(sorted(seen))}", hit=True)
                except Exception as e:
                    print(f"[VISION] object detection error: {e}")

            # Redraw the last known object boxes on every frame (not just the
            # throttled detection frame) so they don't flicker for 1 in N frames.
            for x1, y1, x2, y2, name, conf in self._lv_obj_cache:
                _draw_bracket_box(frame, x1, y1, x2, y2, (255, 150, 30),
                                  label=f"{name} {conf:.2f}")

            # Scene classification (throttled) ──────────────────────────────
            if getattr(self, "_lv_classify_scene", False) and HAS_YOLO \
                    and now - self._lv_last_cls > 2.0:
                self._lv_last_cls = now
                try:
                    results = YOLO_CLASSIFY_MODEL(frame, verbose=False)[0]
                    top1 = int(results.probs.top1)
                    conf = float(results.probs.top1conf)
                    name = results.names.get(top1, str(top1))
                    self._vision_log_line(f"[SCENE] {name} ({conf:.0%})")
                except Exception as e:
                    print(f"[VISION] classification error: {e}")

            # Custom image recognition (throttled, color+ORB feature match) ─
            if not hasattr(self, "_lv_custom_cache"):
                self._lv_custom_cache = None   # persists between throttled runs
            if getattr(self, "_lv_detect_custom_images", False) and HAS_CV2 \
                    and now - self._lv_last_custom > 2.0:
                self._lv_last_custom = now
                match = _match_reference_image(frame)
                if match:
                    name, score, corners = match
                    self._lv_custom_cache = (name, score, corners)
                    self._vision_log_line(f"[MATCH] {name} ({score:.0%})", hit=True)
                else:
                    self._lv_custom_cache = None

            # Redraw the last known match box every frame so it doesn't
            # flicker for 1 in N frames.
            if self._lv_custom_cache:
                name, score, corners = self._lv_custom_cache
                if corners is not None:
                    pts = corners.astype(int).reshape(-1, 1, 2)
                    cv2.polylines(frame, [pts], True, (255, 80, 220), 2)
                    x, y = corners[0]
                    cv2.putText(frame, f"{name} {score:.0%}", (int(x), max(12, int(y) - 8)),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 220), 1)
                else:
                    # No reliable location found — still surface that a
                    # match exists, just without a box (small badge instead).
                    cv2.putText(frame, f"[MATCH] {name} {score:.0%}", (8, 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 80, 220), 1)

            # Pose / skeleton tracking (throttled — YOLO-pose is heavy) ─────
            if not hasattr(self, "_lv_pose_cache"):
                self._lv_pose_cache = []   # persists between throttled runs
            if getattr(self, "_lv_detect_pose", False) and HAS_YOLO \
                    and now - self._lv_last_pose > 0.5:
                self._lv_last_pose = now
                try:
                    results = YOLO_POSE_MODEL(frame, verbose=False)[0]
                    n_people, cache = 0, []
                    for person_kpts in results.keypoints.xy:
                        pts = person_kpts.tolist()
                        if not pts or all(x == 0 and y == 0 for x, y in pts):
                            continue
                        n_people += 1
                        cache.append(pts)
                    self._lv_pose_cache = cache
                    if n_people:
                        self._vision_log_line(f"[POSE] {n_people} person(s) tracked", hit=True)
                except Exception as e:
                    print(f"[VISION] pose estimation error: {e}")

            # Redraw the last known skeleton(s) on every frame so they don't
            # flicker for 1 in N frames.
            for pts in self._lv_pose_cache:
                valid_pts = [(x, y) for x, y in pts if not (x == 0 and y == 0)]
                if valid_pts:
                    xs = [p[0] for p in valid_pts]
                    ys = [p[1] for p in valid_pts]
                    _draw_bracket_box(frame, int(min(xs)) - 6, int(min(ys)) - 6,
                                      int(max(xs)) + 6, int(max(ys)) + 6,
                                      (0, 220, 255), label="person")
                for x, y in pts:
                    if x == 0 and y == 0:
                        continue
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 220, 255), -1)
                for a, b in POSE_SKELETON:
                    if a >= len(pts) or b >= len(pts):
                        continue
                    xa, ya = pts[a]
                    xb, yb = pts[b]
                    if (xa, ya) == (0, 0) or (xb, yb) == (0, 0):
                        continue
                    cv2.line(frame, (int(xa), int(ya)), (int(xb), int(yb)),
                             (0, 220, 255), 2)

            # Always draw crosshair
            cv2.line(frame, (148, 120), (172, 120), (80, 80, 255), 1)
            cv2.line(frame, (160, 108), (160, 132), (80, 80, 255), 1)

            with self._frame_lock:
                self._latest_frame = frame.copy()
                self._face_offset  = offset
                self._face_found   = best is not None

        cap.release()
        print("[CAM] Camera released")

    # ── UI loop ───────────────────────────────────────────────────────────
    def _ui_loop(self):
        if not self._running:
            return

        # Update camera canvas - ALWAYS show feed if camera exists
        if self.cam_canvas and HAS_PIL and HAS_CV2:
            with self._frame_lock:
                frame  = self._latest_frame
                offset = self._face_offset
                found  = self._face_found

            if frame is not None:
                rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img   = Image.fromarray(rgb)
                canvas_w = int(self.cam_canvas["width"])
                canvas_h = int(self.cam_canvas["height"])
                if img.size != (canvas_w, canvas_h):
                    img = img.resize((canvas_w, canvas_h))
                photo = ImageTk.PhotoImage(img)
                self.cam_canvas.create_image(0, 0, anchor="nw", image=photo)
                self.cam_canvas._photo = photo

            # Face tracking - move ALL linked servos (only when tracking on)
            if self._tracking and self._track_servo_comps and found:
                cam_comp = self.cam_comps[0]
                dz = cam_comp.get("dead_zone", 30)
                ts = cam_comp.get("track_step", 2)
                if abs(offset) > dz:
                    direction = 1 if offset > 0 else -1
                    for c in self._track_servo_comps:
                        cur = self._servo_angles.get(c["name"], 90.0)
                        new = max(c["min_deg"], min(c["max_deg"], cur + direction * ts))
                        self._servo_angles[c["name"]] = new
                        servo_set(c["pin"], new, c["pulse_min"], c["pulse_max"])

            # Status label - always update
            if hasattr(self, "track_status"):
                if self._tracking:
                    self.track_status.set("# TRACKING ON" + (" - [*] LOCKED" if found else " - searching..."))
                    self._track_status_lbl.config(fg=GREEN)
                else:
                    self.track_status.set("* TRACKING  OFF")
                    self._track_status_lbl.config(fg=MUTED)

        self.after(33, self._ui_loop)  # ~30fps

    def _toggle_ai(self):
        if self._ai_panel.winfo_ismapped():
            self._ai_panel.pack_forget()
        else:
            self._ai_panel.pack(side="right", fill="y", padx=(12, 0))

    def _toggle_vision_auto(self):
        self._vision_auto_on.set(not self._vision_auto_on.get())
        if self._vision_auto_on.get():
            self._vis_auto_btn.config(text="AUTO ON", bg=GREEN)
            self._schedule_vision()
        else:
            self._vis_auto_btn.config(text="AUTO OFF", bg=BORDER)

    def _schedule_vision(self):
        if self._running and hasattr(self, "_vision_auto_on") and self._vision_auto_on.get():
            self._vision_analyze_once()
            self.after(int(self._vision_interval * 1000), self._schedule_vision)

    def _vision_analyze_once(self):
        if not HAS_CV2 or not hasattr(self, "_vision_busy") or self._vision_busy:
            return
        with self._frame_lock:
            frame = self._latest_frame
        if frame is None:
            return
        self._vision_busy = True
        if hasattr(self, "_vision_result"):
            self._vision_result.set("🧠 analyzing...")

        def _run():
            try:
                import base64
                _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                b64 = base64.b64encode(buf.tobytes()).decode()
                payload = json.dumps({
                    "model": self._vision_model,
                    "messages": [{
                        "role": "user",
                        "content": [
                            {"type": "image_url",
                             "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                            {"type": "text", "text": self._vision_prompt}
                        ]
                    }],
                    "stream": False
                }).encode()
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/chat", data=payload,
                    headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(req, timeout=30) as resp:
                    data = json.loads(resp.read())
                    text = data.get("message", {}).get("content", "No response").strip()
                    self.after(0, lambda t=text: (
                        self._vision_result.set(t) if hasattr(self, "_vision_result") else None
                    ))
                    # Pass to AI panel if connected
                    if hasattr(self, "_ai_panel") and self._ai_panel._connected:
                        self._ai_panel._history.append({
                            "role": "user",
                            "content": f"[Vision sees]: {text}\nShould any components react?"
                        })
            except Exception as e:
                self.after(0, lambda err=str(e): (
                    self._vision_result.set(f"Error: {err}") if hasattr(self, "_vision_result") else None
                ))
            finally:
                self._vision_busy = False

        threading.Thread(target=_run, daemon=True).start()

    def _toggle_tracking(self):
        self._tracking = not self._tracking
        if self._tracking:
            self.btn_track.config(text="#  STOP TRACKING", bg=RED)
        else:
            self.btn_track.config(text="#  START TRACKING", bg=GREEN)

    def _release_mic(self):
        """Stops the mic monitor and kills the arecord subprocess so the
        audio device is freed immediately, rather than waiting for the
        monitor thread to notice on its next read."""
        self._mic_monitor_running = False
        proc = getattr(self, "_mic_proc", None)
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            self._mic_proc = None

    def _back(self):
        self._running = False
        self._release_mic()
        self.app.show_builder()

    def destroy(self):
        self._running = False
        self._release_mic()
        super().destroy()


# ═════════════════════════════════════════════════════════════════════════════
#  OLLAMA AI CONTROLLER
# ═════════════════════════════════════════════════════════════════════════════
OLLAMA_URL = "http://localhost:11434"

def ollama_chat(model: str, messages: list, timeout: int = 120) -> str:
    """Send chat to Ollama, return response text or error string."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "stream": False
    }).encode()
    try:
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
            return data.get("message", {}).get("content", "").strip()
    except urllib.error.URLError as e:
        return f"ERROR: Cannot reach Ollama - make sure 'ollama serve' is running. ({e.reason})"
    except TimeoutError:
        return "ERROR: Ollama timed out - model may be too slow or not loaded yet."
    except Exception as e:
        return f"ERROR: {e}"


def ollama_list_models() -> list:
    """Return list of installed model names."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
            return [m["name"] for m in data.get("models", [])]
    except Exception:
        return []


def build_system_prompt(components: list) -> str:
    """Generate a system prompt describing the robot's components."""
    lines = [
        "You are an AI controller for a physical robot built with a Raspberry Pi.",
        "You control hardware components by responding ONLY with a JSON command object.",
        "Never include explanation text - only output valid JSON.",
        "",
        "Available components:",
    ]
    for c in components:
        if c["type"] == "servo":
            lines.append(f'  - Servo named "{c["name"]}" on GPIO pin {c["pin"]}, range {c["min_deg"]}° to {c["max_deg"]}°')
        elif c["type"] == "motor":
            lines.append(f'  - Motor named "{c["name"]}" on GPIO fwd={c["pin_fwd"]} bwd={c["pin_bwd"]}, speed -100 to 100')
        elif c["type"] == "laser":
            lines.append(f'  - Laser named "{c["name"]}" on GPIO pin {c["pin"]}')
        elif c["type"] == "camera":
            lines.append(f'  - Camera named "{c["name"]}" (index {c["index"]}) with face tracking')
        elif c["type"] == "sound":
            files = [os.path.basename(f) for f in c.get("files", [])]
            lines.append(f'  - Sound player named "{c["name"]}" with files: {", ".join(files) or "none"}')
        elif c["type"] == "sensor":
            st = c.get("sensor_type", "?")
            lines.append(f'  - Sensor named "{c["name"]}" type={st} (read-only, you can react to its data)')

    lines += [
        "",
        "Respond ONLY with a JSON object like this:",
        '{',
        '  "actions": [',
        '    {"type": "servo", "name": "Servo", "angle": 45},',
        '    {"type": "motor", "name": "Motor", "speed": 80},',
        '    {"type": "laser", "name": "Laser", "state": true},',
        '    {"type": "tracking", "state": true},',
        '    {"type": "sound", "name": "Sounds", "file": "alarm.mp3"},',
        '    {"type": "speak", "text": "Done!"}',
        '  ]',
        '}',
        "",
        "Rules:",
        "- Use exact component names as listed above.",
        "- For servos: angle must be within their min/max range.",
        "- For motors: speed is -100 (full reverse) to 100 (full forward), 0 = stop.",
        "- For laser: state is true (on) or false (off).",
        "- For tracking: state true = start face tracking, false = stop.",
        "- 'speak' type adds a message shown to the user.",
        "- You can combine multiple actions in one response.",
        "- If the command is unclear, use speak to ask for clarification.",
        "- Never output anything except the JSON object.",
    ]
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════════════════
#  AI CHAT PANEL  (embedded in ControlPage)
# ═════════════════════════════════════════════════════════════════════════════
class AIChatPanel(tk.Frame):
    def __init__(self, parent, control_page):
        super().__init__(parent, bg=PANEL,
                         highlightthickness=1, highlightbackground=AI_FG)
        self.ctrl   = control_page
        self._model = tk.StringVar(value="")
        self._connected = False
        self._history   = []   # Ollama message history
        self._thinking  = False

        f_title = tkfont.Font(family="Courier", size=12, weight="bold")
        f_med   = tkfont.Font(family="Courier", size=11)
        f_small = tkfont.Font(family="Courier", size=10)

        # ── Header ────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=AI_BG)
        hdr.pack(fill="x")
        tk.Label(hdr, text="[AI] AI CONTROLLER",
                 bg=AI_BG, fg=AI_FG, font=f_title).pack(side="left", padx=10, pady=8)
        tk.Button(hdr, text="✕", bg=AI_BG, fg=MUTED,
                  font=f_small, relief="flat", cursor="hand2",
                  command=self._hide).pack(side="right", padx=8)

        # ── Model selector ────────────────────────────────────────────────
        model_row = tk.Frame(self, bg=PANEL)
        model_row.pack(fill="x", padx=10, pady=(8, 4))

        tk.Label(model_row, text="Model:", bg=PANEL, fg=MUTED,
                 font=f_small).pack(side="left")

        self._model_entry = tk.Entry(model_row, textvariable=self._model,
                                      bg=CARD, fg=FG, insertbackground=FG,
                                      font=f_small, relief="flat", width=16)
        self._model_entry.pack(side="left", padx=6)
        self._model_entry.insert(0, "tinyllama")

        self._connect_btn = tk.Button(model_row, text="CONNECT",
                                       bg=BORDER, fg=FG,
                                       activebackground=AI_FG,
                                       font=f_small, relief="flat", cursor="hand2",
                                       padx=8, command=self._connect)
        self._connect_btn.pack(side="left")

        # Model dropdown (discovered from Ollama)
        self._model_dd = ttk.Combobox(model_row, width=14,
                                       font=f_small, state="readonly")
        self._model_dd.pack(side="left", padx=6)
        self._model_dd.bind("<<ComboboxSelected>>",
                             lambda e: self._model.set(self._model_dd.get()))

        tk.Button(model_row, text="⟳", bg=BORDER, fg=MUTED,
                  font=f_small, relief="flat", cursor="hand2",
                  command=self._refresh_models).pack(side="left")

        # Status
        self._status_var = tk.StringVar(value="o Not connected")
        self._status_lbl = tk.Label(self, textvariable=self._status_var,
                                     bg=PANEL, fg=MUTED, font=f_small)
        self._status_lbl.pack(anchor="w", padx=10)

        # ── Chat log ──────────────────────────────────────────────────────
        log_frame = tk.Frame(self, bg=PANEL)
        log_frame.pack(fill="both", expand=True, padx=10, pady=4)

        self._log = tk.Text(log_frame, bg=CARD, fg=FG,
                             font=f_small, relief="flat",
                             wrap="word", state="disabled",
                             height=14, width=32)
        log_sb = tk.Scrollbar(log_frame, command=self._log.yview)
        self._log.configure(yscrollcommand=log_sb.set)
        log_sb.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True)

        # Text tags for colours
        self._log.tag_config("user",   foreground=ACCENT)
        self._log.tag_config("ai",     foreground=AI_FG)
        self._log.tag_config("action", foreground=GREEN)
        self._log.tag_config("error",  foreground=RED)
        self._log.tag_config("think",  foreground=ORANGE)

        # ── Input row ─────────────────────────────────────────────────────
        inp_row = tk.Frame(self, bg=PANEL)
        inp_row.pack(fill="x", padx=10, pady=(4, 10))

        self._inp = tk.Entry(inp_row, bg=CARD, fg=FG,
                              insertbackground=FG, font=f_med,
                              relief="flat")
        self._inp.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self._inp.bind("<Return>", lambda e: self._send())

        self._send_btn = tk.Button(inp_row, text="SEND",
                                    bg=AI_FG, fg=BG,
                                    activebackground=ACCENT,
                                    font=f_small, relief="flat", cursor="hand2",
                                    padx=10, command=self._send)
        self._send_btn.pack(side="right")

        # Quick command buttons
        quick_frame = tk.Frame(self, bg=PANEL)
        quick_frame.pack(fill="x", padx=10, pady=(0, 6))
        for txt in ["look left", "look right", "center", "fire laser",
                    "start tracking", "stop tracking", "stop all"]:
            tk.Button(quick_frame, text=txt, bg=BORDER, fg=MUTED,
                      font=tkfont.Font(family="Courier", size=9),
                      relief="flat", cursor="hand2", padx=4, pady=2,
                      command=lambda t=txt: self._quick(t)
                      ).pack(side="left", padx=2, pady=2)

        # Refresh models on startup
        self._refresh_models()

    def _hide(self):
        self.pack_forget()

    def _log_msg(self, text, tag="ai"):
        self._log.configure(state="normal")
        self._log.insert("end", text + "\n", tag)
        self._log.see("end")
        self._log.configure(state="disabled")

    def _refresh_models(self):
        def _fetch():
            models = ollama_list_models()
            if models:
                self._model_dd["values"] = models
                self._log_msg(f"Found {len(models)} model(s): {', '.join(models)}", "action")
            else:
                self._log_msg("Could not reach Ollama - is it running?", "error")
        threading.Thread(target=_fetch, daemon=True).start()

    def _connect(self):
        model = self._model.get().strip()
        if not model:
            self._log_msg("Enter a model name first!", "error")
            return
        self._status_var.set("o Connecting...")
        self._status_lbl.config(fg=ORANGE)
        self._connect_btn.config(state="disabled", text="...")

        def _try():
            # Just check if Ollama API is reachable - no model inference needed
            try:
                req = urllib.request.Request(
                    f"{OLLAMA_URL}/api/tags", method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    data   = json.loads(resp.read())
                    models = [m["name"] for m in data.get("models", [])]

                # Build system prompt
                sys_prompt = build_system_prompt(self.ctrl.app.components)
                self._history  = [{"role": "system", "content": sys_prompt}]
                self._connected = True

                model_found = any(model in m for m in models)
                warn = "" if model_found else f"\n⚠ '{model}' not found in installed models - run: ollama pull {model}"

                self.after(0, lambda: [
                    self._connect_btn.config(state="normal", bg=GREEN, text="CONNECTED ✓"),
                    self._status_var.set(f"* Connected - {model}"),
                    self._status_lbl.config(fg=GREEN),
                    self._log_msg(f"[AI] Connected! Ready to control your robot.", "action"),
                    self._log_msg(f"Try: 'look left', 'fire laser', 'start tracking'", "think"),
                    self._log_msg(warn, "error") if warn else None,
                ])

            except Exception as e:
                self._connected = False
                self.after(0, lambda: [
                    self._connect_btn.config(state="normal", bg=BORDER, text="CONNECT"),
                    self._status_var.set("✕ Failed"),
                    self._status_lbl.config(fg=RED),
                    self._log_msg(f"ERROR: {e}", "error"),
                    self._log_msg("[!] Make sure Ollama is running: ollama serve", "think"),
                ])

        threading.Thread(target=_try, daemon=True).start()

    def _quick(self, text):
        self._inp.delete(0, "end")
        self._inp.insert(0, text)
        self._send()

    def _send(self):
        if not self._connected:
            self._log_msg("Connect to Ollama first!", "error")
            return
        if self._thinking:
            return
        text = self._inp.get().strip()
        if not text:
            return
        self._inp.delete(0, "end")
        self._log_msg(f"You: {text}", "user")
        self._history.append({"role": "user", "content": text})
        self._thinking = True
        self._send_btn.config(state="disabled", text="...")
        self._log_msg("[AI] thinking...", "think")

        def _run():
            model  = self._model.get().strip()
            resp   = ollama_chat(model, self._history)
            self._history.append({"role": "assistant", "content": resp})
            self.after(0, lambda: self._handle_response(resp))

        threading.Thread(target=_run, daemon=True).start()

    def _handle_response(self, resp: str):
        self._thinking = False
        self._send_btn.config(state="normal", text="SEND")

        # Remove "thinking..." line
        self._log.configure(state="normal")
        content = self._log.get("1.0", "end")
        lines = content.split("\n")
        lines = [l for l in lines if l != "[AI] thinking..."]
        self._log.delete("1.0", "end")
        for l in lines:
            self._log.insert("end", l + "\n")
        self._log.configure(state="disabled")

        # Try to parse JSON
        try:
            # Strip markdown fences if model adds them
            clean = resp.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            elif clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            data  = json.loads(clean)
            actions = data.get("actions", [])
            self._execute_actions(actions)
        except json.JSONDecodeError:
            self._log_msg(f"[AI] {resp}", "ai")

    def _execute_actions(self, actions: list):
        for action in actions:
            t = action.get("type")

            if t == "speak":
                self._log_msg(f"[AI] {action.get('text', '')}", "ai")

            elif t == "servo":
                name  = action.get("name")
                angle = action.get("angle")
                comp  = self._find_comp("servo", name)
                if comp and angle is not None:
                    angle = max(comp["min_deg"], min(comp["max_deg"], float(angle)))
                    self.ctrl._servo_angles[name] = angle
                    servo_set(comp["pin"], angle, comp["pulse_min"], comp["pulse_max"])
                    self._log_msg(f"[S] {name} -> {angle:.0f}°", "action")
                else:
                    self._log_msg(f"✕ Servo '{name}' not found", "error")

            elif t == "motor":
                name  = action.get("name")
                speed = action.get("speed", 0)
                comp  = self._find_comp("motor", name)
                if comp:
                    motor_set(comp["pin_fwd"], comp["pin_bwd"], int(speed))
                    self.ctrl._motor_speeds[name] = int(speed)
                    self._log_msg(f"[M] {name} speed -> {speed}%", "action")
                else:
                    self._log_msg(f"✕ Motor '{name}' not found", "error")

            elif t == "laser":
                name  = action.get("name")
                state = action.get("state", False)
                comp  = self._find_comp("laser", name)
                if comp:
                    laser_set(comp["pin"], state)
                    self.ctrl._laser_states[name] = state
                    self._log_msg(f"[L] {name} -> {'ON' if state else 'OFF'}", "action")
                else:
                    self._log_msg(f"✕ Laser '{name}' not found", "error")

            elif t == "tracking":
                state = action.get("state", False)
                self.ctrl._tracking = state
                if hasattr(self.ctrl, "btn_track"):
                    self.ctrl.btn_track.config(
                        text="#  STOP TRACKING" if state else "#  START TRACKING",
                        bg=RED if state else GREEN)
                self._log_msg(f"[C] Face tracking -> {'ON' if state else 'OFF'}", "action")

            elif t == "sound":
                name  = action.get("name")
                fname = action.get("file", "")
                comp  = self._find_comp("sound", name)
                if comp and HAS_PYGAME:
                    # Find matching file
                    match = next((f for f in comp.get("files", [])
                                  if os.path.basename(f) == fname or fname in f), None)
                    if match:
                        try:
                            pygame.mixer.music.load(match)
                            pygame.mixer.music.play()
                            self._log_msg(f"[A] Playing: {os.path.basename(match)}", "action")
                        except Exception as e:
                            self._log_msg(f"✕ Sound error: {e}", "error")
                    else:
                        self._log_msg(f"✕ File '{fname}' not found in {name}", "error")
                else:
                    self._log_msg(f"✕ Sound component '{name}' not found", "error")

    def _find_comp(self, ctype, name):
        for c in self.ctrl.app.components:
            if c["type"] == ctype and c["name"] == name:
                return c
        return None
class SentryRigBuilder(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentry Rig Builder")
        self.configure(bg=BG)
        self._auto_size_window()
        self._style_ttk_dark()

        self.components: list  = []
        self._current_page = None

        self.show_builder()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _style_ttk_dark(self):
        """ttk widgets (Combobox etc.) don't obey plain bg=/fg= like classic
        tk widgets — they render via the OS's native theme, which is why
        every Combobox in the app was showing up light/white regardless of
        the dark colors set everywhere else. This forces one dark ttk theme
        for the whole app instead of leaving it to the OS default."""
        style = ttk.Style(self)
        try:
            style.theme_use("clam")   # 'clam' is themeable via .configure(); the
        except tk.TclError:            # OS-native themes mostly ignore custom colors
            pass
        style.configure("TCombobox",
                        fieldbackground=CARD, background=CARD,
                        foreground=FG, arrowcolor=FG,
                        selectbackground=CARD, selectforeground=FG,
                        bordercolor=BORDER, lightcolor=CARD, darkcolor=CARD)
        style.map("TCombobox",
                 fieldbackground=[("readonly", CARD)],
                 selectbackground=[("readonly", CARD)],
                 foreground=[("readonly", FG)])
        # The dropdown listbox itself is a separate Tk widget the style
        # can't reach directly — but tk_setPalette + the option database
        # below is what the popdown listbox actually reads its colors from.
        self.option_add("*TCombobox*Listbox.background", CARD)
        self.option_add("*TCombobox*Listbox.foreground", FG)
        self.option_add("*TCombobox*Listbox.selectBackground", BORDER)
        self.option_add("*TCombobox*Listbox.selectForeground", FG)

    def _auto_size_window(self):
        """Sizes and centers the window relative to whatever screen it's
        actually running on, instead of a hardcoded 1100x650 that gets
        clipped on smaller displays (like Pi touchscreens)."""
        global COMPACT
        self.update_idletasks()
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()

        # 5" Pi touchscreens are commonly 800x480 (or smaller). Below that
        # threshold, switch every page to compact fonts/padding so nothing
        # gets clipped instead of just shrinking the window and hoping.
        COMPACT = screen_w <= 850 or screen_h <= 500

        # Use most of the screen, but never larger than a sane desktop size
        target_w = min(int(screen_w * 0.95 if COMPACT else screen_w * 0.92), 1300)
        target_h = min(int(screen_h * 0.92 if COMPACT else screen_h * 0.88), 800)

        # Never shrink below a size the UI actually needs to render without
        # clipping the header buttons — but cap that floor to the screen
        # itself so this never overflows a genuinely small display.
        min_w = min(480, screen_w - 10)
        min_h = min(320, screen_h - 10)
        target_w = max(target_w, min_w)
        target_h = max(target_h, min_h)

        x = max(0, (screen_w - target_w) // 2)
        y = max(0, (screen_h - target_h) // 2)

        self.geometry(f"{target_w}x{target_h}+{x}+{y}")
        self.minsize(min_w, min_h)

    def show_builder(self):
        if self._current_page:
            self._current_page.destroy()
        self._current_page = BuilderPage(self, self)
        self._current_page.pack(fill="both", expand=True)

    def launch(self):
        if not self.components:
            messagebox.showwarning("No Components",
                                   "Add at least one component before launching!")
            return
        if self._current_page:
            self._current_page.destroy()
        self._current_page = ControlPage(self, self)
        self._current_page.pack(fill="both", expand=True)

    def save_build(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("Sentry Build", "*.json"), ("All", "*.*")],
            title="Save Build")
        if path:
            with open(path, "w") as f:
                json.dump({
                    "components":   self.components,
                }, f, indent=2)
            messagebox.showinfo("Saved", f"Build saved!")

    def load_build(self):
        path = filedialog.askopenfilename(
            filetypes=[("Sentry Build", "*.json"), ("All", "*.*")],
            title="Load Build")
        if path:
            with open(path) as f:
                data = json.load(f)
            self.components   = data.get("components", [])
            if isinstance(self._current_page, BuilderPage):
                self._current_page.refresh()
            messagebox.showinfo("Loaded", f"Loaded {len(self.components)} component(s).")

    def _on_close(self):
        cleanup_all()
        self.destroy()


if __name__ == "__main__":
    app = SentryRigBuilder()
    app.mainloop()
