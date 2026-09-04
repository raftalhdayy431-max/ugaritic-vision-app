import os
import shutil
import threading
import time
from pathlib import Path
import cv2
import numpy as np

from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.uix.slider import Slider
from kivy.core.clipboard import Clipboard

from kivymd.app import MDApp
from kivymd.uix.screen import MDScreen
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import (
    MDFlatButton,
    MDRoundFlatButton,
    MDFillRoundFlatButton,
    MDRoundFlatIconButton,
    MDFillRoundFlatIconButton,
)
from kivymd.uix.label import MDLabel
from kivymd.uix.dialog import MDDialog
from kivymd.uix.scrollview import MDScrollView
from kivymd.uix.card import MDCard

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:
    import tensorflow as tf
    Interpreter = tf.lite.Interpreter
# ====================================================================
# OPTIONAL FILE CHOOSER
# ====================================================================

try:
    from plyer import filechooser
    plyer_available = True
except Exception:
    filechooser = None
    plyer_available = False


# ====================================================================
# BASE PATHS & CACHING
# ====================================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "bestfinetuneing.tflite"
CUSTOM_FONT_PATH = BASE_DIR / "fonts" / "NotoSansUgaritic-Regular.ttf"

_FONT_CACHE = {}

def get_cached_font(path, size):
    key = (str(path), int(size))
    if key not in _FONT_CACHE:
        try:
            _FONT_CACHE[key] = ImageFont.truetype(str(path), int(size))
        except Exception:
            _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


# ====================================================================
# CONFIGURATION DEFAULTS
# ====================================================================

DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_IOU_THRESHOLD = 0.7
DEFAULT_MAX_DET = 40
DEFAULT_FPS = 5
DEFAULT_BOX_THICKNESS = 2
DEFAULT_LABEL_FONT_SIZE = 13
WORD_DIVIDER_CLASS = 30


# ====================================================================
# UGARITIC ALPHABET DEFINITION
# ====================================================================

UGARITIC_CHARS = [
    '𐎀', '𐎁', '𐎂', '𐎃', '𐎄', '𐎅', '𐎆', '𐎇', '𐎈', '𐎉',
    '𐎊', '𐎋', '𐎌', '𐎍', '𐎎', '𐎏', '𐎐', '𐎑', '𐎒', '𐎓',
    '𐎔', '𐎕', '𐎖', '𐎗', '𐎘', '𐎙', '𐎚', '𐎛', '𐎜', '𐎝',
    '𐎟'
]


# ====================================================================
# MATH & GEOMETRY HELPERS
# ====================================================================

def sigmoid(x):
    x = np.asarray(x, dtype=np.float32)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def letterbox(im, new_shape=(256, 256), color=(114, 114, 114), scaleup=True):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2.0
    dh = (new_shape[0] - new_unpad[1]) / 2.0

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, float(r), (float(dw), float(dh))


def xywh_to_xyxy(boxes):
    boxes = np.asarray(boxes, dtype=np.float32)
    result = np.empty_like(boxes, dtype=np.float32)
    result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return result


def iou_xyxy(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])

    return inter / (area1 + area2 - inter + 1e-7)


def classwise_nms(boxes, scores, class_ids, iou_threshold, max_det):
    if len(boxes) == 0:
        return []

    keep = []
    for cls in np.unique(class_ids):
        indexes = np.where(class_ids == cls)[0]
        indexes = indexes[np.argsort(scores[indexes])[::-1]]

        while len(indexes) > 0:
            current = int(indexes[0])
            keep.append(current)
            if len(indexes) == 1:
                break
            rest = indexes[1:]
            ious = iou_xyxy(boxes[current], boxes[rest])
            indexes = rest[ious <= float(iou_threshold)]

    keep.sort(key=lambda i: float(scores[i]), reverse=True)
    return keep[:int(max_det)]


# ====================================================================
# OCR TEXT EXTRACTION & ANNOTATION
# ====================================================================

def build_detections(boxes, scores, class_ids, frame):
    orig_h, orig_w = frame.shape[:2]
    detections = []

    for box, score, class_id in zip(boxes, scores, class_ids):
        x1 = max(0.0, min(float(orig_w - 1), float(box[0])))
        y1 = max(0.0, min(float(orig_h - 1), float(box[1])))
        x2 = max(0.0, min(float(orig_w - 1), float(box[2])))
        y2 = max(0.0, min(float(orig_h - 1), float(box[3])))

        width, height = x2 - x1, y2 - y1
        if width < 2.0 or height < 2.0:
            continue

        detections.append({
            "class_id": int(class_id),
            "confidence": float(score),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "cx": (x1 + x2) / 2.0, "cy": (y1 + y2) / 2.0,
            "width": width, "height": height
        })

    return detections


def group_into_lines(detections):
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["cy"])
    lines = []

    for det in detections:
        added = False
        for line in lines:
            avg_y = np.mean([d["cy"] for d in line])
            avg_h = np.mean([d["height"] for d in line])
            tolerance = max(8.0, avg_h * 0.5)

            if abs(det["cy"] - avg_y) <= tolerance:
                line.append(det)
                added = True
                break

        if not added:
            lines.append([det])

    lines.sort(key=lambda line: np.mean([d["cy"] for d in line]))
    return lines


def extract_ugaritic_text(detections):
    if not detections:
        return ""

    lines = group_into_lines(detections)
    text_lines = []

    for line in lines:
        line = sorted(line, key=lambda d: d["cx"])
        text = ""
        for det in line:
            cid = int(det["class_id"])
            if cid == WORD_DIVIDER_CLASS:
                text += " "
            elif 0 <= cid < len(UGARITIC_CHARS):
                text += UGARITIC_CHARS[cid]

        text = " ".join(text.split())
        if text:
            text_lines.append(text)

    return "\n".join(text_lines)


def draw_ugaritic_annotations(image, detections, font_path, font_size=13, box_thickness=2, show_boxes=True, show_labels=True, show_confidence=True):
    if image is None or not detections:
        return image.copy() if image is not None else None

    output = image.copy()
    h_img, w_img = output.shape[:2]

    rgb_img = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    pil_image = PILImage.fromarray(rgb_img)
    draw = ImageDraw.Draw(pil_image, "RGBA")

    ugaritic_font = get_cached_font(font_path, font_size)
    ascii_font = ImageFont.load_default()

    for det in detections:
        x1, y1 = int(round(det["x1"])), int(round(det["y1"]))
        x2, y2 = int(round(det["x2"])), int(round(det["y2"]))
        cid = int(det["class_id"])
        confidence = float(det["confidence"])

        box_color = (239, 83, 80, 255) if confidence < 0.40 else (0, 180, 216, 255)
        badge_bg = (239, 83, 80, 220) if confidence < 0.40 else (0, 180, 216, 220)

        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=box_color, width=box_thickness)

        if show_confidence or show_labels:
            char = UGARITIC_CHARS[cid] if 0 <= cid < len(UGARITIC_CHARS) else "?"
            if cid == WORD_DIVIDER_CLASS:
                char = "|"

            conf_str = f"{int(confidence * 100)}%" if show_confidence else ""
            display_str = f"{char} {conf_str}".strip() if show_labels else conf_str

            pad_x, pad_y = 4, 2
            badge_w = max(26, len(display_str) * 8 + pad_x * 2)
            badge_h = 18

            lbl_x = max(0, min(x1, w_img - badge_w))
            lbl_y = max(0, y1 - badge_h)

            draw.rectangle([lbl_x, lbl_y, lbl_x + badge_w, lbl_y + badge_h], fill=badge_bg)

            if show_labels and show_confidence:
                draw.text((lbl_x + pad_x, lbl_y + 1), char, font=ugaritic_font, fill=(255, 255, 255, 255))
                draw.text((lbl_x + pad_x + 14, lbl_y + 2), conf_str, font=ascii_font, fill=(255, 255, 255, 255))
            elif show_confidence:
                draw.text((lbl_x + pad_x, lbl_y + 2), conf_str, font=ascii_font, fill=(255, 255, 255, 255))
            elif show_labels:
                draw.text((lbl_x + pad_x, lbl_y + 1), char, font=ugaritic_font, fill=(255, 255, 255, 255))

    return cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)


def restore_boxes_from_letterbox(boxes, frame, ratio, dw, dh):
    boxes = np.asarray(boxes, dtype=np.float32).copy()
    if len(boxes) == 0:
        return boxes

    orig_h, orig_w = frame.shape[:2]
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - float(dw)) / max(float(ratio), 1e-12)
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - float(dh)) / max(float(ratio), 1e-12)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(0, orig_w - 1))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(0, orig_h - 1))
    return boxes


def post_process_yolo(outputs, frame, conf_threshold, iou_threshold, imgsz=256, max_det=40, ratio=None, dw=None, dh=None, font_size=13, box_thickness=2, show_boxes=True, show_labels=True, show_confidence=True):
    empty_res = (frame.copy(), "")
    if outputs is None:
        return empty_res

    if isinstance(outputs, (list, tuple)):
        outputs = np.asarray(outputs[0], dtype=np.float32)
    else:
        outputs = np.asarray(outputs, dtype=np.float32)

    outputs = np.asarray(outputs, dtype=np.float32)
    if outputs.ndim == 3 and outputs.shape[0] == 1:
        outputs = outputs[0]

    expected_channels = 4 + len(UGARITIC_CHARS)
    if outputs.ndim == 2 and outputs.shape[0] == expected_channels and outputs.shape[1] != expected_channels:
        outputs = outputs.T

    if outputs.ndim != 2 or outputs.shape[1] != expected_channels:
        return empty_res

    boxes_raw = outputs[:, :4].copy()
    class_scores = outputs[:, 4:].copy()

    if len(boxes_raw) > 0 and np.nanmax(np.abs(boxes_raw)) <= 2.0:
        boxes_raw[:, [0, 2]] *= float(imgsz)
        boxes_raw[:, [1, 3]] *= float(imgsz)

    finite_scores = class_scores[np.isfinite(class_scores)]
    if finite_scores.size == 0:
        return empty_res

    if float(np.min(finite_scores)) < 0.0 or float(np.max(finite_scores)) > 1.0:
        probabilities = sigmoid(class_scores)
    else:
        probabilities = np.clip(class_scores, 0.0, 1.0)

    class_ids = np.argmax(probabilities, axis=1).astype(np.int32)
    scores = probabilities[np.arange(len(probabilities)), class_ids]
    boxes = xywh_to_xyxy(boxes_raw)

    keep_conf = scores >= float(conf_threshold)
    boxes, scores, class_ids = boxes[keep_conf], scores[keep_conf], class_ids[keep_conf]

    if len(scores) == 0:
        return empty_res

    if ratio is None or dw is None or dh is None:
        _, ratio, (dw, dh) = letterbox(frame, (imgsz, imgsz), scaleup=True)

    boxes = restore_boxes_from_letterbox(boxes, frame, ratio, dw, dh)

    keep = classwise_nms(boxes, scores, class_ids, iou_threshold, max_det)
    if not keep:
        return empty_res

    boxes, scores, class_ids = boxes[keep], scores[keep], class_ids[keep]
    valid_detections = build_detections(boxes, scores, class_ids, frame)

    extracted_text = extract_ugaritic_text(valid_detections)
    annotated_frame = draw_ugaritic_annotations(
        frame, valid_detections, str(CUSTOM_FONT_PATH),
        font_size=font_size, box_thickness=box_thickness,
        show_boxes=show_boxes, show_labels=show_labels, show_confidence=show_confidence
    )

    return annotated_frame, extracted_text


# ====================================================================
# MODEL LOADING
# ====================================================================

def load_inference_model(path):
    path = Path(path).resolve()

    if not path.exists():
        print(f"❌ TFLite model not found: {path}")
        return None

    try:
        with open(path, "rb") as f:
            model_content = f.read()

        interpreter = tf.lite.Interpreter(
            model_content=model_content,
            num_threads=2
        )

        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        print("\n================ TFLITE MODEL ================")

        for i, detail in enumerate(input_details):
            print(
                f"INPUT {i}: "
                f"shape={detail['shape']}, "
                f"dtype={detail['dtype']}, "
                f"quantization={detail.get('quantization')}"
            )

        for i, detail in enumerate(output_details):
            print(
                f"OUTPUT {i}: "
                f"shape={detail['shape']}, "
                f"dtype={detail['dtype']}, "
                f"quantization={detail.get('quantization')}"
            )

        print("===============================================\n")

        return interpreter

    except Exception as e:
        print("❌ Failed to load TFLite:", repr(e))
        return None

# ====================================================================
# MAIN APPLICATION
# ====================================================================

class UgariticStudioApp(MDApp):

    def build(self):
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Cyan"
        self.title = "Ugaritic Vision AI Studio"

        self._capture = None
        self._session = None
        self._thread = None

        self._is_running = True
        self._data_lock = threading.Lock()
        self.dialog = None
        self.text_dialog = None

        self.conf_threshold = DEFAULT_CONF_THRESHOLD
        self.iou_threshold = DEFAULT_IOU_THRESHOLD
        self.img_size = 256
        self.max_det = DEFAULT_MAX_DET
        self.target_fps = DEFAULT_FPS
        self.box_thickness = DEFAULT_BOX_THICKNESS
        self.font_size_label = DEFAULT_LABEL_FONT_SIZE

        self.show_boxes = True
        self.show_labels = False
        self.show_confidence = True

        self._inference_source = "camera"
        self._current_image_path = None
        self._last_extracted_text = ""
        self._frozen_frame = None
        self._latest_valid_frame = None
        self._last_annotated_image = None
        self._force_recompute = False  # متغير فرض إعادة الاستدلال

        self._load_model()
        return self.build_ui()

    def _load_model(self):
        self._session = load_inference_model(MODEL_PATH)

    def build_ui(self):
        screen = MDScreen(md_bg_color=(0.04, 0.05, 0.08, 1))
        main_layout = MDBoxLayout(orientation="vertical", spacing="0dp", padding="0dp")

        # 1. TOP HEADER APP BAR
        header = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height="52dp",
            padding=["16dp", "4dp", "16dp", "4dp"], spacing="10dp", md_bg_color=(0.08, 0.10, 0.14, 1)
        )

        title_label = MDLabel(
            text="UGARITIC VISION AI", halign="left", font_style="Subtitle1",
            bold=True, theme_text_color="Custom", text_color=(0.92, 0.95, 0.98, 1)
        )

        self.status_chip = MDCard(
            size_hint=(None, None), size=("120dp", "28dp"), radius=[14],
            md_bg_color=(0.0, 0.7, 0.85, 0.2), elevation=0, pos_hint={"center_y": 0.5}
        )
        self.status_label = MDLabel(
            text="LIVE CAMERA", halign="center", theme_text_color="Custom",
            text_color=(0.0, 0.7, 0.85, 1), font_style="Caption", bold=True
        )
        self.status_chip.add_widget(self.status_label)

        header.add_widget(title_label)
        header.add_widget(self.status_chip)
        main_layout.add_widget(header)

        # 2. NAVIGATION TABS BAR
        tabs_layout = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height="44dp",
            md_bg_color=(0.06, 0.08, 0.11, 1), padding=["8dp", "2dp", "8dp", "2dp"], spacing="6dp"
        )

        self.btn_tab_visual = MDRoundFlatIconButton(
            icon="eye", text="Visual Mode", font_size="11sp",
            icon_color=(0.0, 0.7, 0.85, 1), theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1),
            line_color=(0.0, 0.7, 0.85, 1), md_bg_color=(0.0, 0.7, 0.85, 0.15)
        )

        self.btn_tab_text = MDRoundFlatIconButton(
            icon="text-box-outline", text="Text View", font_size="11sp",
            icon_color=(0.6, 0.65, 0.7, 1), theme_text_color="Custom", text_color=(0.6, 0.65, 0.7, 1),
            line_color=(0.2, 0.25, 0.3, 1), md_bg_color=(0, 0, 0, 0)
        )
        self.btn_tab_text.bind(on_release=lambda x: self.open_text_dialog())

        self.btn_tab_settings = MDRoundFlatIconButton(
            icon="tune", text="Parameters", font_size="11sp",
            icon_color=(0.6, 0.65, 0.7, 1), theme_text_color="Custom", text_color=(0.6, 0.65, 0.7, 1),
            line_color=(0.2, 0.25, 0.3, 1), md_bg_color=(0, 0, 0, 0)
        )
        self.btn_tab_settings.bind(on_release=lambda x: self.open_settings_dialog())

        tabs_layout.add_widget(self.btn_tab_visual)
        tabs_layout.add_widget(self.btn_tab_text)
        tabs_layout.add_widget(self.btn_tab_settings)
        main_layout.add_widget(tabs_layout)

        # 3. SPLIT VIEW MAIN CONTAINER (HERO CAMERA VIEWPORT)
        self.content_container = MDBoxLayout(orientation="vertical", padding="6dp", spacing="6dp")

        self.image_card = MDCard(
            orientation="vertical", size_hint_y=0.92, elevation=3,
            radius=[16], md_bg_color=(0.02, 0.03, 0.05, 1), padding="2dp"
        )
        self.image_widget = Image(allow_stretch=True, keep_ratio=True)
        self.image_card.add_widget(self.image_widget)
        self.content_container.add_widget(self.image_card)

        chips_row = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height="30dp",
            spacing="6dp", pos_hint={"center_x": 0.5}
        )

        self.chip_boxes = MDRoundFlatIconButton(
            icon="vector-square", text="Boxes", font_size="10sp",
            icon_color=(0.0, 0.7, 0.85, 1), theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1),
            line_color=(0.0, 0.7, 0.85, 1), md_bg_color=(0.0, 0.7, 0.85, 0.15)
        )
        self.chip_boxes.bind(on_release=lambda x: self.toggle_filter("boxes"))

        self.chip_conf = MDRoundFlatIconButton(
            icon="percent-outline", text="Confidence", font_size="10sp",
            icon_color=(0.0, 0.7, 0.85, 1), theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1),
            line_color=(0.0, 0.7, 0.85, 1), md_bg_color=(0.0, 0.7, 0.85, 0.15)
        )
        self.chip_conf.bind(on_release=lambda x: self.toggle_filter("conf"))

        self.chip_chars = MDRoundFlatIconButton(
            icon="translate", text="Ugaritic Labels", font_size="10sp",
            icon_color=(0.5, 0.55, 0.6, 1), theme_text_color="Custom", text_color=(0.5, 0.55, 0.6, 1),
            line_color=(0.2, 0.25, 0.3, 1), md_bg_color=(0, 0, 0, 0)
        )
        self.chip_chars.bind(on_release=lambda x: self.toggle_filter("chars"))

        chips_row.add_widget(self.chip_boxes)
        chips_row.add_widget(self.chip_conf)
        chips_row.add_widget(self.chip_chars)
        self.content_container.add_widget(chips_row)

        main_layout.add_widget(self.content_container)

        # 4. ACTION BAR (FLOATING CONTROL DECK)
        bottom_bar = MDBoxLayout(
            orientation="horizontal", size_hint_y=None, height="56dp",
            padding=["10dp", "4dp", "10dp", "6dp"], spacing="8dp", md_bg_color=(0.08, 0.10, 0.14, 1)
        )

        btn_analyze = MDFillRoundFlatIconButton(
            icon="flash", text="ANALYZE", font_size="11sp", size_hint_x=0.4, height="40dp",
            icon_color=(1, 1, 1, 1), md_bg_color=(1.0, 0.6, 0.1, 1)
        )
        btn_analyze.bind(on_release=self.freeze_and_detect)

        btn_next = MDFillRoundFlatIconButton(
            icon="camera", text="LIVE CAMERA", font_size="11sp", size_hint_x=0.35, height="40dp",
            icon_color=(1, 1, 1, 1), md_bg_color=(0.0, 0.7, 0.85, 1)
        )
        btn_next.bind(on_release=self.switch_to_camera)

        btn_gallery = MDFillRoundFlatIconButton(
            icon="folder-image", text="GALLERY", font_size="11sp", size_hint_x=0.25, height="40dp",
            icon_color=(0.9, 0.9, 0.9, 1), md_bg_color=(0.15, 0.20, 0.28, 1), text_color=(0.9, 0.9, 0.9, 1)
        )
        btn_gallery.bind(on_release=self.open_gallery)

        bottom_bar.add_widget(btn_analyze)
        bottom_bar.add_widget(btn_next)
        bottom_bar.add_widget(btn_gallery)

        main_layout.add_widget(bottom_bar)
        screen.add_widget(main_layout)

        # Thread Initialization
        self._thread = threading.Thread(target=self._inference_loop, daemon=True)
        self._thread.start()

        return screen

# ====================================================================
# TFLITE INFERENCE
# ====================================================================

    def run_tflite_inference(self, interpreter, blob):
        if interpreter is None:
            return None

        try:
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            if not input_details or not output_details:
                return None

            input_detail = input_details[0]
            input_index = input_detail["index"]
            expected_shape = tuple(input_detail["shape"])
            expected_dtype = input_detail["dtype"]

            input_data = np.asarray(blob, dtype=np.float32)

            if tuple(input_data.shape) != expected_shape:
                if (
                    len(expected_shape) == 4
                    and expected_shape[0] == 1
                    and expected_shape[1] == 3
                ):
                    input_data = input_data.reshape(expected_shape)
                elif (
                    len(expected_shape) == 4
                    and expected_shape[0] == 1
                    and expected_shape[3] == 3
                ):
                    input_data = np.transpose(input_data, (0, 2, 3, 1))
                else:
                    return None

            if expected_dtype == np.float32:
                input_tensor = input_data.astype(np.float32)
            elif expected_dtype in (np.uint8, np.int8):
                scale, zero_point = input_detail.get("quantization", (0.0, 0))
                if scale is None or scale == 0:
                    return None
                input_tensor = np.round(input_data / scale + zero_point).astype(expected_dtype)
            else:
                input_tensor = input_data.astype(expected_dtype)

            interpreter.set_tensor(input_index, input_tensor)
            interpreter.invoke()

            output_detail = output_details[0]
            output_index = output_detail["index"]
            output = interpreter.get_tensor(output_index)

            output_dtype = output_detail["dtype"]
            if output_dtype in (np.uint8, np.int8):
                scale, zero_point = output_detail.get("quantization", (0.0, 0))
                if scale is not None and scale != 0:
                    output = (output.astype(np.float32) - float(zero_point)) * float(scale)
            else:
                output = output.astype(np.float32)

            return np.asarray(output, dtype=np.float32)

        except Exception as e:
            print("❌ TFLite inference error:", repr(e))
            return None

    # ====================================================================
    # STANDALONE TEXT VIEW DIALOG
    # ====================================================================

    def open_text_dialog(self, *args):
        if hasattr(self, 'text_dialog') and self.text_dialog:
            try:
                self.text_dialog.dismiss()
            except Exception:
                pass
            self.text_dialog = None

        content = MDBoxLayout(orientation="vertical", size_hint_y=None, spacing="8dp", padding="8dp")
        content.bind(minimum_height=content.setter("height"))

        raw_text = self._last_extracted_text if self._last_extracted_text else ""
        lines = [l.strip() for l in raw_text.split("\n") if l.strip()]

        if lines:
            header_lbl = MDLabel(
                text=f" Number of lines : {len(lines)}", font_style="Caption",
                theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1), bold=True, adaptive_height=True
            )
            content.add_widget(header_lbl)

            for idx, line_text in enumerate(lines, 1):
                line_card = MDCard(
                    orientation="horizontal", size_hint_y=None, height="46dp",
                    padding=["8dp", "4dp", "8dp", "4dp"], spacing="10dp",
                    md_bg_color=(0.08, 0.10, 0.14, 1), elevation=1, radius=[8]
                )

                num_lbl = MDLabel(
                    text=f"#{idx}", size_hint_x=None, width="28dp",
                    theme_text_color="Custom", text_color=(0.5, 0.55, 0.6, 1),
                    font_style="Caption", bold=True, pos_hint={"center_y": 0.5}
                )

                texture, width, height = self.create_ugaritic_text_texture(line_text)
                text_img = Image(
                    texture=texture, size_hint=(None, None), size=(width, height),
                    allow_stretch=False, keep_ratio=True, pos_hint={"center_y": 0.5}
                )

                btn_copy_line = MDRoundFlatIconButton(
                    icon="content-copy", text="copy", font_size="10sp",
                    size_hint=(None, None), height="30dp",
                    icon_color=(0.0, 0.7, 0.85, 1), theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1),
                    line_color=(0.0, 0.7, 0.85, 0.4), pos_hint={"center_y": 0.5}
                )
                btn_copy_line.bind(on_release=lambda x, t=line_text: self.copy_to_clipboard(t))

                line_card.add_widget(num_lbl)
                line_card.add_widget(text_img)
                line_card.add_widget(btn_copy_line)
                content.add_widget(line_card)
        else:
            empty_lbl = MDLabel(
                text="No Ugaritic texts or symbols have been discovered yet.",
                halign="center", theme_text_color="Hint", font_style="Body2",
                adaptive_height=True, padding=["0dp", "20dp", "0dp", "20dp"]
            )
            content.add_widget(empty_lbl)

        scroll = MDScrollView(size_hint=(1, None), height="300dp")
        scroll.add_widget(content)

        buttons = []
        if lines:
            full_copy_text = "\n".join(lines)
            btn_copy_all = MDFillRoundFlatIconButton(
                icon="content-copy", text="Copy the entire text", font_size="11sp",
                md_bg_color=(0.0, 0.7, 0.85, 1),
                on_release=lambda x: self.copy_to_clipboard(full_copy_text)
            )
            buttons.append(btn_copy_all)

        btn_close = MDFlatButton(
            text="close", theme_text_color="Custom", text_color=(0.8, 0.8, 0.8, 1),
            on_release=lambda x: self.text_dialog.dismiss()
        )
        buttons.append(btn_close)

        self.text_dialog = MDDialog(
            title="The extracted text is organized.",
            type="custom",
            content_cls=scroll,
            buttons=buttons
        )
        self.text_dialog.open()

    def update_status_badge(self, mode_text, color_rgb):
        self.status_label.text = mode_text
        self.status_label.text_color = color_rgb
        self.status_chip.md_bg_color = (*color_rgb[:3], 0.2)

    def toggle_filter(self, filter_type):
        with self._data_lock:
            if filter_type == "conf":
                self.show_confidence = not self.show_confidence
                active = self.show_confidence
                self.chip_conf.icon = "check" if active else "percent-outline"
                self.chip_conf.md_bg_color = (0.0, 0.7, 0.85, 0.15) if active else (0, 0, 0, 0)
                self.chip_conf.text_color = self.chip_conf.icon_color = (0.0, 0.7, 0.85, 1) if active else (0.5, 0.55, 0.6, 1)
            elif filter_type == "chars":
                self.show_labels = not self.show_labels
                active = self.show_labels
                self.chip_chars.icon = "check" if active else "translate"
                self.chip_chars.md_bg_color = (0.0, 0.7, 0.85, 0.15) if active else (0, 0, 0, 0)
                self.chip_chars.text_color = self.chip_chars.icon_color = (0.0, 0.7, 0.85, 1) if active else (0.5, 0.55, 0.6, 1)
            elif filter_type == "boxes":
                self.show_boxes = not self.show_boxes
                active = self.show_boxes
                self.chip_boxes.icon = "check" if active else "vector-square"
                self.chip_boxes.md_bg_color = (0.0, 0.7, 0.85, 0.15) if active else (0, 0, 0, 0)
                self.chip_boxes.text_color = self.chip_boxes.icon_color = (0.0, 0.7, 0.85, 1) if active else (0.5, 0.55, 0.6, 1)

            self._force_recompute = True
            self._last_annotated_image = None

    # ====================================================================
    # BACKGROUND INFERENCE THREAD (OPTIMIZED FPS & LATEST FRAME STRATEGY)
    # ====================================================================

    def _inference_loop(self):
        last_img_path = last_conf = last_iou = last_imgsz = last_max_det = last_source = None
        last_thickness = last_font_sz = last_show_conf = last_show_boxes = last_show_labels = None
        cached_image_frame = None
        image_processed = False
        freeze_processed = False

        while self._is_running:
            start_time = time.time()
            try:
                with self._data_lock:
                    source = self._inference_source
                    conf = float(self.conf_threshold)
                    iou = float(self.iou_threshold)
                    img_size = int(self.img_size)
                    max_det = int(self.max_det)
                    fps = int(self.target_fps)
                    box_thick = int(self.box_thickness)
                    font_sz = int(self.font_size_label)
                    show_conf = bool(self.show_confidence)
                    show_bx = bool(self.show_boxes)
                    show_lbl = bool(self.show_labels)
                    img_path = self._current_image_path
                    frozen_target = None if self._frozen_frame is None else self._frozen_frame.copy()
                    force_recomp = self._force_recompute
                    # Reset the flag after reading
                    self._force_recompute = False

                if source != last_source:
                    last_source = source
                    if source == "camera":
                        Clock.schedule_once(lambda dt: self.update_status_badge("LIVE CAMERA", (0.0, 0.7, 0.85, 1)))
                    elif source == "freeze":
                        Clock.schedule_once(lambda dt: self.update_status_badge("ANALYZED FRAME", (1.0, 0.6, 0.1, 1)))
                    elif source == "image":
                        Clock.schedule_once(lambda dt: self.update_status_badge("STATIC IMAGE", (0.1, 0.8, 0.4, 1)))

                frame_to_process = None
                run_inference = False

                if source == "camera":
                    freeze_processed = False
                    image_processed = False
                    if self._capture is None or not self._capture.isOpened():
                        try:
                            self._capture = cv2.VideoCapture(0)
                        except Exception:
                            self._capture = None
                        time.sleep(0.1)

                    if self._capture and self._capture.isOpened():
                        for _ in range(2):
                            ret, frame = self._capture.read()
                        if ret and frame is not None:
                            frame_to_process = frame
                            with self._data_lock:
                                self._latest_valid_frame = frame.copy()
                            run_inference = True
                        else:
                            try: self._capture.release()
                            except Exception: pass
                            self._capture = None

                    if frame_to_process is None:
                        frame_to_process = self._get_placeholder_frame("Camera Stream Unavailable")

                elif source == "freeze":
                    if self._capture and self._capture.isOpened():
                        try: self._capture.release()
                        except Exception: pass
                        self._capture = None

                    changed = (force_recomp or conf != last_conf or iou != last_iou or img_size != last_imgsz or 
                               max_det != last_max_det or box_thick != last_thickness or font_sz != last_font_sz or 
                               show_conf != last_show_conf or show_bx != last_show_boxes or show_lbl != last_show_labels or 
                               self._last_annotated_image is None or not freeze_processed)

                    if frozen_target is not None:
                        if changed:
                            frame_to_process = frozen_target
                            run_inference = True
                            freeze_processed = True
                        else:
                            frame_to_process = self._last_annotated_image if self._last_annotated_image is not None else frozen_target
                            run_inference = False
                    else:
                        frame_to_process = self._get_placeholder_frame("No Frozen Frame")

                elif source == "image":
                    freeze_processed = False
                    if self._capture and self._capture.isOpened():
                        try: self._capture.release()
                        except Exception: pass
                        self._capture = None

                    changed = (force_recomp or img_path != last_img_path or conf != last_conf or iou != last_iou or 
                               img_size != last_imgsz or max_det != last_max_det or box_thick != last_thickness or
                               font_sz != last_font_sz or show_conf != last_show_conf or
                               show_bx != last_show_boxes or show_lbl != last_show_labels or not image_processed or
                               self._last_annotated_image is None)

                    if changed:
                        cached_image_frame = None
                        if img_path and os.path.exists(img_path):
                            try:
                                data = np.fromfile(img_path, dtype=np.uint8)
                                cached_image_frame = cv2.imdecode(data, cv2.IMREAD_COLOR)
                            except Exception:
                                cached_image_frame = None

                        if cached_image_frame is None:
                            cached_image_frame = self._get_placeholder_frame("Image Load Error")

                        h, w = cached_image_frame.shape[:2]
                        if max(h, w) > 1280:
                            scale = 1280 / max(h, w)
                            cached_image_frame = cv2.resize(cached_image_frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

                        frame_to_process = cached_image_frame
                        run_inference = True
                        image_processed = True
                    else:
                        frame_to_process = self._last_annotated_image if self._last_annotated_image is not None else cached_image_frame
                        run_inference = False

                (last_img_path, last_conf, last_iou, last_imgsz, last_max_det, 
                 last_thickness, last_font_sz, last_show_conf, last_show_boxes, last_show_labels) = (
                    img_path, conf, iou, img_size, max_det, box_thick, font_sz, show_conf, show_bx, show_lbl
                )

                if frame_to_process is not None:
                    annotated_frame = frame_to_process.copy()
                    extracted_text = self._last_extracted_text if not run_inference else ""

                    if self._session is not None and run_inference:
                        try:
                            processed_img, ratio, (dw, dh) = letterbox(
                                frame_to_process,
                                (img_size, img_size),
                                color=(114, 114, 114),
                                scaleup=True
                            )

                            processed_rgb = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
                            blob = processed_rgb.astype(np.float32) / 255.0
                            blob = np.transpose(blob, (2, 0, 1))
                            blob = np.expand_dims(blob, axis=0)
                            blob = np.ascontiguousarray(blob)

                            raw_outputs = self.run_tflite_inference(self._session, blob)

                            annotated_frame, extracted_text = post_process_yolo(
                                raw_outputs,
                                frame_to_process,
                                conf,
                                iou,
                                imgsz=img_size,
                                max_det=max_det,
                                ratio=ratio,
                                dw=dw,
                                dh=dh,
                                font_size=font_sz,
                                box_thickness=box_thick,
                                show_boxes=show_bx,
                                show_labels=show_lbl,
                                show_confidence=show_conf
                            )

                            self._last_annotated_image = annotated_frame.copy()
                            self._last_extracted_text = extracted_text

                        except Exception as e:
                            print("❌ INFERENCE LOOP ERROR:", repr(e))
                            annotated_frame = frame_to_process.copy()

                    elif source in ["image", "freeze"] and self._last_annotated_image is not None:
                        annotated_frame = self._last_annotated_image

                    Clock.schedule_once(lambda dt, f=annotated_frame: self.update_frame_texture(f))

                elapsed = time.time() - start_time
                target_interval = 1.0 / max(1, fps) if source == "camera" else 0.05
                sleep_time = target_interval - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

            except Exception:
                time.sleep(0.05)

    def update_frame_texture(self, frame):
        try:
            if frame is None: return
            frame = np.ascontiguousarray(frame)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            flipped = cv2.flip(rgb_frame, 0)
            buffer = flipped.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt="rgb")
            texture.blit_buffer(buffer, colorfmt="rgb", bufferfmt="ubyte")
            self.image_widget.texture = texture
        except Exception:
            pass

    def _get_placeholder_frame(self, text):
        frame = np.full((480, 640, 3), 20, dtype=np.uint8)
        cv2.putText(frame, text, (140, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (120, 130, 140), 2, cv2.LINE_AA)
        return frame

    def create_ugaritic_text_texture(self, text_str):
        font = get_cached_font(CUSTOM_FONT_PATH, 22)
        try:
            bbox = font.getbbox(text_str)
            text_width = bbox[2] - bbox[0]
        except Exception:
            text_width = len(text_str) * 18

        width, height = max(120, text_width + 30), 40
        image = PILImage.new("RGB", (width, height), color=(15, 20, 28))
        draw = ImageDraw.Draw(image)
        draw.text((10, 4), text_str, fill=(0, 180, 216), font=font)
        image = image.transpose(PILImage.FLIP_TOP_BOTTOM)

        texture = Texture.create(size=(width, height), colorfmt="rgb")
        texture.blit_buffer(image.tobytes(), colorfmt="rgb", bufferfmt="ubyte")
        return texture, width, height

    def copy_to_clipboard(self, text):
        Clipboard.copy(text)
        self.show_toast("تم نسخ النص إلى الحافظة")

    def open_settings_dialog(self, *args):
        if self.dialog:
            try: self.dialog.dismiss()
            except Exception: pass
            self.dialog = None

        content = MDBoxLayout(orientation="vertical", size_hint_y=None, spacing="10dp", padding="10dp")
        content.bind(minimum_height=content.setter("height"))

        with self._data_lock:
            current_conf, current_iou = self.conf_threshold, self.iou_threshold
            current_fps, current_thick = self.target_fps, self.box_thickness

        label_conf = MDLabel(text=f"Confidence Threshold: {current_conf:.2f}", adaptive_height=True, theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1))
        slider_conf = Slider(min=0.05, max=0.95, value=current_conf, step=0.05, size_hint_y=None, height="30dp")
        slider_conf.bind(value=lambda instance, val: self.on_setting_change(val, label_conf, "conf"))
        content.add_widget(label_conf)
        content.add_widget(slider_conf)

        label_iou = MDLabel(text=f"IoU Threshold: {current_iou:.2f}", adaptive_height=True, theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1))
        slider_iou = Slider(min=0.05, max=0.95, value=current_iou, step=0.05, size_hint_y=None, height="30dp")
        slider_iou.bind(value=lambda instance, val: self.on_setting_change(val, label_iou, "iou"))
        content.add_widget(label_iou)
        content.add_widget(slider_iou)

        label_fps = MDLabel(text=f"Frame Rate (FPS): {current_fps}", adaptive_height=True, theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1))
        slider_fps = Slider(min=5, max=30, value=current_fps, step=1, size_hint_y=None, height="30dp")
        slider_fps.bind(value=lambda instance, val: self.on_setting_change(val, label_fps, "fps"))
        content.add_widget(label_fps)
        content.add_widget(slider_fps)

        label_thick = MDLabel(text=f"Bounding Box Thickness: {current_thick} px", adaptive_height=True, theme_text_color="Custom", text_color=(0.9, 0.9, 0.9, 1))
        slider_thick = Slider(min=1, max=4, value=current_thick, step=1, size_hint_y=None, height="30dp")
        slider_thick.bind(value=lambda instance, val: self.on_setting_change(val, label_thick, "box_thick"))
        content.add_widget(label_thick)
        content.add_widget(slider_thick)

        scroll = MDScrollView(size_hint=(1, None), height="260dp")
        scroll.add_widget(content)

        self.dialog = MDDialog(
            title="Vision AI Parameters",
            type="custom", content_cls=scroll,
            buttons=[
                MDFlatButton(text="Apply", theme_text_color="Custom", text_color=(0.0, 0.7, 0.85, 1), on_release=lambda x: self.dialog.dismiss()),
                MDFlatButton(text="Reset Defaults", theme_text_color="Custom", text_color=(1.0, 0.6, 0.1, 1), on_release=self.reset_settings)
            ]
        )
        self.dialog.open()

    def on_setting_change(self, value, label_widget, setting_type):
        if setting_type == "conf":
            with self._data_lock: self.conf_threshold = float(value)
            label_widget.text = f"Confidence Threshold: {float(value):.2f}"
        elif setting_type == "iou":
            with self._data_lock: self.iou_threshold = float(value)
            label_widget.text = f"IoU Threshold: {float(value):.2f}"
        elif setting_type == "fps":
            with self._data_lock: self.target_fps = int(value)
            label_widget.text = f"Frame Rate (FPS): {int(value)}"
        elif setting_type == "box_thick":
            with self._data_lock: self.box_thickness = int(value)
            label_widget.text = f"Bounding Box Thickness: {int(value)} px"

        with self._data_lock:
            self._force_recompute = True
        self._last_annotated_image = None

    def reset_settings(self, *args):
        with self._data_lock:
            self.conf_threshold = DEFAULT_CONF_THRESHOLD
            self.iou_threshold = DEFAULT_IOU_THRESHOLD
            self.img_size = 256
            self.max_det = DEFAULT_MAX_DET
            self.target_fps = DEFAULT_FPS
            self.box_thickness = DEFAULT_BOX_THICKNESS
            self.font_size_label = DEFAULT_LABEL_FONT_SIZE
            self._force_recompute = True

        self._last_annotated_image = None
        if self.dialog:
            self.dialog.dismiss()
            self.dialog = None
        self.show_toast("Default parameters restored")

    def show_toast(self, message):
        try:
            from kivymd.uix.snackbar import MDSnackbar, MDSnackbarText
            MDSnackbar(MDSnackbarText(text=message)).open()
        except Exception:
            print("📢", message)

    def open_gallery(self, *args):
        if not plyer_available:
            self.show_toast("File chooser unavailable")
            return

        try:
            filechooser.open_file(
                title="Select Ugaritic Inscription Image",
                filters=[("Image Files", "*.jpg"), ("Image Files", "*.png"), ("Image Files", "*.jpeg"), ("Image Files", "*.bmp")],
                on_selection=self.handle_file_selection
            )
        except Exception:
            self.show_toast("Cannot open image gallery")

    def handle_file_selection(self, selection):
        if selection and len(selection) > 0:
            with self._data_lock:
                self._current_image_path = selection[0]
                self._inference_source = "image"
                self._force_recompute = True
            self._last_extracted_text = ""
            self._last_annotated_image = None

    def switch_to_camera(self, *args):
        with self._data_lock:
            self._inference_source = "camera"
            self._frozen_frame = None
        self._last_annotated_image = None

    def freeze_and_detect(self, *args):
        with self._data_lock:
            if self._inference_source == "camera":
                if self._latest_valid_frame is not None:
                    self._frozen_frame = self._latest_valid_frame.copy()
                    self._inference_source = "freeze"
                    self._force_recompute = True
                    self._last_annotated_image = None
                    self._last_extracted_text = ""
                    Clock.schedule_once(lambda dt: self.show_toast("Frame captured & analyzed"))
                else:
                    Clock.schedule_once(lambda dt: self.show_toast("No camera frame available"))
            elif self._inference_source in ["image", "freeze"]:
                # Force re-inference immediately on the current frozen/static image like a fresh upload
                self._force_recompute = True
                self._last_annotated_image = None
                Clock.schedule_once(lambda dt: self.show_toast("Re-analyzing frame"))

    def on_stop(self):
        self._is_running = False
        if self._capture is not None:
            try:
                if self._capture.isOpened(): self._capture.release()
            except Exception: pass
            finally: self._capture = None
        self._session = None


if __name__ == "__main__":
    UgariticStudioApp().run()
