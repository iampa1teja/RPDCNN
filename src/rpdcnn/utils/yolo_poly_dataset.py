import os 
import glob 
import torch 
from torch.utils.data import Dataset
import numpy as np 
import cv2

class YoloPolyDataset(Dataset):
    """
    Expects:
        img_dir/  *.jpg | *.png
        label_dir/ *.txt   (same stem as image)

    Label line format (YOLO segmentation):
        class_id x1 y1 x2 y2 x3 y3 ... xn yn
    All coordinates normalized to [0, 1]. One line per instance. Multiple
    rings for a single instance are NOT supported by the standard YOLO-seg
    format, so each line is treated as one ring / one instance.
    """

    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

    def __init__(self, img_dir: str, label_dir: str, img_h: int, img_w: int):
        self.img_h = img_h
        self.img_w = img_w

        self.img_paths = sorted([
            p for p in glob.glob(os.path.join(img_dir, "*"))
            if os.path.splitext(p)[1].lower() in self.IMG_EXTS
        ])
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        self.label_dir = label_dir

    def __len__(self):
        return len(self.img_paths)

    def _label_path(self, img_path: str) -> str:
        stem = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.label_dir, stem + ".txt")

    def _load_labels(self, label_path: str):
        labels, polygons = [], []
        if not os.path.exists(label_path):
            return np.zeros((0,), dtype=np.int64), []

        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 7:  
                    continue
                cls_id = int(float(parts[0]))
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                labels.append(cls_id)
                polygons.append([coords])  

        return np.array(labels, dtype=np.int64), polygons

    def __getitem__(self, idx: int):
        img_path = self.img_paths[idx]
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise IOError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        orig_h, orig_w = image.shape[:2]

        image = cv2.resize(image, (self.img_w, self.img_h), interpolation=cv2.INTER_LINEAR)

        labels, polygons = self._load_labels(self._label_path(img_path))

        image_t = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return {
            "image": image_t,
            "labels": labels,
            "polygons": polygons,
            "orig_size": (orig_h, orig_w),
            "img_path": img_path,
        }


def yolo_poly_collate_fn(batch):
    images = torch.stack([b["image"] for b in batch], dim=0)
    labels_batch = [b["labels"] for b in batch]
    polygons_batch = [b["polygons"] for b in batch]
    img_paths = [b["img_path"] for b in batch]
    return {
        "images": images,
        "labels_batch": labels_batch,
        "polygons_batch": polygons_batch,
        "img_paths": img_paths,
    }
