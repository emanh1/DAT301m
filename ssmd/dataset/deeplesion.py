"""DeepLesion tf.data pipeline for CT lesion detection."""

import os
import numpy as np
import tensorflow as tf

# Dataset-level statistics (approximate; recalculated on first run if not cached)
_DL_MEAN = 0.0
_DL_STD = 1.0

# HU clipping range
HU_MIN, HU_MAX = -1100.0, 1100.0


def _hu_to_float(image_np):
    """Clip HU values and normalize to [-1, 1]."""
    image_np = np.clip(image_np.astype(np.float32), HU_MIN, HU_MAX)
    image_np = (image_np - HU_MIN) / (HU_MAX - HU_MIN)  # [0, 1]
    image_np = image_np * 2.0 - 1.0                      # [-1, 1]
    return image_np


def _load_dl_sample(image_path, bbox, target_size, mean, std):
    """Load one DeepLesion PNG slice.

    Args:
        image_path: bytes path to 16-bit PNG (HU + 32768 offset per DL convention)
        bbox: [x1, y1, x2, y2] in original pixel coords
        target_size: int
        mean, std: float32 scalars for final normalization

    Returns:
        image: (target_size, target_size, 3) float32
        boxes: (1, 4) float32
        labels: (1,) int32
    """
    import cv2

    img = cv2.imread(image_path.decode(), cv2.IMREAD_UNCHANGED)
    if img is None:
        image = np.zeros((target_size, target_size, 3), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)
        return image, boxes, labels

    # DeepLesion PNGs are stored as uint16 with HU + 32768
    if img.dtype == np.uint16:
        img = img.astype(np.float32) - 32768.0
    else:
        img = img.astype(np.float32)

    orig_h, orig_w = img.shape[:2]
    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    # Normalize HU
    img = _hu_to_float(img)
    # Dataset-level normalization
    img = (img - float(mean)) / (float(std) + 1e-8)

    # Resize to target
    img_resized = cv2.resize(img, (target_size, target_size))
    if len(img_resized.shape) == 2:
        img_resized = img_resized[:, :, np.newaxis]

    # Replicate to 3 channels if grayscale
    if img_resized.shape[2] == 1:
        img_resized = np.repeat(img_resized, 3, axis=2)

    # Scale bbox
    scale_x = target_size / orig_w
    scale_y = target_size / orig_h
    bx1, by1, bx2, by2 = (
        float(bbox[0]) * scale_x,
        float(bbox[1]) * scale_y,
        float(bbox[2]) * scale_x,
        float(bbox[3]) * scale_y,
    )
    boxes = np.array([[bx1, by1, bx2, by2]], dtype=np.float32)
    labels = np.array([1], dtype=np.int32)

    return img_resized.astype(np.float32), boxes, labels


def _py_load_dl(image_path, bbox, target_size, mean, std):
    image, boxes, labels = tf.numpy_function(
        func=lambda ip, bb: _load_dl_sample(
            ip, bb, int(target_size), float(mean), float(std)
        ),
        inp=[image_path, bbox],
        Tout=[tf.float32, tf.float32, tf.int32],
    )
    image.set_shape([target_size, target_size, 3])
    boxes.set_shape([None, 4])
    labels.set_shape([None])
    return image, boxes, labels


def _parse_dl_csv(csv_path):
    """Parse DL_info.csv.

    Returns list of (relative_image_path, [x1, y1, x2, y2]).
    The CSV columns we need (0-indexed):
      0: File_name  (e.g. 000001_01_01/000.png)
      6: Bounding_boxes (x1,y1,x2,y2 space-separated or comma-separated)
    """
    import csv

    records = []
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader)

        # Detect column indices
        try:
            fn_idx = header.index("File_name")
        except ValueError:
            fn_idx = 0
        try:
            bb_idx = header.index("Bounding_boxes")
        except ValueError:
            bb_idx = 6

        for row in reader:
            if len(row) <= max(fn_idx, bb_idx):
                continue
            fname = row[fn_idx].strip()
            bb_str = row[bb_idx].strip()
            try:
                coords = [float(v) for v in bb_str.replace(",", " ").split()]
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = coords[:4]
                records.append((fname, [x1, y1, x2, y2]))
            except ValueError:
                continue
    return records


def make_dataset(
    data_dir,
    labeled_fraction=0.2,
    split="train",
    target_size=512,
    batch_size=8,
    shuffle=True,
    seed=42,
    mean=_DL_MEAN,
    std=_DL_STD,
):
    """Build a tf.data.Dataset for DeepLesion.

    Expected layout:
        data_dir/DL_info.csv
        data_dir/Images_png/<patient_study_slice>/????.png

    Args:
        data_dir: root directory (e.g. 'dataset/deeplesion')
        labeled_fraction: fraction of samples used as labeled
        split: 'labeled', 'unlabeled', 'val', or 'train'
        target_size: 512 for DeepLesion
        batch_size: batch size
        shuffle: whether to shuffle
        seed: random seed
        mean, std: dataset statistics for final normalization

    Returns:
        tf.data.Dataset yielding (image, boxes, labels)
    """
    csv_path = os.path.join(data_dir, "DL_info.csv")
    images_root = os.path.join(data_dir, "Images_png")

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"DL_info.csv not found at {csv_path}")

    records = _parse_dl_csv(csv_path)
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    records = [records[i] for i in indices]

    n_total = len(records)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    n_labeled = max(1, int(labeled_fraction * n_train))

    train_recs = records[:n_train]
    val_recs = records[n_train:]
    labeled_recs = train_recs[:n_labeled]
    unlabeled_recs = train_recs[n_labeled:]

    split_map = {
        "labeled": labeled_recs,
        "unlabeled": unlabeled_recs,
        "val": val_recs,
        "train": train_recs,
    }
    recs = split_map[split]

    image_paths = [os.path.join(images_root, r[0]) for r in recs]
    bboxes = [r[1] for r in recs]

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, tf.cast(bboxes, tf.float32))
    )
    if shuffle:
        ds = ds.shuffle(len(image_paths), seed=seed)

    ds = ds.map(
        lambda ip, bb: _py_load_dl(ip, bb, target_size, mean, std),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.padded_batch(
        batch_size,
        padded_shapes=(
            [target_size, target_size, 3],  # image
            [None, 4],                      # boxes
            [None],                         # labels
        ),
        padding_values=(
            0.0,                            # image padding
            0.0,                            # box padding
            tf.cast(-1, tf.int32),          # label padding
        ),
        drop_remainder=False,
    )
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds
