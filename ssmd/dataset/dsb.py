"""DSB 2018 (Data Science Bowl) tf.data pipeline for nuclei detection."""

import os
import numpy as np
import tensorflow as tf


def _masks_to_boxes(mask_dir):
    """Convert per-nucleus PNG masks to bounding boxes.

    Args:
        mask_dir: path to directory of mask PNGs

    Returns:
        boxes: (G, 4) float32 array [x1, y1, x2, y2]
    """
    import cv2

    boxes = []
    if not os.path.isdir(mask_dir):
        return np.zeros((0, 4), dtype=np.float32)

    for mask_file in sorted(os.listdir(mask_dir)):
        mask_path = os.path.join(mask_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            continue
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        boxes.append([float(x1), float(y1), float(x2), float(y2)])

    if not boxes:
        return np.zeros((0, 4), dtype=np.float32)
    return np.array(boxes, dtype=np.float32)


def _load_sample(image_path, mask_dir, target_size=448):
    """Load one DSB sample: resize image, convert masks, scale boxes."""
    import cv2

    image = cv2.imread(image_path.decode())
    if image is None:
        image = np.zeros((target_size, target_size, 3), dtype=np.float32)
        boxes = np.zeros((0, 4), dtype=np.float32)
        labels = np.zeros((0,), dtype=np.int32)
        return image, boxes, labels

    orig_h, orig_w = image.shape[:2]
    image = cv2.resize(image, (target_size, target_size)).astype(np.float32) / 255.0
    # Convert BGR → RGB
    image = image[..., ::-1]

    scale_x = target_size / orig_w
    scale_y = target_size / orig_h

    boxes = _masks_to_boxes(mask_dir.decode())
    if len(boxes) > 0:
        boxes[:, 0] *= scale_x
        boxes[:, 2] *= scale_x
        boxes[:, 1] *= scale_y
        boxes[:, 3] *= scale_y

    labels = np.ones((len(boxes),), dtype=np.int32)
    return image.astype(np.float32), boxes.astype(np.float32), labels.astype(np.int32)


def _py_load_sample(image_path, mask_dir, target_size):
    image, boxes, labels = tf.numpy_function(
        func=lambda ip, md: _load_sample(ip, md, int(target_size)),
        inp=[image_path, mask_dir],
        Tout=[tf.float32, tf.float32, tf.int32],
    )
    image.set_shape([target_size, target_size, 3])
    boxes.set_shape([None, 4])
    labels.set_shape([None])
    return image, boxes, labels


def make_dataset(
    data_dir,
    labeled_fraction=0.2,
    split="train",
    target_size=448,
    batch_size=8,
    shuffle=True,
    seed=42,
):
    """Build a tf.data.Dataset for DSB 2018.

    Expected layout:
        data_dir/train/<image_id>/images/*.png
        data_dir/train/<image_id>/masks/*.png

    Args:
        data_dir: root directory (e.g. 'dataset/dsb')
        labeled_fraction: fraction of training ids used as labeled
        split: 'labeled', 'unlabeled', or 'val' (uses last 10% of ids for val)
        target_size: resize target (448 for DSB)
        batch_size: batch size
        shuffle: whether to shuffle
        seed: random seed for deterministic split

    Returns:
        tf.data.Dataset yielding (image, boxes, labels)
    """
    train_dir = os.path.join(data_dir, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"DSB train dir not found: {train_dir}")

    image_ids = sorted(os.listdir(train_dir))
    rng = np.random.default_rng(seed)
    rng.shuffle(image_ids)

    n_total = len(image_ids)
    n_val = max(1, int(0.1 * n_total))
    n_train = n_total - n_val
    n_labeled = max(1, int(labeled_fraction * n_train))

    train_ids = image_ids[:n_train]
    val_ids = image_ids[n_train:]
    labeled_ids = train_ids[:n_labeled]
    unlabeled_ids = train_ids[n_labeled:]

    split_map = {
        "labeled": labeled_ids,
        "unlabeled": unlabeled_ids,
        "val": val_ids,
        "train": train_ids,
    }
    ids = split_map[split]

    image_paths = []
    mask_dirs = []
    for img_id in ids:
        img_folder = os.path.join(train_dir, img_id, "images")
        msk_folder = os.path.join(train_dir, img_id, "masks")
        pngs = [f for f in os.listdir(img_folder) if f.endswith(".png")] \
            if os.path.isdir(img_folder) else []
        if not pngs:
            continue
        image_paths.append(os.path.join(img_folder, sorted(pngs)[0]))
        mask_dirs.append(msk_folder)

    if not image_paths:
        raise ValueError(f"No samples found for split '{split}' in {data_dir}")

    ds = tf.data.Dataset.from_tensor_slices(
        (image_paths, mask_dirs)
    )
    if shuffle:
        ds = ds.shuffle(len(image_paths), seed=seed)

    ds = ds.map(
        lambda ip, md: _py_load_sample(ip, md, target_size),
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
