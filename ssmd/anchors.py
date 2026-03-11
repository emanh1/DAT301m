"""Anchor generation, encoding/decoding, and IoU for SSMD."""

import math
import tensorflow as tf
import numpy as np


def generate_anchors(
    image_size,
    strides=(8, 16, 32, 64, 128),
    scales=(1.0, 2 ** (1 / 3), 2 ** (2 / 3)),
    ratios=(0.5, 1.0, 2.0),
):
    """Generate all anchors for an image.

    Uses ceil(image_size / stride) to match Conv2D(stride=2, padding='same')
    feature map sizes produced by the FPN (especially P6, P7).

    Args:
        image_size: int, assumes square images.
        strides: FPN level strides.
        scales: anchor scale multipliers per level.
        ratios: anchor aspect ratios.

    Returns:
        anchors: (N, 4) tensor [x1, y1, x2, y2] in absolute pixel coords.
    """
    all_anchors = []
    base_sizes = {s: s * 4 for s in strides}  # base anchor size = 4× stride

    for stride in strides:
        base_size = base_sizes[stride]
        feat_size = math.ceil(image_size / stride)
        xs = tf.cast(tf.range(feat_size) * stride + stride // 2, tf.float32)
        ys = tf.cast(tf.range(feat_size) * stride + stride // 2, tf.float32)
        # (feat_size, feat_size)
        cx, cy = tf.meshgrid(xs, ys)
        cx = tf.reshape(cx, [-1])
        cy = tf.reshape(cy, [-1])

        level_anchors = []
        for scale in scales:
            for ratio in ratios:
                area = (base_size * scale) ** 2
                w = tf.sqrt(area / ratio)
                h = w * ratio
                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2
                boxes = tf.stack([x1, y1, x2, y2], axis=-1)  # (n, 4)
                level_anchors.append(boxes)

        all_anchors.append(tf.concat(level_anchors, axis=0))

    return tf.concat(all_anchors, axis=0)  # (N, 4)


def encode_boxes(gt_boxes, anchors):
    """Encode GT boxes as (dx, dy, dw, dh) relative to anchors (eq. 3).

    Args:
        gt_boxes: (N, 4) [x1, y1, x2, y2]
        anchors:  (N, 4) [x1, y1, x2, y2]

    Returns:
        targets: (N, 4) [dx, dy, dw, dh]
    """
    eps = 1e-8
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2
    acy = anchors[:, 1] + ah / 2

    gw = gt_boxes[:, 2] - gt_boxes[:, 0]
    gh = gt_boxes[:, 3] - gt_boxes[:, 1]
    gcx = gt_boxes[:, 0] + gw / 2
    gcy = gt_boxes[:, 1] + gh / 2

    dx = (gcx - acx) / (aw + eps)
    dy = (gcy - acy) / (ah + eps)
    dw = tf.math.log(gw / (aw + eps) + eps)
    dh = tf.math.log(gh / (ah + eps) + eps)

    return tf.stack([dx, dy, dw, dh], axis=-1)


def decode_boxes(preds, anchors):
    """Decode predicted (dx, dy, dw, dh) back to absolute [x1, y1, x2, y2].

    Args:
        preds:   (N, 4) [dx, dy, dw, dh]
        anchors: (N, 4) [x1, y1, x2, y2]

    Returns:
        boxes: (N, 4) [x1, y1, x2, y2]
    """
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    acx = anchors[:, 0] + aw / 2
    acy = anchors[:, 1] + ah / 2

    gcx = preds[:, 0] * aw + acx
    gcy = preds[:, 1] * ah + acy
    gw = tf.exp(tf.clip_by_value(preds[:, 2], -10, 10)) * aw
    gh = tf.exp(tf.clip_by_value(preds[:, 3], -10, 10)) * ah

    x1 = gcx - gw / 2
    y1 = gcy - gh / 2
    x2 = gcx + gw / 2
    y2 = gcy + gh / 2

    return tf.stack([x1, y1, x2, y2], axis=-1)


def compute_iou(boxes_a, boxes_b):
    """Compute pairwise IoU matrix.

    Args:
        boxes_a: (M, 4) [x1, y1, x2, y2]
        boxes_b: (N, 4) [x1, y1, x2, y2]

    Returns:
        iou: (M, N)
    """
    # Expand dims for broadcasting
    a = tf.expand_dims(boxes_a, 1)  # (M, 1, 4)
    b = tf.expand_dims(boxes_b, 0)  # (1, N, 4)

    inter_x1 = tf.maximum(a[..., 0], b[..., 0])
    inter_y1 = tf.maximum(a[..., 1], b[..., 1])
    inter_x2 = tf.minimum(a[..., 2], b[..., 2])
    inter_y2 = tf.minimum(a[..., 3], b[..., 3])

    inter_w = tf.maximum(inter_x2 - inter_x1, 0.0)
    inter_h = tf.maximum(inter_y2 - inter_y1, 0.0)
    inter_area = inter_w * inter_h

    area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
    area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])

    area_a = tf.expand_dims(area_a, 1)  # (M, 1)
    area_b = tf.expand_dims(area_b, 0)  # (1, N)

    union = area_a + area_b - inter_area
    iou = inter_area / (union + 1e-8)
    return iou
