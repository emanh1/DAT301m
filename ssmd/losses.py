"""SSMD loss functions: supervised (focal + smooth_l1) and consistency."""

import tensorflow as tf
from .anchors import compute_iou, encode_boxes


def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss for binary classification.

    Args:
        y_true: (...) float32, 0 or 1.
        y_pred: (...) float32, probabilities in (0, 1).

    Returns:
        Scalar mean loss.
    """
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
    bce = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(1 - y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    fl = alpha_t * ((1 - p_t) ** gamma) * bce
    return tf.reduce_mean(fl)


def smooth_l1_loss(y_true, y_pred, beta=0.1):
    """Smooth L1 (Huber) loss.

    Args:
        y_true: (..., 4)
        y_pred: (..., 4)

    Returns:
        Scalar mean loss.
    """
    diff = tf.abs(y_true - y_pred)
    loss = tf.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta)
    return tf.reduce_mean(loss)


def _assign_anchors(gt_boxes, anchors, pos_iou=0.5, neg_iou=0.4):
    """Assign anchors to GT boxes via IoU thresholds.

    Args:
        gt_boxes: (G, 4)
        anchors:  (N, 4)
        pos_iou:  IoU threshold for positive assignment
        neg_iou:  IoU threshold for negative assignment

    Returns:
        cls_targets: (N,) int32  -1=ignore, 0=neg, 1=pos
        reg_targets: (N, 4)     only valid where cls_targets==1
    """
    num_anchors = tf.shape(anchors)[0]

    if tf.shape(gt_boxes)[0] == 0:
        cls_targets = tf.zeros([num_anchors], dtype=tf.int32)
        reg_targets = tf.zeros([num_anchors, 4], dtype=tf.float32)
        return cls_targets, reg_targets

    iou = compute_iou(anchors, gt_boxes)  # (N, G)
    max_iou = tf.reduce_max(iou, axis=1)  # (N,)
    best_gt = tf.argmax(iou, axis=1, output_type=tf.int32)  # (N,)

    pos_mask = max_iou >= pos_iou
    neg_mask = max_iou < neg_iou
    ignore_mask = ~(pos_mask | neg_mask)

    # Assign: -1=ignore, 0=neg, 1=pos
    cls_targets = tf.where(pos_mask, tf.ones_like(best_gt),
                  tf.where(neg_mask, tf.zeros_like(best_gt),
                  -tf.ones_like(best_gt)))

    # Regression targets for positive anchors
    matched_gt = tf.gather(gt_boxes, best_gt)  # (N, 4)
    reg_targets = encode_boxes(matched_gt, anchors)  # (N, 4)

    return cls_targets, reg_targets


def supervised_loss(
    cls_preds_list,
    reg_preds_list,
    gt_boxes_batch,
    anchors,
    num_classes=1,
):
    """Compute supervised loss (eq. 1) for a batch.

    Args:
        cls_preds_list: list of (B, H_i, W_i, A*C) per FPN level
        reg_preds_list: list of (B, H_i, W_i, A*4) per FPN level
        gt_boxes_batch: list of (G_b, 4) tensors, one per image in batch
        anchors: (N, 4) all anchors
        num_classes: int

    Returns:
        Scalar total supervised loss.
    """
    num_anchors_per_loc = 9  # 3 scales × 3 ratios
    batch_size = len(gt_boxes_batch)

    # Flatten predictions across FPN levels: (B, N, ...)
    cls_flat = tf.concat(
        [tf.reshape(p, [tf.shape(p)[0], -1, num_classes]) for p in cls_preds_list],
        axis=1,
    )  # (B, N, C)
    reg_flat = tf.concat(
        [tf.reshape(p, [tf.shape(p)[0], -1, 4]) for p in reg_preds_list],
        axis=1,
    )  # (B, N, 4)

    total_cls = 0.0
    total_reg = 0.0
    num_pos = 0

    for b in range(batch_size):
        cls_t, reg_t = _assign_anchors(gt_boxes_batch[b], anchors)

        pos_mask = cls_t == 1
        neg_mask = cls_t == 0
        valid_mask = pos_mask | neg_mask

        # Classification loss on valid anchors
        valid_cls_pred = tf.boolean_mask(cls_flat[b], valid_mask)  # (V, C)
        valid_cls_true = tf.cast(
            tf.boolean_mask(pos_mask, valid_mask), tf.float32
        )  # (V,)
        if num_classes == 1:
            valid_cls_true = tf.expand_dims(valid_cls_true, -1)

        if tf.shape(valid_cls_pred)[0] > 0:
            total_cls += focal_loss(valid_cls_true, valid_cls_pred)

        # Regression loss on positive anchors only
        pos_reg_pred = tf.boolean_mask(reg_flat[b], pos_mask)  # (P, 4)
        pos_reg_true = tf.boolean_mask(reg_t, pos_mask)       # (P, 4)

        n_pos = tf.reduce_sum(tf.cast(pos_mask, tf.int32))
        num_pos += n_pos

        if tf.shape(pos_reg_pred)[0] > 0:
            total_reg += smooth_l1_loss(pos_reg_true, pos_reg_pred)

    return (total_cls + total_reg) / tf.cast(tf.maximum(batch_size, 1), tf.float32)


def adaptive_weight(p_s, p_t):
    """Adaptive consistency weight W (eq. 4).

    W = ((1 - p_s[0])^2 + (1 - p_t[0])^2) / 2

    Args:
        p_s: (..., num_classes) student class probabilities
        p_t: (..., num_classes) teacher class probabilities

    Returns:
        W: (...) weights, near 0 for background, near 1 for foreground
    """
    bg_s = p_s[..., 0]
    bg_t = p_t[..., 0]
    return ((1 - bg_s) ** 2 + (1 - bg_t) ** 2) / 2.0


def consistency_loss(p_s_cls_list, p_t_cls_list, p_s_reg_list, p_t_reg_list,
                     num_classes=1):
    """Compute W ⊗ (KL + MSE) consistency loss (eq. 5).

    Args:
        p_s_cls_list: list of (B, H_i, W_i, A*num_classes) per level, student
        p_t_cls_list: same for teacher
        p_s_reg_list: list of (B, H_i, W_i, A*4) per level, student
        p_t_reg_list: same for teacher
        num_classes:  number of foreground classes (default 1)

    Returns:
        Scalar consistency loss.
    """
    total = 0.0
    n_levels = len(p_s_cls_list)

    for cls_s, cls_t, reg_s, reg_t in zip(
        p_s_cls_list, p_t_cls_list, p_s_reg_list, p_t_reg_list
    ):
        B = tf.shape(cls_s)[0]

        # cls: (B, H, W, A*C) → (B, H*W*A, C)
        # reg: (B, H, W, A*4) → (B, H*W*A, 4)
        cls_s_flat = tf.reshape(cls_s, [B, -1, num_classes])   # (B, N, C)
        cls_t_flat = tf.reshape(cls_t, [B, -1, num_classes])
        reg_s_flat = tf.reshape(reg_s, [B, -1, 4])             # (B, N, 4)
        reg_t_flat = tf.reshape(reg_t, [B, -1, 4])

        # Adaptive weight per proposal (B, N)
        w = adaptive_weight(cls_s_flat, cls_t_flat)

        # KL divergence: teacher is target distribution (stop_gradient)
        cls_t_sg = tf.stop_gradient(cls_t_flat)
        kl = tf.reduce_sum(
            cls_t_sg * tf.math.log((cls_t_sg + 1e-7) / (cls_s_flat + 1e-7)),
            axis=-1,
        )  # (B, N)

        # MSE on regression (B, N)
        mse = tf.reduce_sum(
            (reg_s_flat - tf.stop_gradient(reg_t_flat)) ** 2, axis=-1
        )

        level_loss = tf.reduce_mean(w * (kl + mse))
        total += level_loss

    return total / tf.cast(n_levels, tf.float32)
