"""Data augmentation: rotation, horizontal flip, cutout."""

import tensorflow as tf


def _rotate_image(image, angle_rad):
    """Rotate image by angle_rad using bilinear interpolation."""
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    # Build 2D rotation transform for tfa-free affine via tf.raw_ops if available,
    # otherwise fall back to a crop-and-pad approximation.
    # We use tf.keras.layers.RandomRotation logic: image center rotation.
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    # Affine transform matrix coefficients [a0, a1, a2, b0, b1, b2, c0, c1]
    # for tf.raw_ops.ImageProjectiveTransformV3 (available since TF 2.4)
    cx, cy = w / 2.0, h / 2.0
    a0 = cos_a
    a1 = -sin_a
    a2 = (1 - cos_a) * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = -sin_a * cx + (1 - cos_a) * cy

    transform = [a0, a1, a2, b0, b1, b2, 0.0, 0.0]
    transform = tf.reshape(tf.cast(transform, tf.float32), [1, 8])

    image = tf.expand_dims(image, 0)  # add batch dim
    image = tf.raw_ops.ImageProjectiveTransformV3(
        images=image,
        transforms=transform,
        output_shape=tf.shape(image)[1:3],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    return tf.squeeze(image, 0)


def _rotate_boxes(boxes, angle_rad, image_h, image_w):
    """Rotate bounding boxes around image center.

    Rotates all 4 corners, then takes bounding box of rotated corners.

    Args:
        boxes: (G, 4) [x1, y1, x2, y2] (float32)
        angle_rad: scalar
        image_h, image_w: image dimensions (float32 scalars)

    Returns:
        rotated_boxes: (G, 4)
    """
    cx = image_w / 2.0
    cy = image_h / 2.0
    cos_a = tf.math.cos(angle_rad)
    sin_a = tf.math.sin(angle_rad)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    # All 4 corners
    corners_x = tf.stack([x1, x2, x1, x2], axis=1) - cx  # (G, 4)
    corners_y = tf.stack([y1, y1, y2, y2], axis=1) - cy

    rot_x = corners_x * cos_a - corners_y * sin_a + cx
    rot_y = corners_x * sin_a + corners_y * cos_a + cy

    new_x1 = tf.reduce_min(rot_x, axis=1)
    new_y1 = tf.reduce_min(rot_y, axis=1)
    new_x2 = tf.reduce_max(rot_x, axis=1)
    new_y2 = tf.reduce_max(rot_y, axis=1)

    # Clip to image bounds
    new_x1 = tf.clip_by_value(new_x1, 0.0, image_w)
    new_y1 = tf.clip_by_value(new_y1, 0.0, image_h)
    new_x2 = tf.clip_by_value(new_x2, 0.0, image_w)
    new_y2 = tf.clip_by_value(new_y2, 0.0, image_h)

    return tf.stack([new_x1, new_y1, new_x2, new_y2], axis=1)


def random_rotation(image, boxes, max_degrees=10.0):
    """Randomly rotate image and boxes by up to ±max_degrees."""
    max_rad = max_degrees * (3.14159265 / 180.0)
    angle = tf.random.uniform([], -max_rad, max_rad)

    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)

    image = _rotate_image(image, angle)
    boxes = _rotate_boxes(boxes, angle, h, w)
    return image, boxes


def horizontal_flip(image, boxes):
    """Horizontally flip image and adjust boxes."""
    w = tf.cast(tf.shape(image)[1], tf.float32)
    image = tf.image.flip_left_right(image)
    x1 = w - boxes[:, 2]
    y1 = boxes[:, 1]
    x2 = w - boxes[:, 0]
    y2 = boxes[:, 3]
    boxes = tf.stack([x1, y1, x2, y2], axis=1)
    return image, boxes


def cutout(image, n_holes=1, min_size=32, max_size=64):
    """Zero out n_holes random rectangles of random size."""
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    mask = tf.ones([h, w, 1], dtype=image.dtype)

    for _ in range(n_holes):
        size_h = tf.random.uniform([], min_size, max_size + 1, dtype=tf.int32)
        size_w = tf.random.uniform([], min_size, max_size + 1, dtype=tf.int32)
        cy = tf.random.uniform([], 0, h, dtype=tf.int32)
        cx = tf.random.uniform([], 0, w, dtype=tf.int32)

        y1 = tf.maximum(cy - size_h // 2, 0)
        x1 = tf.maximum(cx - size_w // 2, 0)
        y2 = tf.minimum(cy + size_h // 2, h)
        x2 = tf.minimum(cx + size_w // 2, w)

        # Create a hole mask (1 everywhere, 0 in the hole)
        top = tf.ones([y1, w, 1], dtype=image.dtype)
        mid_left = tf.ones([y2 - y1, x1, 1], dtype=image.dtype)
        hole = tf.zeros([y2 - y1, x2 - x1, 1], dtype=image.dtype)
        mid_right = tf.ones([y2 - y1, w - x2, 1], dtype=image.dtype)
        bottom = tf.ones([h - y2, w, 1], dtype=image.dtype)

        mid = tf.concat([mid_left, hole, mid_right], axis=1)
        patch_mask = tf.concat([top, mid, bottom], axis=0)
        mask = mask * patch_mask

    return image * mask


def student_augment(image, boxes):
    """Student augmentation: rotation then cutout."""
    image, boxes = random_rotation(image, boxes, max_degrees=10.0)
    image = cutout(image)
    return image, boxes


def teacher_augment(image, boxes):
    """Teacher augmentation: flip then rotation then cutout."""
    image, boxes = horizontal_flip(image, boxes)
    image, boxes = random_rotation(image, boxes, max_degrees=10.0)
    image = cutout(image)
    return image, boxes
