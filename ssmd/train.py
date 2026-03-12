"""SSMD training loop with lambda schedule (eq. 10)."""

import math
import os
import tensorflow as tf
from tqdm import tqdm

from .model import SSMDModel
from .anchors import generate_anchors
from .losses import supervised_loss, consistency_loss
from .perturbation import compute_r_adv
from .augment import student_augment, teacher_augment


# ─── Lambda schedule (eq. 10) ────────────────────────────────────────────────

def lambda_schedule(j, N):
    """Ramp-up / plateau / ramp-down schedule for consistency weight.

    Args:
        j: current training step (int or tf scalar)
        N: total training steps (int)

    Returns:
        float lambda value in [0, 1]
    """
    j = int(j)
    N = int(N)
    if N == 0:
        return 1.0
    if j < N // 4:
        return math.exp(-5 * (1 - 4 * j / N) ** 2)
    elif j < 3 * N // 4:
        return 1.0
    else:
        tail = max(N - j, 0)
        return math.exp(-12.5 * (1 - 7 * tail / N) ** 2)


# ─── Single train step ────────────────────────────────────────────────────────

def _augment_batch(images, boxes_list):
    """Apply student augmentation to a batch.

    Args:
        images: (B, H, W, C) float32 tensor
        boxes_list: list of (G_b, 4) boxes per image (length may be 0 for unlabeled)

    Returns:
        aug_images: (B, H, W, C)
        aug_boxes_list: list of (G_b, 4)
    """
    aug_imgs = []
    aug_boxes = []
    batch_size = images.shape[0] or int(tf.shape(images)[0])
    for i in range(batch_size):
        img = images[i]
        boxes = boxes_list[i] if i < len(boxes_list) else tf.zeros([0, 4], dtype=tf.float32)
        aug_img, aug_box = student_augment(img, tf.cast(boxes, tf.float32))
        aug_imgs.append(aug_img)
        aug_boxes.append(aug_box)
    return tf.stack(aug_imgs, axis=0), aug_boxes


def _teacher_augment_batch(images, boxes_list):
    aug_imgs = []
    aug_boxes = []
    batch_size = images.shape[0] or int(tf.shape(images)[0])
    for i in range(batch_size):
        img = images[i]
        boxes = boxes_list[i] if i < len(boxes_list) else tf.zeros([0, 4], dtype=tf.float32)
        aug_img, aug_box = teacher_augment(img, tf.cast(boxes, tf.float32))
        aug_imgs.append(aug_img)
        aug_boxes.append(aug_box)
    return tf.stack(aug_imgs, axis=0), aug_boxes


def train_step(ssmd, optimizer, labeled_batch, unlabeled_batch, anchors, step, N,
               use_adv=True):
    """One SSMD training step.

    Args:
        ssmd: SSMDModel instance
        optimizer: tf.keras.optimizers.Optimizer
        labeled_batch: (images, boxes_list, labels_list)
        unlabeled_batch: (images, boxes_list, labels_list) — labels ignored
        anchors: (A, 4) anchor tensor
        step: current global step (int)
        N: total steps (int)

    Returns:
        dict with keys 'loss_sup', 'loss_cont', 'loss_total'
    """
    l_images, l_boxes_padded, l_labels_padded = labeled_batch
    u_images, _, _ = unlabeled_batch

    # Strip padding rows (label == -1 marks padding added by padded_batch)
    n_labeled = l_images.shape[0] or int(tf.shape(l_images)[0])
    n_unlabeled = u_images.shape[0] or int(tf.shape(u_images)[0])
    l_boxes_list = [
        tf.boolean_mask(l_boxes_padded[b], l_labels_padded[b] >= 0)
        for b in range(n_labeled)
    ]

    # 1. Augment
    l_images_s, l_boxes_s = _augment_batch(l_images, l_boxes_list)
    u_images_s, _ = _augment_batch(u_images, [])

    # 2. Compute r_adv on combined batch (eq. 8-9)
    all_images_s = tf.concat([l_images_s, u_images_s], axis=0)
    if use_adv:
        r_adv = compute_r_adv(
            ssmd.student, ssmd.teacher, all_images_s, tau=0.5, eps=1.0, xi=0.01
        )
    else:
        r_adv = tf.zeros_like(all_images_s)

    # 3. Teacher augmentation + add r_adv
    l_images_t, _ = _teacher_augment_batch(l_images, l_boxes_list)
    u_images_t, _ = _teacher_augment_batch(u_images, [])
    all_images_t = tf.concat([l_images_t, u_images_t], axis=0) + r_adv

    # 4. Teacher forward pass (outside tape — no gradients needed through teacher)
    t_cls_all, t_reg_all = ssmd.teacher(all_images_t, training=False)

    # 5. Student forward + loss under GradientTape
    with tf.GradientTape() as tape:
        # Student (training=True → NRB active)
        s_cls_all, s_reg_all = ssmd.student(all_images_s, training=True)

        # Split labeled / unlabeled predictions
        n_labeled = l_images_s.shape[0] or tf.shape(l_images_s)[0]

        # Supervised loss (labeled portion of student predictions)
        s_cls_labeled = [p[:n_labeled] for p in s_cls_all]
        s_reg_labeled = [p[:n_labeled] for p in s_reg_all]

        loss_sup = supervised_loss(
            s_cls_labeled,
            s_reg_labeled,
            l_boxes_s,
            anchors,
            num_classes=ssmd.student.num_classes,
        )

        # Consistency loss (all images, student vs teacher)
        loss_cont = consistency_loss(s_cls_all, t_cls_all, s_reg_all, t_reg_all,
                                     num_classes=ssmd.student.num_classes)

        lam = lambda_schedule(step, N)
        loss_total = loss_sup + lam * loss_cont

    grads = tape.gradient(loss_total, ssmd.student.trainable_variables)
    optimizer.apply_gradients(zip(grads, ssmd.student.trainable_variables))

    # 5. EMA update
    ssmd.update_ema()

    return {
        "loss_sup": float(loss_sup),
        "loss_cont": float(loss_cont),
        "loss_total": float(loss_total),
    }


# ─── Main training function ───────────────────────────────────────────────────

def _lr_schedule_fn(epoch, lr):
    """Decay LR by ×0.1 at epoch 75."""
    if epoch == 75:
        return lr * 0.1
    return lr


def train(config):
    """Build datasets, model, and run training.

    Args:
        config: dict with keys:
            dataset: 'dsb' or 'deeplesion'
            labeled_fraction: float
            epochs: int
            batch_size: int
            data_dir: str (optional, defaults to 'dataset/<name>')
    """
    dataset_name = config.get("datasets", "dsb")
    labeled_fraction = config.get("labeled_fraction", 0.2)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 8)
    use_adv = config.get("use_adv", True)
    data_dir = config.get("data_dir", os.path.join("datasets", dataset_name))

    num_classes = 1

    # ── Dataset ──────────────────────────────────────────────────────────────
    if dataset_name == "dsb":
        from .dataset.dsb import make_dataset
        image_size = 448
    elif dataset_name == "deeplesion":
        from .dataset.deeplesion import make_dataset
        image_size = 512
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    print(f"Loading {dataset_name} dataset from {data_dir} ...")
    labeled_ds = make_dataset(
        data_dir,
        labeled_fraction=labeled_fraction,
        split="labeled",
        target_size=image_size,
        batch_size=batch_size,
    )
    unlabeled_ds = make_dataset(
        data_dir,
        labeled_fraction=labeled_fraction,
        split="unlabeled",
        target_size=image_size,
        batch_size=batch_size,
    )

    # ── Anchors ──────────────────────────────────────────────────────────────
    anchors = generate_anchors(image_size)
    print(f"Total anchors: {anchors.shape[0]}")

    # ── Model ─────────────────────────────────────────────────────────────────
    ssmd = SSMDModel(
        num_classes=num_classes,
        num_anchors=9,
        fpn_channels=256,
        ema_alpha=0.999,
    )

    # Build by running a dummy forward pass
    dummy = tf.zeros([1, image_size, image_size, 3], dtype=tf.float32)
    print("Building model (downloading ImageNet weights if needed)...")
    ssmd.initialize_weights(dummy)
    print(f"Student trainable params: {ssmd.student.count_params():,}")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

    # ── Training loop ─────────────────────────────────────────────────────────
    # Zip labeled and unlabeled datasets; repeat unlabeled to always have batches
    labeled_list = list(labeled_ds)
    unlabeled_list = list(unlabeled_ds)

    n_labeled_batches = len(labeled_list)
    steps_per_epoch = n_labeled_batches
    total_steps = epochs * steps_per_epoch

    print(f"Training: {epochs} epochs × {steps_per_epoch} steps = {total_steps} steps")

    global_step = 0
    for epoch in range(epochs):
        # LR decay at epoch 75
        if epoch == 75:
            optimizer.learning_rate.assign(optimizer.learning_rate * 0.1)
            print(f"Epoch {epoch}: LR decayed to {float(optimizer.learning_rate):.2e}")

        epoch_losses = {"loss_sup": [], "loss_cont": [], "loss_total": []}

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch+1}/{epochs}")
        for local_step in pbar:
            l_batch = labeled_list[local_step % n_labeled_batches]
            u_batch = unlabeled_list[local_step % len(unlabeled_list)]

            losses = train_step(
                ssmd, optimizer, l_batch, u_batch, anchors, global_step, total_steps, use_adv
            )
            global_step += 1

            for k, v in losses.items():
                epoch_losses[k].append(v)

            pbar.set_postfix({
                "sup": f"{losses['loss_sup']:.4f}",
                "cont": f"{losses['loss_cont']:.4f}",
                "total": f"{losses['loss_total']:.4f}",
            })

        mean_losses = {k: sum(v) / len(v) for k, v in epoch_losses.items() if v}
        print(
            f"Epoch {epoch+1}/{epochs} — "
            + " | ".join(f"{k}: {v:.4f}" for k, v in mean_losses.items())
        )

    print("Training complete.")
    return ssmd
