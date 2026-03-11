"""Instance-level adversarial perturbation r_adv (eq. 8-9)."""

import tensorflow as tf
from .losses import consistency_loss, adaptive_weight


def compute_r_adv(
    student_model,
    teacher_model,
    x,
    tau=0.5,
    eps=1.0,
    xi=0.01,
):
    """Compute instance-level adversarial perturbation r_adv (eq. 8-9).

    Only high-confidence foreground proposals (sum of fg probs > tau)
    contribute gradients. Perturbation is shared across labeled and
    unlabeled images in the batch.

    Args:
        student_model: RetinaNetDetector
        teacher_model: RetinaNetDetector
        x:  (B, H, W, C) images (combined labeled + unlabeled, or just the batch)
        tau: foreground confidence threshold
        eps: final perturbation scale
        xi:  initial normalisation scale

    Returns:
        r_adv: (B, H, W, C) adversarial perturbation, tf.Tensor (no grad)
    """
    x_shape = tf.shape(x)
    r = tf.Variable(
        tf.random.normal(x_shape, dtype=x.dtype),
        trainable=True,
        dtype=x.dtype,
    )
    # Normalise to small ball
    r.assign(xi * r / (tf.norm(r) + 1e-8))

    with tf.GradientTape() as tape:
        tape.watch(r)

        # Student forward with perturbation (training=True to keep NRB active)
        s_cls_list, s_reg_list = student_model(x + r, training=True)
        # Teacher without gradient (training=False — no noisy blocks)
        t_cls_list, t_reg_list = teacher_model(x + r, training=False)

        # Indicator: mask proposals where total foreground confidence < tau
        # We compute a per-proposal weight and zero out low-confidence ones
        losses = []
        for cls_s, cls_t, reg_s, reg_t in zip(
            s_cls_list, t_cls_list, s_reg_list, t_reg_list
        ):
            B = tf.shape(cls_s)[0]
            # cls: (B, H, W, A*1) → (B, H*W*A, 1)  reg: → (B, H*W*A, 4)
            cls_s_flat = tf.reshape(cls_s, [B, -1, 1])
            cls_t_flat = tf.reshape(cls_t, [B, -1, 1])
            reg_s_flat = tf.reshape(reg_s, [B, -1, 4])
            reg_t_flat = tf.reshape(reg_t, [B, -1, 4])

            # Foreground probability = 1 - background (B, N)
            fg_s = 1.0 - cls_s_flat[..., 0]
            fg_t = 1.0 - cls_t_flat[..., 0]

            # High-confidence fg indicator (eq. 9)
            indicator = tf.cast((fg_s + fg_t) / 2.0 > tau, tf.float32)  # (B, N)

            w = adaptive_weight(cls_s_flat, cls_t_flat) * indicator  # (B, N)

            cls_t_sg = tf.stop_gradient(cls_t_flat)
            kl = tf.reduce_sum(
                cls_t_sg * tf.math.log((cls_t_sg + 1e-7) / (cls_s_flat + 1e-7)),
                axis=-1,
            )  # (B, N)
            mse = tf.reduce_sum(
                (reg_s_flat - tf.stop_gradient(reg_t_flat)) ** 2, axis=-1
            )  # (B, N)
            losses.append(tf.reduce_mean(w * (kl + mse)))

        adv_loss = tf.add_n(losses) / tf.cast(len(losses), tf.float32)

    grad = tape.gradient(adv_loss, r)

    if grad is None:
        return tf.zeros_like(x)

    r_adv = eps * grad / (tf.norm(grad) + 1e-8)
    return tf.stop_gradient(r_adv)
