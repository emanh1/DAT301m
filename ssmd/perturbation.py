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
    """
    Memory-efficient adversarial perturbation (SSMD eq. 8-9).
    """

    B = tf.shape(x)[0]

    # Initialize small random perturbation
    r = tf.random.normal(tf.shape(x), dtype=x.dtype)

    r_norm = tf.norm(tf.reshape(r, [B, -1]), axis=1, keepdims=True)
    r_norm = tf.reshape(r_norm, [-1, 1, 1, 1])
    r = xi * r / (r_norm + 1e-8)

    # Teacher forward pass (NO gradient tracking)
    t_cls_list, t_reg_list = teacher_model(x + r, training=False)

    with tf.GradientTape() as tape:
        tape.watch(r)

        # Student forward pass
        s_cls_list, s_reg_list = student_model(x + r, training=True)

        losses = []

        for cls_s, cls_t, reg_s, reg_t in zip(
            s_cls_list, t_cls_list, s_reg_list, t_reg_list
        ):

            B = tf.shape(cls_s)[0]

            cls_s = tf.reshape(cls_s, [B, -1, 1])
            cls_t = tf.reshape(cls_t, [B, -1, 1])

            reg_s = tf.reshape(reg_s, [B, -1, 4])
            reg_t = tf.reshape(reg_t, [B, -1, 4])

            cls_t = tf.stop_gradient(cls_t)
            reg_t = tf.stop_gradient(reg_t)

            # Foreground probability
            fg_s = 1.0 - cls_s[..., 0]
            fg_t = 1.0 - cls_t[..., 0]

            indicator = tf.cast((fg_s + fg_t) / 2.0 > tau, tf.float32)

            w = adaptive_weight(cls_s, cls_t) * indicator

            kl = tf.reduce_sum(
                cls_t * tf.math.log((cls_t + 1e-7) / (cls_s + 1e-7)),
                axis=-1,
            )

            mse = tf.reduce_sum((reg_s - reg_t) ** 2, axis=-1)

            losses.append(tf.reduce_mean(w * (kl + mse)))

        adv_loss = tf.add_n(losses) / tf.cast(len(losses), tf.float32)

    grad = tape.gradient(adv_loss, r)

    if grad is None:
        return tf.zeros_like(x)

    grad_norm = tf.norm(tf.reshape(grad, [B, -1]), axis=1, keepdims=True)
    grad_norm = tf.reshape(grad_norm, [-1, 1, 1, 1])

    r_adv = eps * grad / (grad_norm + 1e-8)

    return tf.stop_gradient(r_adv)
