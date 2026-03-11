"""SSMD model: student + teacher RetinaNet detectors with EMA update."""

import tensorflow as tf
from tensorflow import keras

from .backbone import ResNet50FPN
from .heads import ClassificationHead, RegressionHead


class RetinaNetDetector(keras.Model):
    """Single-network RetinaNet detector (student or teacher).

    Args:
        num_classes:  number of foreground classes (1 for nuclei/lesion)
        num_anchors:  anchors per location (default 9 = 3 scales × 3 ratios)
        use_noisy:    insert NoisyResidualBlocks in backbone
        fpn_channels: FPN feature width
    """

    def __init__(
        self,
        num_classes=1,
        num_anchors=9,
        use_noisy=True,
        fpn_channels=256,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.backbone = ResNet50FPN(
            fpn_channels=fpn_channels,
            use_noisy_blocks=use_noisy,
        )
        self.cls_head = ClassificationHead(num_classes, num_anchors)
        self.reg_head = RegressionHead(num_anchors)

    def call(self, images, training=False):
        """Forward pass.

        Args:
            images:   (B, H, W, C)
            training: bool

        Returns:
            cls_preds: list of (B, H_i, W_i, num_anchors * num_classes) per level
            reg_preds: list of (B, H_i, W_i, num_anchors * 4) per level
        """
        features = self.backbone(images, training=training)
        cls_preds = self.cls_head(features)
        reg_preds = self.reg_head(features)
        return cls_preds, reg_preds


class SSMDModel:
    """Student-Teacher SSMD wrapper.

    Manages two RetinaNetDetector instances: student (with NRB, trained via
    gradient descent) and teacher (no NRB, updated via EMA of student weights).
    """

    def __init__(
        self,
        num_classes=1,
        num_anchors=9,
        fpn_channels=256,
        ema_alpha=0.999,
    ):
        self.ema_alpha = ema_alpha

        self.student = RetinaNetDetector(
            num_classes=num_classes,
            num_anchors=num_anchors,
            use_noisy=True,
            fpn_channels=fpn_channels,
            name="student",
        )
        self.teacher = RetinaNetDetector(
            num_classes=num_classes,
            num_anchors=num_anchors,
            use_noisy=False,
            fpn_channels=fpn_channels,
            name="teacher",
        )

    def initialize_weights(self, dummy_input):
        """Build both models by doing a dummy forward pass, then copy student → teacher."""
        self.student(dummy_input, training=False)
        self.teacher(dummy_input, training=False)
        self._copy_student_to_teacher()

    def _shared_variable_pairs(self):
        """Yield (teacher_var, student_var) pairs for variables with matching names.

        Student has NoisyResidualBlock vars that teacher lacks; we match by the
        suffix of the variable path after the model name prefix.
        """
        t_vars = {self._strip_prefix(v.path, "teacher"): v
                  for v in self.teacher.trainable_variables}
        s_vars = {self._strip_prefix(v.path, "student"): v
                  for v in self.student.trainable_variables}
        for name, t_var in t_vars.items():
            if name in s_vars:
                yield t_var, s_vars[name]

    @staticmethod
    def _strip_prefix(path, prefix):
        """Remove leading '<prefix>/' from a variable path."""
        parts = path.split("/", 1)
        if len(parts) == 2 and parts[0] == prefix:
            return parts[1]
        return path

    def _copy_student_to_teacher(self):
        """Copy shared student weights to teacher (used at init)."""
        for t_var, s_var in self._shared_variable_pairs():
            t_var.assign(s_var)

    def update_ema(self):
        """EMA update: θ_t ← α·θ_t + (1-α)·θ_s  (eq. 2)."""
        alpha = self.ema_alpha
        for t_var, s_var in self._shared_variable_pairs():
            t_var.assign(alpha * t_var + (1 - alpha) * s_var)
