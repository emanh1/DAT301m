"""RetinaNet-style classification and regression heads."""

import math
import tensorflow as tf
from tensorflow import keras


class ClassificationHead(keras.layers.Layer):
    """4× Conv3x3 + ReLU -> Conv3x3 -> sigmoid.

    Shared weights applied across all FPN levels.
    Output shape per level: (B, H, W, num_anchors * num_classes)
    """

    def __init__(self, num_classes, num_anchors, prior_prob=0.01, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        bias_init = -math.log((1 - prior_prob) / prior_prob)

        self.convs = [
            keras.layers.Conv2D(256, 3, padding="same", activation="relu",
                                name=f"cls_conv{i}")
            for i in range(4)
        ]
        self.out_conv = keras.layers.Conv2D(
            num_anchors * num_classes, 3, padding="same",
            bias_initializer=keras.initializers.Constant(bias_init),
            name="cls_out",
        )

    def call(self, features):
        """Apply head to a list of FPN feature maps.

        Args:
            features: list of (B, H_i, W_i, C) tensors.

        Returns:
            list of (B, H_i, W_i, num_anchors * num_classes) tensors.
        """
        outputs = []
        for feat in features:
            x = feat
            for conv in self.convs:
                x = conv(x)
            x = tf.sigmoid(self.out_conv(x))
            outputs.append(x)
        return outputs


class RegressionHead(keras.layers.Layer):
    """4× Conv3x3 + ReLU -> Conv3x3.

    Shared weights applied across all FPN levels.
    Output shape per level: (B, H, W, num_anchors * 4)
    """

    def __init__(self, num_anchors, **kwargs):
        super().__init__(**kwargs)
        self.num_anchors = num_anchors

        self.convs = [
            keras.layers.Conv2D(256, 3, padding="same", activation="relu",
                                name=f"reg_conv{i}")
            for i in range(4)
        ]
        self.out_conv = keras.layers.Conv2D(
            num_anchors * 4, 3, padding="same", name="reg_out"
        )

    def call(self, features):
        """Apply head to a list of FPN feature maps.

        Args:
            features: list of (B, H_i, W_i, C) tensors.

        Returns:
            list of (B, H_i, W_i, num_anchors * 4) tensors.
        """
        outputs = []
        for feat in features:
            x = feat
            for conv in self.convs:
                x = conv(x)
            x = self.out_conv(x)
            outputs.append(x)
        return outputs
