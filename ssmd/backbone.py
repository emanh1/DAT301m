"""ResNet-50 backbone with Noisy Residual Blocks and Feature Pyramid Network."""

import tensorflow as tf
from tensorflow import keras


class NoisyResidualBlock(keras.layers.Layer):
    """Adds channel-wise attention-gated Gaussian noise (eq. 7).

    X^q = X^l + X^n ⊗ sigmoid(γ · X^p)
    where:
      X^p = Conv1x1(GlobalAvgPool(X^l))  — channel importance
      X^n = N(0, I) at train time, 0 at eval
    """

    def __init__(self, gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma

    def build(self, input_shape):
        channels = input_shape[-1]
        self.proj = keras.layers.Conv2D(
            channels, 1, padding="same", use_bias=True, name="nrb_proj"
        )
        self.gap = keras.layers.GlobalAveragePooling2D(keepdims=True)
        super().build(input_shape)

    def call(self, x_l, training=False):
        x_p = self.proj(self.gap(x_l))  # (B, 1, 1, C)

        if training:
            x_n = tf.random.normal(tf.shape(x_l))
        else:
            x_n = tf.zeros_like(x_l)

        return x_l + x_n * tf.sigmoid(self.gamma * x_p)

    def get_config(self):
        config = super().get_config()
        config.update({"gamma": self.gamma})
        return config


def _lateral_conv(channels, name):
    return keras.layers.Conv2D(channels, 1, padding="same", name=name)


def _merge_conv(channels, name):
    return keras.layers.Conv2D(channels, 3, padding="same", name=name)


class ResNet50FPN(keras.Model):
    """ResNet-50 + FPN with optional Noisy Residual Blocks.

    Produces P3, P4, P5, P6, P7 feature maps.
    """

    def __init__(self, fpn_channels=256, use_noisy_blocks=True, gamma=0.9, **kwargs):
        super().__init__(**kwargs)
        self.fpn_channels = fpn_channels
        self.use_noisy_blocks = use_noisy_blocks

        # Load ImageNet-pretrained ResNet-50 backbone
        base = keras.applications.ResNet50(include_top=False, weights="imagenet")

        # Extract intermediate feature maps
        c3_out = base.get_layer("conv3_block4_out").output
        c4_out = base.get_layer("conv4_block6_out").output
        c5_out = base.get_layer("conv5_block3_out").output

        self.encoder = keras.Model(
            inputs=base.input, outputs=[c3_out, c4_out, c5_out], name="resnet50_encoder"
        )
        # Freeze batch-norm layers from pretrained backbone
        for layer in self.encoder.layers:
            if isinstance(layer, keras.layers.BatchNormalization):
                layer.trainable = False

        # Optional noisy blocks for each C-level output
        if use_noisy_blocks:
            self.nrb3 = NoisyResidualBlock(gamma=gamma, name="nrb_c3")
            self.nrb4 = NoisyResidualBlock(gamma=gamma, name="nrb_c4")
            self.nrb5 = NoisyResidualBlock(gamma=gamma, name="nrb_c5")

        # FPN lateral connections
        self.lat3 = _lateral_conv(fpn_channels, "fpn_lat3")
        self.lat4 = _lateral_conv(fpn_channels, "fpn_lat4")
        self.lat5 = _lateral_conv(fpn_channels, "fpn_lat5")

        # FPN merge convolutions
        self.merge3 = _merge_conv(fpn_channels, "fpn_merge3")
        self.merge4 = _merge_conv(fpn_channels, "fpn_merge4")
        self.merge5 = _merge_conv(fpn_channels, "fpn_merge5")

        # Extra levels
        self.p6_conv = keras.layers.Conv2D(
            fpn_channels, 3, strides=2, padding="same", name="fpn_p6"
        )
        self.p7_relu = keras.layers.ReLU(name="fpn_p7_relu")
        self.p7_conv = keras.layers.Conv2D(
            fpn_channels, 3, strides=2, padding="same", name="fpn_p7"
        )

        self.upsample = keras.layers.UpSampling2D(size=2, interpolation="nearest")

    def call(self, images, training=False):
        c3, c4, c5 = self.encoder(images, training=training)

        if self.use_noisy_blocks:
            c3 = self.nrb3(c3, training=training)
            c4 = self.nrb4(c4, training=training)
            c5 = self.nrb5(c5, training=training)

        # Top-down FPN
        p5 = self.merge5(self.lat5(c5))
        p4 = self.merge4(self.lat4(c4) + self.upsample(p5))
        p3 = self.merge3(self.lat3(c3) + self.upsample(p4))

        p6 = self.p6_conv(p5)
        p7 = self.p7_conv(self.p7_relu(p6))

        return [p3, p4, p5, p6, p7]
