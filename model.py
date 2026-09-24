"""
PainSense - model definitions.

CNNEncoder:
    ResNet18 backbone (ImageNet-pretrained) -> embedding.

CNNClassifier:
    CNNEncoder + classification head -> NoPain / Pain.
    Used for stage-1 frame classification.

PainCNNLSTM:
    CNNEncoder applied to every frame
    -> LSTM over real video sequences
    -> Pain / NoPain classification for each timestep.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


class CNNEncoder(nn.Module):
    """
    ResNet18 feature extractor.

    Input:
        (B, C, H, W)

    Output:
        (B, EMBED_DIM)
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        pretrained=True
    ):
        super().__init__()

        weights = (
            tvm.ResNet18_Weights.IMAGENET1K_V1
            if pretrained
            else None
        )

        backbone = tvm.resnet18(weights=weights)

        in_features = backbone.fc.in_features

        # Remove ResNet's original classification layer.
        backbone.fc = nn.Identity()

        self.backbone = backbone

        # Project ResNet features to the dimension
        # expected by the LSTM.
        self.proj = nn.Linear(
            in_features,
            embed_dim
        )

    def forward(self, x):
        # x: (B, C, H, W)

        feats = self.backbone(x)

        # (B, in_features)
        return self.proj(feats)

    def freeze_backbone(self, freeze=True):

        for p in self.backbone.parameters():
            p.requires_grad = not freeze


class CNNClassifier(nn.Module):
    """
    Stage 1:

    Individual face frame
        ↓
    CNN
        ↓
    NoPain / Pain

    Classes:
        0 = NoPain
        1 = Pain
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim
        )

        # Two output logits:
        # 0 = NoPain
        # 1 = Pain
        self.head = nn.Linear(
            embed_dim,
            2
        )

    def forward(self, x):

        emb = self.encoder(x)

        return self.head(emb)


class PainCNNLSTM(nn.Module):
    """
    Stage 2:

    Real consecutive video frames
        ↓
    CNN / ResNet18
        ↓
    Frame-level spatial features
        ↓
    LSTM
        ↓
    Pain / NoPain logits

    Input:
        (B, T, C, H, W)

    Output:
        (B, T, 2)

    Classes:
        0 = NoPain
        1 = Pain
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        # Two output logits for each timestep.
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        x shape:

            (B, T, C, H, W)

        Example:

            (8, 8, 3, 224, 224)

        means:
            8 sequences
            8 frames per sequence
            RGB
            224 x 224
        """

        b, t, c, h, w = x.shape

        # Treat every frame as an individual image
        # while passing it through the shared CNN.
        x = x.reshape(
            b * t,
            c,
            h,
            w
        )

        # CNN spatial features.
        emb = self.encoder(x)

        # Put the time dimension back.
        emb = emb.reshape(
            b,
            t,
            -1
        )

        # Temporal processing.
        lstm_out, _ = self.lstm(emb)

        # Classification for every timestep.
        logits = self.head(lstm_out)

        # (B, T, 2)
        return logits

    def load_cnn_weights(
        self,
        cnn_classifier_state_dict
    ):
        """
        Transfer the trained CNN encoder from
        CNNClassifier into the CNN-LSTM.
        """

        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in cnn_classifier_state_dict.items()
            if k.startswith("encoder.")
        }

        self.encoder.load_state_dict(
            encoder_state
        )


class PainIntensityLSTM(nn.Module):
    """
    CNN-LSTM regression head.

    Input:
        (B, T, C, H, W)

    Output:
        (B, T) intensity scores on the generated 0-10 scale.
    """

    def __init__(
        self,
        embed_dim=config.EMBED_DIM,
        hidden_dim=config.LSTM_HIDDEN
    ):
        super().__init__()

        self.encoder = CNNEncoder(
            embed_dim=embed_dim
        )

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        b, t, c, h, w = x.shape

        x = x.reshape(b * t, c, h, w)
        emb = self.encoder(x)
        emb = emb.reshape(b, t, -1)
        lstm_out, _ = self.lstm(emb)
        scores = self.head(lstm_out).squeeze(-1)
        return scores

    def load_cnn_weights(self, cnn_classifier_state_dict):
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in cnn_classifier_state_dict.items()
            if k.startswith("encoder.")
        }
        self.encoder.load_state_dict(encoder_state)