"""
PainSense - model definitions.

CNNEncoder   : ResNet18 backbone (ImageNet-pretrained) -> EMBED_DIM embedding.
               This is the "spatial features" half of the pipeline.
CNNClassifier: CNNEncoder + linear head -> binary pain/no_pain logits.
               Used for stage-1 training (real supervision from SynPAIN).
PainCNNLSTM  : CNNEncoder (per-frame, shared weights) -> LSTM over the
               sequence -> per-timestep pain-intensity regression (0-10).
               Used for stage-2 training (synthetic onset sequences) and
               for live inference.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

import config


class CNNEncoder(nn.Module):
    def __init__(self, embed_dim=config.EMBED_DIM, pretrained=True):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Linear(in_features, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        feats = self.backbone(x)          # (B, in_features)
        return self.proj(feats)           # (B, embed_dim)

    def freeze_backbone(self, freeze=True):
        for p in self.backbone.parameters():
            p.requires_grad = not freeze


class CNNClassifier(nn.Module):
    """Stage 1: per-frame pain / no_pain classifier."""

    def __init__(self, embed_dim=config.EMBED_DIM):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.head = nn.Linear(embed_dim, 2)

    def forward(self, x):
        emb = self.encoder(x)
        return self.head(emb)


class PainCNNLSTM(nn.Module):
    """Stage 2: CNN spatial features -> LSTM temporal dynamics -> pain score."""

    def __init__(self, embed_dim=config.EMBED_DIM, hidden_dim=config.LSTM_HIDDEN):
        super().__init__()
        self.encoder = CNNEncoder(embed_dim=embed_dim)
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, T, C, H, W)
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        emb = self.encoder(x)                      # (B*T, embed_dim)
        emb = emb.view(b, t, -1)                    # (B, T, embed_dim)
        lstm_out, _ = self.lstm(emb)                 # (B, T, hidden_dim)
        scores = self.head(lstm_out).squeeze(-1)     # (B, T)
        scores = torch.clamp(scores, 0.0, config.PAIN_SCORE_MAX)
        return scores

    def load_cnn_weights(self, cnn_classifier_state_dict):
        """Transfer the encoder weights from a trained CNNClassifier checkpoint."""
        encoder_state = {k.replace("encoder.", ""): v
                          for k, v in cnn_classifier_state_dict.items() if k.startswith("encoder.")}
        self.encoder.load_state_dict(encoder_state)
