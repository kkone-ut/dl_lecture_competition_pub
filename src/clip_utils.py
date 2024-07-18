from PIL import Image
import torch
from torch import nn

from torch.utils.data import Dataset
import torch.nn.functional as F


class EEGImageDataset(Dataset):
    def __init__(self, eeg_data, image_paths, preprocess):
        self.eeg_data = eeg_data
        self.image_paths = image_paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.eeg_data)

    def __getitem__(self, idx):
        eeg = self.eeg_data[idx]
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.preprocess(image)
        return eeg, image


class EEGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(3, 3), padding=1),   # (1, 271, 281) -> (16, 271, 281)
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(4, 4)),                  # (16, 271, 281) -> (16, 67, 70)
            nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1),  # (16, 67, 70) -> (32, 67, 70)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),                  # (32, 67, 70) -> (32, 33, 35)
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # (32, 33, 35) -> (64, 33, 35)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),                  # (64, 33, 35) -> (64, 16, 17)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16 * 17, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        batch_size, seq_len, _ = x.size()    # (batch_size, 271, 281)
        x = x.view(batch_size, 1, 271, 281)  # (batch_size, 1, 271, 281)
        x = self.cnn(x)                      # (batch_size, 64, 16, 17)
        x = x.view(batch_size, -1)           # (batch_size, 64 * 16 * 17)
        x = self.fc(x)                       # (batch_size, 512)
        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)


class EEGCLIPModel(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.eeg_encoder = EEGEncoder()
        self.clip_model = clip_model
        self.eeg_encoder.init_weights()

    def forward(self, eeg, image):
        eeg_features = self.eeg_encoder(eeg)
        image_features = self.clip_model.encode_image(image)
        # 正規化
        eeg_features = F.normalize(eeg_features, dim=-1)
        image_features = F.normalize(image_features, dim=-1)
        return eeg_features.float(), image_features.float()

    def calculate_clip_loss(self, eeg_features, image_features):
        # 一致度を計算
        logits_per_eeg = eeg_features @ image_features.t()
        logits_per_image = image_features @ eeg_features.t()
        # ラベルを生成
        labels = torch.arange(logits_per_eeg.size(0), device=logits_per_eeg.device)
        # クロスエントロピー損失を計算
        loss_eeg = F.cross_entropy(logits_per_eeg, labels)
        loss_image = F.cross_entropy(logits_per_image, labels)
        return (loss_eeg + loss_image) / 2
