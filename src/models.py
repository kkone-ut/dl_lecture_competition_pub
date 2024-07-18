import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class BasicConvClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        seq_len: int,
        in_channels: int,
        hid_dim: int = 128
    ) -> None:
        super().__init__()

        self.blocks = nn.Sequential(
            ConvBlock(in_channels, hid_dim),
            ConvBlock(hid_dim, hid_dim),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(hid_dim, num_classes),
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """_summary_
        Args:
            X ( b, c, t ): _description_
        Returns:
            X ( b, num_classes ): _description_
        """
        X = self.blocks(X)

        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size: int = 3,
        p_drop: float = 0.1,
    ) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")
        # self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size) # , padding="same")

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        # X = self.conv2(X)
        # X = F.glu(X, dim=-2)

        return self.dropout(X)


class myEEGNet(nn.Module):
    def __init__(self, num_classes, in_channels=271, seq_len=281, dropout_rate=0.25):
        super().__init__()

        # Block 1: (batch_size, 1, 271, 281) -> (batch_size, 16, 271, 281)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same', bias=False),  # (batch_size, 1, 271, 281) -> (batch_size, 16, 271, 281)
            nn.BatchNorm2d(16),
        )

        # Block 2 (Depthwise Convolution): (batch_size, 16, 271, 281) -> (batch_size, 32, 1, 70)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(in_channels, 1), padding=(0, 0), groups=16, bias=False),  # (batch_size, 16, 271, 281) -> (batch_size, 32, 1, 281)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 32, 1, 281) -> (batch_size, 32, 1, 70)
            nn.Dropout(dropout_rate)
        )

        # Block 3 (Separable Convolution): (batch_size, 32, 1, 70) -> (batch_size, 64, 1, 17)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 16), padding='same', groups=32, bias=False),  # (batch_size, 32, 1, 70) -> (batch_size, 64, 1, 70)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 32, 1, 70) -> (batch_size, 64, 1, 17)
            nn.Dropout(dropout_rate)
        )

        # Block 4 (Skip Connection)
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 16), padding='same', bias=False),  # (batch_size, 64, 1, 17) -> (batch_size, 64, 1, 17)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Dropout(dropout_rate)
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 17, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 271, 281) -> (batch_size, 1, 271, 281)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        residual = self.conv4(x)
        x += residual
        x = self.classify(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class myEEG_v2(nn.Module):
    def __init__(self, num_classes, in_channels=271, seq_len=281, dropout_rate=0.5):
        super().__init__()

        # Block 1: (batch_size, 1, 271, 281) -> (batch_size, 16, 271, 281)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(1, 64), padding='same', bias=False),  # (batch_size, 1, 271, 281) -> (batch_size, 16, 271, 281)
            nn.BatchNorm2d(16),
            SEBlock(16),  # Attention block
        )

        # Block 2 (Depthwise Convolution): (batch_size, 16, 271, 281) -> (batch_size, 32, 1, 70)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(in_channels, 1), padding=(0, 0), groups=16, bias=False),  # (batch_size, 16, 271, 281) -> (batch_size, 32, 1, 281)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 32, 1, 281) -> (batch_size, 32, 1, 70)
            nn.Dropout(dropout_rate),
            SEBlock(32),  # Attention block
        )

        # Block 3 (Separable Convolution): (batch_size, 32, 1, 70) -> (batch_size, 32, 1, 70)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=(1, 16), padding='same', groups=32, bias=False),  # (batch_size, 32, 1, 70) -> (batch_size, 32, 1, 70)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.Dropout(dropout_rate),
            SEBlock(32),  # Attention block
        )

        self.classify = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 70, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 271, 281) -> (batch_size, 1, 271, 281)
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3(x)
        x += residual
        x = self.classify(x)
        return x


class EEGNet_v1(nn.Module):
    def __init__(self, num_classes, in_channels=271, seq_len=281, dropout_rate=0.5):
        super().__init__()

        # First Convolution Block
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(in_channels, 1), padding=(0, 0), bias=False),  # (batch_size, 1, 271, 281) -> (batch_size, 16, 1, 281)
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=(1, 64), padding=(0, 32), bias=False, groups=16),  # (batch_size, 16, 1, 281) -> (batch_size, 16, 1, 281)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 16, 1, 281) -> (batch_size, 16, 1, 70)
            nn.Dropout(dropout_rate)
        )

        # Depthwise Convolution Block 1
        self.depthwiseConv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 8), padding=(0, 4), groups=16, bias=False),  # (batch_size, 16, 1, 70) -> (batch_size, 32, 1, 70)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 32, 1, 70) -> (batch_size, 32, 1, 17)
            nn.Dropout(dropout_rate)
        )

        # Depthwise Convolution Block 2
        self.depthwiseConv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 4), padding=(0, 2), groups=32, bias=False),  # (batch_size, 32, 1, 17) -> (batch_size, 64, 1, 17)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 64, 1, 17) -> (batch_size, 64, 1, 4)
            nn.Dropout(dropout_rate)
        )

        # Separable Convolution Block
        self.separableConv = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=(1, 4), padding=(0, 2), bias=False),  # (batch_size, 64, 1, 4) -> (batch_size, 64, 1, 4)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 64, 1, 4) -> (batch_size, 64, 1, 1)
            nn.Dropout(dropout_rate)
        )

        # Classification Block
        self.classify = nn.Sequential(
            nn.Flatten(),  # (batch_size, 64, 1, 1) -> (batch_size, 64)
            nn.Linear(64, num_classes)  # (batch_size, 64) -> (batch_size, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 271, 281) -> (batch_size, 1, 271, 281)
        x = self.firstconv(x)
        x = self.depthwiseConv1(x)
        x = self.depthwiseConv2(x)
        x = self.separableConv(x)
        x = self.classify(x)
        return x


class EEGNet_v2(nn.Module):
    def __init__(self, num_classes, in_channels=271, seq_len=281, dropout_rate=0.5):
        super().__init__()

        # First Convolution Block
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=(in_channels, 1), padding=(0, 0), bias=False),  # (batch_size, 1, 271, 281) -> (batch_size, 16, 1, 281)
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 16, 1, 281) -> (batch_size, 16, 1, 70)
            nn.Dropout(dropout_rate)
        )

        # Depthwise Convolution Block 1
        self.depthwiseConv1 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=(1, 8), padding=(0, 4), groups=16, bias=False),  # (batch_size, 16, 1, 70) -> (batch_size, 32, 1, 70)
            nn.BatchNorm2d(32),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 32, 1, 70) -> (batch_size, 32, 1, 17)
            nn.Dropout(dropout_rate)
        )

        # Depthwise Convolution Block 2
        self.depthwiseConv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(1, 4), padding=(0, 2), groups=32, bias=False),  # (batch_size, 32, 1, 17) -> (batch_size, 64, 1, 17)
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 64, 1, 17) -> (batch_size, 64, 1, 4)
            nn.Dropout(dropout_rate)
        )

        # Separable Convolution Block
        self.separableConv = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 4), padding=(0, 2), bias=False),  # (batch_size, 64, 1, 4) -> (batch_size, 128, 1, 4)
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),  # (batch_size, 128, 1, 4) -> (batch_size, 128, 1, 1)
            nn.Dropout(dropout_rate)
        )

        # Attention Mechanism
        self.attention = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=(1, 1)),  # (batch_size, 128, 1, 1) -> (batch_size, 1, 1, 1)
            nn.Softmax(dim=1)
        )

        # Classification Block
        self.classify = nn.Sequential(
            nn.Flatten(),  # (batch_size, 128, 1, 1) -> (batch_size, 128)
            nn.Linear(128, num_classes)  # (batch_size, 128) -> (batch_size, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # Add channel dimension: (batch_size, 271, 281) -> (batch_size, 1, 271, 281)
        x = self.firstconv(x)
        x = self.depthwiseConv1(x)
        x = self.depthwiseConv2(x)
        x = self.separableConv(x)
        attn = self.attention(x)
        x = x * attn
        x = self.classify(x)
        return F.softmax(x, dim=1)


class CLIPEncoder(nn.Module):
    def __init__(self, eeg_clip_model, num_classes):
        super().__init__()
        self.eeg_clip_model = eeg_clip_model
        self.eeg_encoder = self.eeg_clip_model.eeg_encoder
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.eeg_encoder(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class TransformerModel(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)  # (input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model * seq_len, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x += self.positional_encoding  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.view(x.size(0), -1)  # (batch_size, seq_len * d_model)
        x = self.fc(x)  # (batch_size, num_classes)
        return x


class Transformer_v2(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=256, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.3):
        """
        Args:
            input_dim (int): 入力次元数
            seq_len (int): シーケンス長
            num_classes (int): クラス数
            d_model (int): モデルの次元数
            nhead (int): ヘッド数
            num_layers (int): レイヤー数
            dim_feedforward (int): feedforward の次元数
            dropout (float): ドロップアウト率
        """
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.batch_norm = nn.BatchNorm1d(seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x += self.positional_encoding  # (batch_size, seq_len, d_model)
        x = self.dropout(x)  # (batch_size, seq_len, d_model)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = self.layer_norm(x)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

    def _generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding


class Transformer_v3(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=512, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.batch_norm = nn.BatchNorm1d(seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self._init_weights()  # 重みの初期化

    def forward(self, x):
        x = self.embedding(x)
        x += self.positional_encoding
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

    def _generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)


class Transformer_v4(nn.Module):
    def __init__(self, input_dim, seq_len, num_classes, d_model=384, nhead=12, num_layers=8, dim_feedforward=1024, dropout=0.3):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.d_model = d_model

        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(self._generate_positional_encoding(seq_len, d_model))
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        self.batch_norm = nn.BatchNorm1d(seq_len)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, num_classes)
        self._init_weights()  # 重みの初期化

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, d_model)
        x += self.positional_encoding
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)  # (batch_size, seq_len, d_model)
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.fc(x)  # (batch_size, num_classes)
        return x

    def _generate_positional_encoding(self, seq_len, d_model):
        position = torch.arange(seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        positional_encoding = torch.zeros(seq_len, d_model)
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        positional_encoding = positional_encoding.unsqueeze(0)
        return positional_encoding

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        nn.init.xavier_uniform_(self.fc.weight)
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
