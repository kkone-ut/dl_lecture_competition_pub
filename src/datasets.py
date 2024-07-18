import os

import torch


class ThingsMEGDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, data_dir: str = "data", filter_flag=True, baseline_flag=True) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # 画像データ
        if baseline_flag:
            X_path = os.path.join(data_dir, f"{split}_X_baseline_corrected.pt")
        elif filter_flag:
            X_path = os.path.join(data_dir, f"{split}_X_filtered.pt")
        else:
            X_path = os.path.join(data_dir, f"{split}_X.pt")
        self.X = torch.load(X_path)
        # 4人の被験者のうち、どの被験者かを示すインデックス
        self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))

        if split in ["train", "val"]:
            # 正解クラス
            self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
            assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

    # len({インスタンス名})
    def __len__(self) -> int:
        return len(self.X)

    # {インスタンス名}[i]
    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    # {インスタンス名}.num_channels
    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    # {インスタンス名}.seq_len
    @property
    def seq_len(self) -> int:
        return self.X.shape[2]
