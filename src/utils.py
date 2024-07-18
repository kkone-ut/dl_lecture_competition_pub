import random

import numpy as np
import torch


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path=None):
        """
        Args:
            patience (int): val_lossが改善されないエポック数の許容値
            verbose (bool): Trueの場合、早期終了のメッセージを出力
            delta (float): 直前のベストスコアとの比較で改善とみなされる最小差異
            path (str): 保存パス
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.path = path

    def __call__(self, val_loss, model):
        score = - val_loss

        if self.best_score is None:  # 1Epoch目
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:  # 更新できない場合
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:  # 更新できる場合
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        バリデーション損失が改善された際に現在のモデルを保存する
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class CosineScheduler:
    def __init__(self, epochs, lr, warmup_length=5):
        """
        Args:
            epochs (int): 学習のエポック数
            lr (float): 初期学習率
            warmup_length (int): warmupを適用するエポック数
        """
        self.epochs = epochs
        self.lr = lr
        self.warmup = warmup_length

    def __call__(self, epoch):
        """
        Args:
            epoch (int): 現在のエポック数
        """
        progress = (epoch - self.warmup) / (self.epochs - self.warmup)
        progress = np.clip(progress, 0.0, 1.0)
        lr = self.lr * 0.5 * (1. + np.cos(np.pi * progress))
        if self.warmup:
            lr = lr * min(1., (epoch+1) / self.warmup)
        return lr


def set_lr(lr, optimizer):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
