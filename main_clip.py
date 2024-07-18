import os
from time import time

import hydra
import numpy as np
import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torchmetrics import Accuracy
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.models import *
from src.utils import CosineScheduler, EarlyStopping, set_lr, set_seed
import clip
from src.clip_utils import EEGCLIPModel, EEGImageDataset
from torch.cuda.amp import GradScaler, autocast


# 使いたい config を変更したい場合、config_name を変更する
@hydra.main(version_base=None, config_path="configs", config_name="config_main_clip")
def run(args: DictConfig):
    """
    Args:
        args (DictConfig): configuration defined in {config_path}/{config_name}.yaml
    """
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-classification")

    print('args: ', args)
    start_time = time()
    print('start_time: ', start_time - start_time)

    # ------------------
    #    Dataloader
    # ------------------
    loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
    train_set = ThingsMEGDataset("train", args.data_dir, filter_flag=args.filter_flag, baseline_flag=args.baseline_flag)
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
    val_set = ThingsMEGDataset("val", args.data_dir, filter_flag=args.filter_flag, baseline_flag=args.baseline_flag)
    val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
    test_set = ThingsMEGDataset("test", args.data_dir, filter_flag=args.filter_flag, baseline_flag=args.baseline_flag)
    test_loader = torch.utils.data.DataLoader(
        test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers
    )

    dataset_time = time()
    print('dataset_time: ', dataset_time - start_time)

    # ------------------
    #       Model
    # ------------------
    # models/clip_models/model_best.pt からモデルを読み込む
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)
    eeg_clip_model = EEGCLIPModel(clip_model)
    eeg_clip_model.load_state_dict(torch.load("models/clip_models/model_best.pt"))
    eeg_clip_model.to(args.device)
    model = CLIPEncoder(eeg_clip_model, train_set.num_classes).to(args.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader), epochs=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(logdir, "checkpoint.pt"))

    # ------------------
    #   Start training
    # ------------------
    max_val_acc = 0
    accuracy = Accuracy(
        task="multiclass", num_classes=train_set.num_classes, top_k=10
    ).to(args.device)

    scaler = GradScaler()

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, train_acc, val_loss, val_acc = [], [], [], []

        model.train()
        for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
            X, y = X.to(args.device), y.to(args.device)

            with autocast():
                y_pred = model(X)
                loss = F.cross_entropy(y_pred, y)
            train_loss.append(loss.item())

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            scaler.step(optimizer)
            scaler.update()
            acc = accuracy(y_pred, y)
            train_acc.append(acc.item())
        scheduler.step()

        model.eval()
        for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
            X, y = X.to(args.device), y.to(args.device)

            with torch.no_grad():
                with autocast():
                    y_pred = model(X)
                    loss = F.cross_entropy(y_pred, y)
            val_loss.append(loss.item())
            acc = accuracy(y_pred, y)
            val_acc.append(acc.item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

        if np.mean(val_acc) > max_val_acc:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            max_val_acc = np.mean(val_acc)

        # early stopping
        early_stopping(np.mean(val_loss), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

    # ----------------------------------
    #  Start evaluation with best model
    # ----------------------------------
    model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

    preds = []
    # preds: (test_size, num_classes)
    # preds[i]: i枚目のテストデータの予測結果
    # preds[i][j]: i枚目のテストデータのクラスjの確率 (正規化されていない気がする)
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):
        preds.append(model(X.to(args.device)).detach().cpu())

    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(logdir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {logdir}", "cyan")


if __name__ == "__main__":
    run()
