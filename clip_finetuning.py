import os
from time import time

import clip
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import wandb
from omegaconf import DictConfig
from termcolor import cprint
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.clip_utils import EEGCLIPModel, EEGImageDataset
from src.utils import EarlyStopping, set_seed
from torch.cuda.amp import GradScaler, autocast


# 使いたい config を変更したい場合、config_name を変更する
@hydra.main(version_base=None, config_path="configs", config_name="config_clip")
def run(args: DictConfig):
    """
    Args:
        args (DictConfig): configuration defined in {config_path}/{config_name}.yaml
    """
    set_seed(args.seed)
    logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    if args.use_wandb:
        wandb.init(mode="online", dir=logdir, project="MEG-CLIP")

    print('args: ', args)
    start_time = time()
    print('start_time: ', start_time - start_time)

    # ------------------
    #    CLIPモデル
    # ------------------
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device, jit=False)

    # ------------------
    #    Dataloader
    # ------------------
    train_X_path = os.path.join(args.data_dir, 'train_X.pt')
    val_X_path = os.path.join(args.data_dir, 'val_X.pt')

    # データを読み込む
    train_X = torch.load(train_X_path)
    val_X = torch.load(val_X_path)

    # 画像パスを読み込む
    train_image_paths_txt_path = os.path.join(args.data_dir, 'train_image_paths.txt')
    with open(train_image_paths_txt_path, 'r') as f:
        train_image_paths = [os.path.join(args.image_dir, path.strip()) for path in f.readlines()]

    val_image_paths_txt_path = os.path.join(args.data_dir, 'val_image_paths.txt')
    with open(val_image_paths_txt_path, 'r') as f:
        val_image_paths = [os.path.join(args.image_dir, path.strip()) for path in f.readlines()]

    # データ拡張
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        preprocess
    ])

    # データセットとデータローダの作成
    train_dataset = EEGImageDataset(train_X, train_image_paths, transform)
    val_dataset = EEGImageDataset(val_X, val_image_paths, preprocess)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    dataset_time = time()
    print('dataset_time: ', dataset_time - start_time)

    # ------------------
    #      Model等
    # ------------------
    model = EEGCLIPModel(clip_model).to(args.device)
    model.float()  # float32に変換
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_dataloader), epochs=args.epochs)
    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=os.path.join(logdir, "checkpoint.pt"))
    scaler = GradScaler()

    min_val_loss = np.inf

    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")

        train_loss, val_loss = [], []
        train_clip_loss, train_mse_loss = [], []
        val_clip_loss, val_mse_loss = [], []

        model.train()
        for egg, image in tqdm(train_dataloader, desc="Train"):
            optimizer.zero_grad()
            egg, image = egg.to(args.device), image.to(args.device)

            with autocast():
                eeg_features, image_features = model(egg, image)

                clip_loss = model.calculate_clip_loss(eeg_features, image_features)
                mse_loss = criterion(eeg_features, image_features)
                loss = clip_loss + mse_loss
            train_loss.append(loss.item())
            train_clip_loss.append(clip_loss.item())
            train_mse_loss.append(mse_loss.item())

            # backward
            scaler.scale(loss).backward()
            # クリップ時に正しくできるように一度スケールを戻す
            scaler.unscale_(optimizer)
            # 大きすぎる勾配をクリップ
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            # パラメタの更新
            scaler.step(optimizer)
            # スケールの更新
            scaler.update()
        scheduler.step()

        model.eval()
        for egg, image in tqdm(val_dataloader, desc="Validation"):
            egg, image = egg.to(args.device), image.to(args.device)

            with autocast():
                with torch.no_grad():
                    eeg_features, image_features = model(egg, image)

                    clip_loss = model.calculate_clip_loss(eeg_features, image_features)
                    mse_loss = criterion(eeg_features, image_features)
                    loss = clip_loss + mse_loss
            val_loss.append(loss.item())
            val_clip_loss.append(clip_loss.item())
            val_mse_loss.append(mse_loss.item())

        print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | val loss: {np.mean(val_loss):.3f}")
        print(f"Epoch {epoch+1}/{args.epochs} | train clip loss: {np.mean(train_clip_loss):.3f} | val clip loss: {np.mean(val_clip_loss):.3f}")
        print(f"Epoch {epoch+1}/{args.epochs} | train mse loss: {np.mean(train_mse_loss):.3f} | val mse loss: {np.mean(val_mse_loss):.3f}")
        torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))

        if args.use_wandb:
            wandb.log({"train_loss": np.mean(train_loss), "val_loss": np.mean(val_loss)})

        if np.mean(val_loss) < min_val_loss:
            cprint("New best.", "cyan")
            torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
            torch.save(model.state_dict(), os.path.join('models/clip_models/', "model_best.pt"))
            min_val_loss = np.mean(val_loss)

        # early stopping
        early_stopping(np.mean(val_loss), model)

        if early_stopping.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    run()
