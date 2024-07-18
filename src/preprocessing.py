from argparse import ArgumentParser
from time import time

import numpy as np
import torch
from scipy import signal
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', type=str, default='train')
    parser.add_argument('-i', type=int, default=0)
    args = parser.parse_args()
    data_type = args.t
    idx = args.i

    start_time = time()
    print('start_time: ', start_time - start_time)

    data_dir = '../data/'
    train_X_path = data_dir + 'train_X.pt'
    val_X_path = data_dir + 'val_X.pt'
    test_X_path = data_dir + 'test_X.pt'

    # データを読み込む
    if data_type == 'train':
        X = torch.load(train_X_path)
    elif data_type == 'val':
        X = torch.load(val_X_path)
    elif data_type == 'test':
        X = torch.load(test_X_path)

    # 処理が重いため、データを分割して行う
    data_size = X.size(0)
    split_size = data_size // 16
    X = X[idx * split_size:(idx + 1) * split_size]
    print(data_type, idx, X.size(0))

    dataset_time = time()
    print('dataset_time: ', dataset_time - start_time)

    fs = 200
    lowcut = 0.1
    highcut = 40.0

    X_filtered = filter_data(X, fs, lowcut, highcut)
    X_baseline_corrected = baseline_correct_data(X_filtered)

    # データを保存
    torch.save(X_filtered, data_dir + f'{data_type}_X_filtered_{idx}.pt')
    torch.save(X_baseline_corrected, data_dir + f'{data_type}_X_baseline_corrected_{idx}.pt')


# バンドパスフィルタの適用
def filter_data(data, fs, lowcut, highcut):
    nyq = 0.5 * fs  # ナイキスト周波数
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(5, [low, high], btype="band")
    filterd_data = np.empty_like(data)
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            filterd_data[i, j, :] = signal.filtfilt(b, a, data[i, j, :])
    return torch.tensor(filterd_data)


# ベースライン補正の適用
# -100ms~1300msのデータのうち、-100ms~0msのデータをベースラインとして補正
def baseline_correct_data(data):
    baseline_corrected_data = np.empty_like(data)
    for i in tqdm(range(data.shape[0])):
        for j in range(data.shape[1]):
            baseline = np.mean(data[i, j, :20].numpy())
            baseline_corrected_data[i, j, :] = data[i, j, :] - baseline
    return torch.tensor(baseline_corrected_data)


if __name__ == '__main__':
    main()
