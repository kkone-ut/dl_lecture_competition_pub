from argparse import ArgumentParser

import torch


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', type=str, default='train')
    args = parser.parse_args()
    data_type = args.t

    data_dir = '../data/'

    filterd_data = []
    baseline_correctd_data = []

    # データを読み込む
    if data_type == 'train':
        for i in range(16):
            # fdata = torch.load(data_dir + f'{data_type}_X_filtered_{i}.pt')
            bdata = torch.load(data_dir + f'{data_type}_X_baseline_corrected_{i}.pt')
            # filterd_data.append(fdata)
            baseline_correctd_data.append(bdata)
        # X_filtered = torch.cat(filterd_data, dim=0)
        X_baseline_corrected = torch.cat(baseline_correctd_data, dim=0)

    elif data_type == 'val':
        for i in range(4):
            fdata = torch.load(data_dir + f'{data_type}_X_filtered_{i}.pt')
            bdata = torch.load(data_dir + f'{data_type}_X_baseline_corrected_{i}.pt')
            filterd_data.append(fdata)
            baseline_correctd_data.append(bdata)
        X_filtered = torch.cat(filterd_data, dim=0)
        X_baseline_corrected = torch.cat(baseline_correctd_data, dim=0)

    elif data_type == 'test':
        for i in range(4):
            fdata = torch.load(data_dir + f'{data_type}_X_filtered_{i}.pt')
            bdata = torch.load(data_dir + f'{data_type}_X_baseline_corrected_{i}.pt')
            filterd_data.append(fdata)
            baseline_correctd_data.append(bdata)
        X_filtered = torch.cat(filterd_data, dim=0)
        X_baseline_corrected = torch.cat(baseline_correctd_data, dim=0)

    # データを保存
    torch.save(X_filtered, data_dir + f'{data_type}_X_filtered.pt')
    torch.save(X_baseline_corrected, data_dir + f'{data_type}_X_baseline_corrected.pt')


if __name__ == '__main__':
    main()
