import itertools
import cupy as cp
import pickle
from struct import unpack
import gzip
import mynn as nn
import numpy as np
import pandas as pd
import os

# Load dataset
def load_data():
    with gzip.open('./dataset/MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
        _, num, _, _ = unpack('>4I', f.read(16))
        train_imgs = cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num, 28*28)
    with gzip.open('./dataset/MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
        _, num = unpack('>2I', f.read(8))
        train_labs = cp.frombuffer(f.read(), dtype=cp.uint8)
    
    idx = cp.random.permutation(cp.arange(num))
    train_imgs = train_imgs[idx]
    train_labs = train_labs[idx]
    valid_imgs = train_imgs[:10000] / 255.0
    valid_labs = train_labs[:10000]
    train_imgs = train_imgs[10000:] / 255.0
    train_labs = train_labs[10000:]

    return train_imgs, train_labs, valid_imgs, valid_labs

# Hyperparameter grid
lr_list = [5e-5, 1e-5, 5e-6]
reg_list = [1e-3]
# arch_list = [
#     [784, 128, 10],
#     [784, 256, 128, 10],
#     [784, 600, 128, 10]
# ]
warmup_epochs_list = [0, 100]
batch_size_list = [512, 1024, 2048]

results = []
train_imgs, train_labs, valid_imgs, valid_labs = load_data()

# Grid Search
# for lr, reg, arch, warmup_epoch, batch_size in itertools.product(
#     lr_list, reg_list, arch_list, warmup_epochs_list, batch_size_list
# ):import os

output_path = 'search_results_CNN.csv'
is_first_write = not os.path.exists(output_path)

for lr, reg, warmup_epoch, batch_size in itertools.product(
    lr_list, reg_list, warmup_epochs_list, batch_size_list
):
    cp.random.seed(309)
    print(f'Training with lr={lr}, reg={reg}, warmup={warmup_epoch}, batch={batch_size} ...')

    model = nn.models.Model_CNN()
    optimizer = nn.optimizer.SGD(init_lr=lr, model=model)
    scheduler = nn.lr_scheduler.StepLR(
        optimizer=optimizer,
        step_size=100,
        gamma=0.9,
        warmup_epoch=warmup_epoch,
        warmup_lr=1e-6
    )
    loss_fn = nn.modules.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max() + 1)
    runner = nn.trainer.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=batch_size, scheduler=scheduler)

    runner.train(
        [train_imgs, train_labs],
        [valid_imgs, valid_labs],
        num_epochs=8,
        log_iters=50,
        save_dir=f'./best_models/hyper_search_result/CNN-{lr}_{reg}_{warmup_epoch}_{batch_size}'
    )

    best_acc = runner.best_score
    row = {
        'lr': lr,
        'reg': reg,
        'warmup_epochs': warmup_epoch,
        'batch_size': batch_size,
        'best_val_acc': best_acc
    }
    df_row = pd.DataFrame([row])
    df_row.to_csv(output_path, mode='a', index=False, header=is_first_write)
    is_first_write = False  # 后续写入不再包含表头