# An example of read in the data and train the model. The runner is implemented, while the model used for training need your implementation.
import mynn as nn
from draw_tools.plot import plot, plot_compare

import cupy as cp
import numpy as np
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

# fixed seed for experiment
cp.random.seed(309)

train_images_path = r'./dataset/MNIST/train-images-idx3-ubyte.gz'
train_labels_path = r'./dataset/MNIST/train-labels-idx1-ubyte.gz'

with gzip.open(train_images_path, 'rb') as f:
        magic, num, rows, cols = unpack('>4I', f.read(16))
        train_imgs=cp.frombuffer(f.read(), dtype=cp.uint8).reshape(num, 28*28)
    
with gzip.open(train_labels_path, 'rb') as f:
        magic, num = unpack('>2I', f.read(8))
        train_labs = cp.frombuffer(f.read(), dtype=cp.uint8)


# choose 10000 samples from train set as validation set.
idx = cp.random.permutation(cp.arange(num))
# save the index.
with open('idx.pickle', 'wb') as f:
        pickle.dump(idx, f)
train_imgs = train_imgs[idx]
train_labs = train_labs[idx]
valid_imgs = train_imgs[:10000]
valid_labs = train_labs[:10000]
train_imgs = train_imgs[10000:]
train_labs = train_labs[10000:]

# normalize from [0, 255] to [0, 1]
train_imgs = train_imgs / train_imgs.max()
valid_imgs = valid_imgs / valid_imgs.max()



# Init the model
# model = nn.models.Model_MLP([train_imgs.shape[-1], 600, 128, 10], 'ReLU', [1e-3, 1e-3, 1e-3])
# model = nn.models.Model_CNN()
model = nn.models.Model_CNN_timing()

# Choose the optimizer
optimizer = nn.optimizer.SGD(init_lr=0.00001, model=model)
# optimizer = nn.optimizer.MomentGD(init_lr=0.00001, model=model)

# Choose the scheduler
# scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
scheduler = nn.lr_scheduler.StepLR(optimizer=optimizer, step_size=100, gamma=0.9, warmup_epoch=100, warmup_lr=1e-6)
# scheduler = nn.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9)

# Choose the loss function
loss_fn = nn.modules.MultiCrossEntropyLoss(model=model, max_classes=train_labs.max()+1)
# Set the runner
runner = nn.trainer.RunnerM(model, optimizer, nn.metric.accuracy, loss_fn, batch_size=512, scheduler=scheduler)

runner.train([train_imgs, train_labs], [valid_imgs, valid_labs], num_epochs=5, log_iters=10, save_dir=r'./best_models')

_, axes = plt.subplots(1, 2)
axes.reshape(-1)
_.set_tight_layout(1)
plot(runner, axes)
plt.show()
 
# # 数据增强
# train_imgs_aug = train_imgs.reshape(-1, 1, 28, 28)
# train_imgs_aug = nn.augment.random_translate(train_imgs_aug)
# train_imgs_aug = nn.augment.random_rotate(train_imgs_aug)
# train_imgs_aug = nn.augment.add_noise(train_imgs_aug)
# # 恢复为一维向量输入模型
# train_imgs_aug = train_imgs_aug.reshape(-1, 784)


# # compare
# model1 = nn.models.Model_MLP([train_imgs_aug.shape[-1], 600, 128, 10], 'ReLU', [1e-3, 1e-3, 1e-3])
# optimizer1 = nn.optimizer.SGD(init_lr=0.01, model=model1)
# # optimizer1 = nn.optimizer.MomentGD(init_lr=0.01, model=model1)

# # scheduler = nn.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[800, 2400, 4000], gamma=0.5)
# scheduler1 = nn.lr_scheduler.StepLR(optimizer=optimizer1, step_size=100, gamma=0.9, warmup_epoch=100, warmup_lr=1e-5)
# loss_fn1 = nn.modules.MultiCrossEntropyLoss(model=model1, max_classes=train_labs.max()+1)
# runner1 = nn.trainer.RunnerM(model1, optimizer1, nn.metric.accuracy, loss_fn1, batch_size=256, scheduler=scheduler1)

# runner1.train([train_imgs_aug, train_labs], [valid_imgs, valid_labs], num_epochs=6, log_iters=10, save_dir=r'./best_models')

# _, axes = plt.subplots(1, 2)
# axes.reshape(-1)
# _.set_tight_layout(1)
# plot_compare(runner,runner1, axes, name="MLP_compare_aug")
# plt.show()