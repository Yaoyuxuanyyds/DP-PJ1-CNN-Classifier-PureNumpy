# codes to make visualization of your weights.
import mynn as nn
import cupy as cp
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'./best_models/saved_models/best_Model_CNN.pickle')


mats = []
for layer in model.layers:
    if layer.optimizable:
        mats.append(layer.params['W'].get())

weights = mats[2]  # (out_channels, in_channels, 3, 3)

out_channels, in_channels, kh, kw = weights.shape
fig, axes = plt.subplots(out_channels, in_channels, figsize=(in_channels * 2, out_channels * 2))

for i in range(out_channels):
    for j in range(in_channels):
        ax = axes[i, j] if in_channels > 1 else axes[i]
        kernel = weights[i, j]
        ax.matshow(kernel, cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])

plt.tight_layout()
plt.savefig('./figs/conv3_kernels.png')
plt.show()


# _, axes = plt.subplots(1, 3)
# _.set_tight_layout(1)
# axes = axes.reshape(-1)
# for i in range(3):
#     axes[i].matshow(mats[i])
#     axes[i].set_xticks([])
#     axes[i].set_yticks([])
# plt.show()
# plt.savefig('./figs/weights_CNN.png')

