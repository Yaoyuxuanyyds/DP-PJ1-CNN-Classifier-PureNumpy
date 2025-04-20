# codes to make visualization of your weights.
import mynn as nn
import cupy as cp
from struct import unpack
import gzip
import matplotlib.pyplot as plt
import pickle

model = nn.models.Model_CNN()
model.load_model(r'./saved_models/best_Model_CNN.pickle')


mats = []
for layer in model.layers:
    if layer.optimizable:
        mats.append(layer.params['W'].get())
print(len(mats))

_, axes = plt.subplots(1, 3)
_.set_tight_layout(1)
axes = axes.reshape(-1)
for i in range(3):
    axes[i].matshow(mats[i])
    axes[i].set_xticks([])
    axes[i].set_yticks([])
plt.show()
plt.savefig('./figs/weights_CNN.png')

