import cupy as cp 
import mynn as nn


# === 1. 初始化 Conv2D 层 ===
conv = nn.modules.conv2D(
    in_channels=1,
    out_channels=1,
    kernel_size=3,
    stride=1,
    padding=0,
    initialize_method=lambda size: cp.ones(size) * 0.1  # 固定值方便验证
)

# === 2. 构造输入和目标 ===
X = cp.arange(25).reshape(1, 1, 5, 5).astype(cp.float32)  # [1, 1, 5, 5]
target = cp.ones((1, 1, 3, 3), dtype=cp.float32)          # 期望输出为全1，方便loss计算

# === 3. 前向传播和 loss ===
out = conv(X)  # [1, 1, 3, 3]
loss = cp.mean((out - target) ** 2)
print("Forward output:\n", out)
print("Loss:", loss)

# === 4. 反向传播 ===
grad_out = 2 * (out - target) / cp.prod(out.shape)
grad_input = conv.backward(grad_out)

print("Gradient w.r.t input:\n", grad_input)
print("Gradient w.r.t weight:\n", conv.grads['W'])
print("Gradient w.r.t bias:\n", conv.grads['b'])

# === 5. 数值梯度检验 ===
epsilon = 1e-5
W_original = conv.W.copy()
numerical_grad = cp.zeros_like(conv.W)

for oc in range(conv.W.shape[0]):
    for ic in range(conv.W.shape[1]):
        for i in range(conv.W.shape[2]):
            for j in range(conv.W.shape[3]):
                conv.W = W_original.copy()
                conv.W[oc, ic, i, j] += epsilon
                out_plus = conv(X)
                loss_plus = cp.mean((out_plus - target) ** 2)

                conv.W = W_original.copy()
                conv.W[oc, ic, i, j] -= epsilon
                out_minus = conv(X)
                loss_minus = cp.mean((out_minus - target) ** 2)

                numerical_grad[oc, ic, i, j] = (loss_plus - loss_minus) / (2 * epsilon)

# 比较差异
analytic_grad = conv.grads['W']
difference = cp.abs(numerical_grad - analytic_grad)

print("\n=== Gradient Check ===")
print("Numerical Grad:\n", numerical_grad)
print("Analytic Grad:\n", analytic_grad)
print("Absolute Difference:\n", difference)
print("Max Difference:", cp.max(difference))