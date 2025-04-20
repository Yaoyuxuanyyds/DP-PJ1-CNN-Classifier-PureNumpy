from abc import abstractmethod
import cupy as cp
from cupy.lib.stride_tricks import as_strided

class Layer():
    def __init__(self) -> None:
        self.optimizable = True
    
    @abstractmethod
    def forward():
        pass

    @abstractmethod
    def backward():
        pass




# --------------------------------------------------------------------------------------------------------------------
# Basic modules for a MLP
class Linear(Layer):
    """
    The linear layer for a neural network. You need to implement the forward function and the backward function.
    """
    def __init__(self, in_dim, out_dim, initialize_method=cp.random.normal, weight_decay=False, weight_decay_lambda=1e-8) -> None:
        super().__init__()
        self.W = initialize_method(size=(in_dim, out_dim))
        self.b = initialize_method(size=(1, out_dim))
        self.grads = {'W' : None, 'b' : None}
        self.input = None # Record the input for backward process.

        self.params = {'W' : self.W, 'b' : self.b}

        self.weight_decay = weight_decay # whether using weight decay
        self.weight_decay_lambda = weight_decay_lambda # control the intensity of weight decay
            
    
    def __call__(self, X) -> cp.ndarray:
        return self.forward(X)

    def forward(self, X):
        """
        input: [batch_size, in_dim]
        out: [batch_size, out_dim]
        """
        self.input = X
        # Y = X @ W + b
        output = cp.matmul(X, self.W) + self.b
        return output

    def backward(self, grad : cp.ndarray):
        """
        input: [batch_size, out_dim] the grad passed by the next layer.
        output: [batch_size, in_dim] the grad to be passed to the previous layer.
        This function also calculates the grads for W and b.
        """
        # Grad for W: X^T @ grad
        self.grads['W'] = cp.matmul(self.input.T, grad)  # [in_dim, batch_size] @ [batch_size, out_dim]
        # Grad for b: [1, batch_size] @ grad
        self.grads['b'] = cp.sum(grad, axis=0, keepdims=True)  # [1, out_dim]
        # Grad for input: grad @ W^T
        grad_input = cp.matmul(grad, self.W.T)  # [batch_size, in_dim]

        return grad_input
    
    def clear_grad(self):
        self.grads = {'W' : None, 'b' : None}




# --------------------------------------------------------------------------------------------------------------------
# Basic modules for a CNN

# class conv2D(Layer):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
#                  initialize_method=cp.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
#         super().__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding
#         self.weight_decay = weight_decay
#         self.weight_decay_lambda = weight_decay_lambda

#         self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))  # [C_out, C_in, kH, kW]
#         self.b = initialize_method(size=(out_channels, 1))  # [C_out, 1]
#         self.params = {'W': self.W, 'b': self.b}
#         self.grads = {'W': None, 'b': None}
#         self.input = None

#     def __call__(self, X):
#         return self.forward(X)

#     def forward(self, X):
#         self.input = X  # shape: [B, C_in, H, W]
#         B, C_in, H, W = X.shape
#         kH, kW = self.kernel_size
#         C_out = self.out_channels
#         s = self.stride
#         # output shape: [B, C_out, out_H, out_W]
#         out_H = (H - kH + 2 * self.padding) // s + 1
#         out_W = (W - kW + 2 * self.padding) // s + 1
#         output = cp.zeros((B, C_out, out_H, out_W))
#         # padding
#         if self.padding > 0:
#             X = cp.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
#         # convolution
#         for oc in range(C_out):
#             for i in range(out_H):
#                 for j in range(out_W):
#                     h_start, w_start = i * s, j * s
#                     patch = X[:, :, h_start:h_start+kH, w_start:w_start+kW]  # [B, C_in, kH, kW]
#                     w = self.W[oc].reshape((1, C_in, kH, kW))    # [C_in, kH, kW] -> [B, C_in, kH, kW]
#                     output[:, oc, i, j] = cp.sum(patch * w, axis=(1, 2, 3)) + self.b[oc]
#         return output


#     def backward(self, grad_output):
#         X = self.input
#         B, C_in, H, W = X.shape
#         kH, kW = self.kernel_size
#         s = self.stride
#         C_out = self.out_channels
#         # grad from last layer: [B, C_out, out_H, out_W]
#         out_H, out_W = grad_output.shape[2], grad_output.shape[3]
#         # padding
#         if self.padding > 0:
#             X_padded = cp.pad(X, ((0,0), (0,0), (self.padding,self.padding), (self.padding,self.padding)), mode='constant')
#         else:
#             X_padded = X

#         dX_padded = cp.zeros_like(X_padded)
#         dW = cp.zeros_like(self.W)
#         db = cp.zeros_like(self.b)

#         for b in range(B):
#             for oc in range(C_out):
#                 for i in range(out_H):
#                     for j in range(out_W):
#                         h_start, w_start = i * s, j * s
#                         patch = X_padded[b, :, h_start:h_start+kH, w_start:w_start+kW]
#                         # dw
#                         dW[oc] += patch * grad_output[b, oc, i, j]
#                         # db
#                         db[oc] += grad_output[b, oc, i, j]
#                         # dX
#                         dX_padded[b, :, h_start:h_start+kH, w_start:w_start+kW] += self.W[oc] * grad_output[b, oc, i, j]

#         if self.padding > 0:
#             dX = dX_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
#         else:
#             dX = dX_padded

#         self.grads['W'] = dW
#         self.grads['b'] = db

#         return dX

#     def clear_grad(self):
#         self.grads = {
#             'W': cp.zeros_like(self.W),
#             'b': cp.zeros_like(self.b)
#         }



class conv2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 initialize_method=cp.random.normal, weight_decay=False, weight_decay_lambda=1e-8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight_decay = weight_decay
        self.weight_decay_lambda = weight_decay_lambda

        self.W = initialize_method(size=(out_channels, in_channels, *self.kernel_size))  # [out, C_in, kH, kW]
        self.b = initialize_method(size=(out_channels, 1))  # [out, 1]
        self.params = {'W': self.W, 'b': self.b}
        self.grads = {'W': None, 'b': None}
        self.input = None

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C_in, H, W = X.shape
        kH, kW = self.kernel_size
        C_out = self.out_channels
        s = self.stride
        p = self.padding

        if p > 0:
            X = cp.pad(X, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

        out_H = (H - kH + 2 * p) // s + 1
        out_W = (W - kW + 2 * p) // s + 1

        # 构造滑动窗口
        shape = (B, C_in, out_H, out_W, kH, kW)
        strides = (
            X.strides[0],
            X.strides[1],
            s * X.strides[2],
            s * X.strides[3],
            X.strides[2],
            X.strides[3],
        )
        windows = as_strided(X, shape=shape, strides=strides)  # [B, C_in, out_H, out_W, kH, kW]
        self.windows = windows

        # 利用 einsum 高效计算卷积
        # W: [C_out, C_in, kH, kW] -> 与 windows [B, C_in, out_H, out_W, kH, kW]
        output = cp.einsum('bihwuv,oiuv->bohw', windows, self.W)  # [B, C_out, out_H, out_W]
        output += self.b.reshape(1, -1, 1, 1)
        return output


    def backward(self, grad_output):
        B, C_in, H, W = self.input.shape
        kH, kW = self.kernel_size
        s = self.stride
        p = self.padding
        C_out = self.out_channels
        out_H, out_W = grad_output.shape[2:]

        dW = cp.einsum('bihwuv,bohw->oiuv', self.windows, grad_output)
        db = grad_output.sum(axis=(0, 2, 3), keepdims=True).reshape(self.b.shape)

        # 计算输入梯度
        dX_padded = cp.zeros((B, C_in, H + 2 * p, W + 2 * p))
        flipped_W = self.W[:, :, ::-1, ::-1]  # 卷积核旋转 180 度

        # 反卷积将梯度传回输入
        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * s, j * s
                grad_slice = grad_output[:, :, i, j]  # [B, C_out]
                # 把梯度映射回输入区域
                dX_padded[:, :, h_start:h_start+kH, w_start:w_start+kW] += cp.einsum('bo,oiuv->biuv', grad_slice, flipped_W)

        dX = dX_padded[:, :, p:H + p, p:W + p] if p > 0 else dX_padded
        self.grads['W'] = dW
        self.grads['b'] = db
        return dX


    def clear_grad(self):
        self.grads = {
            'W': cp.zeros_like(self.W),
            'b': cp.zeros_like(self.b)
        }

# class MaxPool2D(Layer):
#     """
#     2D Max Pooling layer (no learnable parameters).
#     Input: [B, C, H, W]
#     Output: [B, C, out_H, out_W]
#     """
#     def __init__(self, kernel_size=2, stride=2, padding=0) -> None:
#         super().__init__()
#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding  # 不支持 padding，只是保留接口
#         self.input = None
#         self.max_indices = None  # 记录最大值的位置用于反向传播
#         self.optimizable = False

#     def __call__(self, X):
#         return self.forward(X)

#     def forward(self, X):
#         """
#         X: [B, C, H, W]
#         output: [B, C, out_H, out_W]
#         """
#         self.input = X
#         B, C, H, W = X.shape
#         kH, kW = self.kernel_size
#         s = self.stride

#         out_H = (H - kH) // s + 1
#         out_W = (W - kW) // s + 1
#         output = cp.zeros((B, C, out_H, out_W))
#         self.max_indices = cp.zeros((B, C, out_H, out_W, 2), dtype=cp.int32)

#         for b in range(B):
#             for c in range(C):
#                 for i in range(out_H):
#                     for j in range(out_W):
#                         h_start, w_start = i * s, j * s
#                         window = X[b, c, h_start:h_start+kH, w_start:w_start+kW]
#                         max_val = cp.max(window)
#                         output[b, c, i, j] = max_val
#                         # 找到最大值的位置，并标记下来用于反向传播
#                         max_pos = cp.unravel_index(cp.argmax(window), window.shape)
#                         self.max_indices[b, c, i, j, 0] = h_start + max_pos[0]
#                         self.max_indices[b, c, i, j, 1] = w_start + max_pos[1]
#         return output

#     def backward(self, grads):
#         """
#         grads: [B, C, out_H, out_W]  grad from last layer
#         grad_input: [B, C, H, W]   grad to input X
#         """
#         B, C, H, W = self.input.shape
#         out_H, out_W = grads.shape[2], grads.shape[3]

#         grad_input = cp.zeros_like(self.input)

#         for b in range(B):
#             for c in range(C):
#                 for i in range(out_H):
#                     for j in range(out_W):
#                         max_h, max_w = self.max_indices[b, c, i, j]
#                         grad_input[b, c, max_h, max_w] += grads[b, c, i, j]
#         return grad_input



class MaxPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.input = None
        self.max_indices = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        kH, kW = self.kernel_size
        s = self.stride

        out_H = (H - kH) // s + 1
        out_W = (W - kW) // s + 1

        # 计算滑窗的 shape 和 strides
        shape = (B, C, out_H, out_W, kH, kW)
        strides = (
            X.strides[0],  # B
            X.strides[1],  # C
            s * X.strides[2],  # H
            s * X.strides[3],  # W
            X.strides[2],  # kH
            X.strides[3],  # kW
        )

        windows = as_strided(X, shape=shape, strides=strides)
        # shape: (B, C, out_H, out_W, kH, kW)

        output = cp.max(windows, axis=(4, 5))  # 在滑动窗口内取最大
        # 记录最大值的位置，用于反向传播
        max_mask = (windows == output[..., None, None])
        self.max_indices = max_mask.astype(cp.uint8)

        return output

    def backward(self, grads):
        B, C, H, W = self.input.shape
        kH, kW = self.kernel_size
        s = self.stride
        out_H, out_W = grads.shape[2], grads.shape[3]

        grad_input = cp.zeros_like(self.input)

        shape = (B, C, out_H, out_W, kH, kW)
        strides = (
            grad_input.strides[0],
            grad_input.strides[1],
            s * grad_input.strides[2],
            s * grad_input.strides[3],
            grad_input.strides[2],
            grad_input.strides[3],
        )
        grad_windows = as_strided(grad_input, shape=shape, strides=strides)

        grad_windows += (self.max_indices * grads[..., None, None])

        return grad_input


# class AvgPool2D(Layer):
#     """
#     2D Average Pooling layer (no learnable parameters).
#     Input: [B, C, H, W]
#     Output: [B, C, out_H, out_W]
#     """
#     def __init__(self, kernel_size=2, stride=2, padding=0) -> None:
#         super().__init__()

#         self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
#         self.stride = stride
#         self.padding = padding  # 不支持 padding，只是保留接口
#         self.input = None
#         self.optimizable = False

#     def __call__(self, X):
#         return self.forward(X)

#     def forward(self, X):
#         self.input = X
#         B, C, H, W = X.shape
#         kH, kW = self.kernel_size
#         s = self.stride

#         out_H = (H - kH) // s + 1
#         out_W = (W - kW) // s + 1
#         output = cp.zeros((B, C, out_H, out_W))

#         for b in range(B):
#             for c in range(C):
#                 for i in range(out_H):
#                     for j in range(out_W):
#                         h_start, w_start = i * s, j * s
#                         window = X[b, c, h_start:h_start+kH, w_start:w_start+kW]
#                         avg_val = cp.mean(window)
#                         output[b, c, i, j] = avg_val
#         return output

#     def backward(self, grads):
#         """
#         grads: [B, C, out_H, out_W]  grad from last layer
#         grad_input: [B, C, H, W]   grad to input X
#         """
#         B, C, H, W = self.input.shape
#         out_H, out_W = grads.shape[2], grads.shape[3]

#         grad_input = cp.zeros_like(self.input)

#         for b in range(B):
#             for c in range(C):
#                 for i in range(out_H):
#                     for j in range(out_W):
#                         h_start, w_start = i * self.stride, j * self.stride
#                         grad_input[b, c, h_start:h_start+self.kernel_size[0], w_start:w_start+self.kernel_size[1]] += grads[b, c, i, j] / (self.kernel_size[0] * self.kernel_size[1])
#         return grad_input   
    

class AvgPool2D(Layer):
    def __init__(self, kernel_size=2, stride=2, padding=0) -> None:
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        kH, kW = self.kernel_size
        s = self.stride

        out_H = (H - kH) // s + 1
        out_W = (W - kW) // s + 1

        # 计算滑窗的 shape 和 strides
        shape = (B, C, out_H, out_W, kH, kW)
        strides = (
            X.strides[0],  # B
            X.strides[1],  # C
            s * X.strides[2],  # H
            s * X.strides[3],  # W
            X.strides[2],  # kH
            X.strides[3],  # kW
        )

        windows = as_strided(X, shape=shape, strides=strides)
        # shape: (B, C, out_H, out_W, kH, kW)

        output = cp.mean(windows, axis=(4, 5))  # 在滑动窗口内取平均

        return output

    def backward(self, grads):
        B, C, H, W = self.input.shape
        kH, kW = self.kernel_size
        s = self.stride
        out_H, out_W = grads.shape[2], grads.shape[3]

        grad_input = cp.zeros_like(self.input)

        shape = (B, C, out_H, out_W, kH, kW)
        strides = (
            grad_input.strides[0],
            grad_input.strides[1],
            s * grad_input.strides[2],
            s * grad_input.strides[3],
            grad_input.strides[2],
            grad_input.strides[3],
        )
        grad_windows = as_strided(grad_input, shape=shape, strides=strides)

        grad_windows += (grads[..., None, None] / (kH+kH))

        return grad_input
    

class Flatten(Layer):
    """
    Flatten layer (no learnable parameters).
    Input: [B, C, H, W]
    Output: [B, C*H*W]
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None
        self.optimizable = False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        B, C, H, W = X.shape
        output = X.reshape(B, -1)
        return output
    
    def backward(self, grads):
        B, C, H, W = self.input.shape
        grad_input = grads.reshape(B, C, H, W)
        return grad_input






# --------------------------------------------------------------------------------------------------------------------
# Activation functions
class ReLU(Layer):
    """
    An activation layer.
    """
    def __init__(self) -> None:
        super().__init__()
        self.input = None

        self.optimizable =False

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        self.input = X
        output = cp.where(X<0, 0, X)
        return output
    
    def backward(self, grads):
        assert self.input.shape == grads.shape
        output = cp.where(self.input < 0, 0, grads)
        return output








# --------------------------------------------------------------------------------------------------------------------
# Loss functions
class MultiCrossEntropyLoss(Layer):
    """
    A multi-cross-entropy loss layer, with Softmax layer in it, which could be cancelled by method cancel_softmax
    """
    def __init__(self, model = None, max_classes = 10) -> None:
        super().__init__()
        self.model = model
        self.max_classes = max_classes
        self.probs = None
        self.one_hot_label = None
        self.grads = None
        self.loss = None
        

    def __call__(self, predicts, labels):
        return self.forward(predicts, labels)
    
    def forward(self, predicts, labels):
        """
        predicts: [batch_size, D] (before softmax)
        labels: [batch_size, ] 
        This function generates the loss.
        """
        assert predicts.shape[1] == self.max_classes
        # Softmax
        probs = self.softmax(predicts)
        self.probs = probs
        # Turn labels into one-hot
        self.one_hot_label = cp.zeros_like(predicts)
        self.one_hot_label[cp.arange(len(labels)), labels] = 1
        # Compute cross-entropy loss
        log_probs = cp.log(probs + 1e-12)
        loss = -cp.sum(self.one_hot_label * log_probs) / len(labels)
        self.loss = loss

        return loss

    def backward(self):
        """
        First compute the grads from the loss to the input.
        Then send the grads to model for back propagation
        """
        batch_size = self.probs.shape[0]
        # Grads for logit: softmax-prob - one_hot
        self.grads = (self.probs - self.one_hot_label) / batch_size
        # Send grads to model
        self.model.backward(self.grads)

    def softmax(self, X):
        x_max = cp.max(X, axis=1, keepdims=True)
        x_exp = cp.exp(X - x_max)
        partition = cp.sum(x_exp, axis=1, keepdims=True)
        return x_exp / partition




# --------------------------------------------------------------------------------------------------------------------
# Regularization
class L2Regularization(Layer):
    """
    L2 Reg can act as weight decay that can be implemented in class Linear.
    """
    pass
       

        
