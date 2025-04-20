from .modules import *
import pickle
import time


class Model_MLP(Layer):
    """
    A model with linear layers. We provied you with this example about a structure of a model.
    """
    def __init__(self, size_list=None, act_func=None, lambda_list=None):
        self.size_list = size_list  # list of the size of each layer
        self.act_func = act_func

        if size_list is not None and act_func is not None:
            self.layers = []
            for i in range(len(size_list) - 1):
                layer = Linear(in_dim=size_list[i], out_dim=size_list[i + 1])
                if lambda_list is not None:
                    layer.weight_decay = True
                    layer.weight_decay_lambda = lambda_list[i]
                if act_func == 'Logistic':
                    raise NotImplementedError
                elif act_func == 'ReLU':
                    layer_f = ReLU()
                self.layers.append(layer)
                # Add activation function
                if i < len(size_list) - 2:
                    self.layers.append(layer_f)

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        assert self.size_list is not None and self.act_func is not None, 'Model has not initialized yet. Use model.load_model to load a model or create a new model with size_list and act_func offered.'
        outputs = X
        for layer in self.layers:
            outputs = layer(outputs)
        return outputs

    def backward(self, loss_grad):
        # Grad is passed by the loss function
        grads = loss_grad
        # Backpropagation 
        for layer in reversed(self.layers):
            grads = layer.backward(grads)
        return grads

    def load_model(self, param_list):
        with open(param_list, 'rb') as f:
            param_list = pickle.load(f)

        self.size_list = param_list[0]
        self.act_func = param_list[1]
        self.layers = []

        param_dicts = param_list[2:]

        for i in range(len(self.size_list) - 1):
            layer = Linear(in_dim=self.size_list[i], out_dim=self.size_list[i + 1])
            params = param_dicts[i]
            layer.W = cp.asarray(params['W'])
            layer.b = cp.asarray(params['b'])
            
            layer.params['W'] = layer.W
            layer.params['b'] = layer.b
            layer.weight_decay = params.get('weight_decay', False)
            layer.weight_decay_lambda = params.get('lambda', 1e-8)
            self.layers.append(layer)

            # Add activation function (no activation function for the last layer)
            if i < len(self.size_list) - 2:
                if self.act_func == 'ReLU':
                    self.layers.append(ReLU())
                elif self.act_func == 'Logistic':
                    raise NotImplementedError("Logistic activation not implemented.")

        
    def save_model(self, save_path):
        param_list = [self.size_list, self.act_func]
        for layer in self.layers:
            if layer.optimizable:
                param_list.append({'W' : layer.params['W'], 'b' : layer.params['b'], 'weight_decay' : layer.weight_decay, 'lambda' : layer.weight_decay_lambda})
        
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)
        

class Model_CNN(Layer):
    """
    A model with conv2D layers. The structure of the model is similar to the LeNet model.
    """
    def __init__(self, in_channels=1, out_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.layers = [
            conv2D(in_channels=self.in_channels, out_channels=8, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            conv2D(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            conv2D(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            ReLU(),
            MaxPool2D(kernel_size=2, stride=2),

            Flatten(),
            Linear(in_dim=32 * 3 * 3, out_dim=784),
            ReLU(),
            Linear(in_dim=784, out_dim=128),
            ReLU(),
            Linear(in_dim=128, out_dim=self.out_dim),
        ]

    def __call__(self, X):
        return self.forward(X)

    def forward(self, X):
        # Reshape the input
        if len(X.shape) != 4:
            X = X.reshape(X.shape[0], self.in_channels, 28, 28)

        for layer in self.layers:
            X = layer(X)
        return X

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)
    
    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                layer_dict = {}
                for k, v in layer.params.items():
                    layer_dict[k] = v.get()  # 从 GPU 拿到 CPU cupy
                param_list.append(layer_dict)
            else:
                param_list.append(None)  # 非参数层填 None
        with open(save_path, 'wb') as f:
            pickle.dump(param_list, f)


    def load_model(self, model_path):
        with open(model_path, 'rb') as f:
            param_list = pickle.load(f)
        for layer, layer_params in zip(self.layers, param_list):
            if layer_params is not None and hasattr(layer, 'params'):
                for k in layer_params:
                    layer.params[k] = cp.array(layer_params[k])  # 转回 GPU






class Model_CNN_timing(Layer):
    """
    CNN模型(计时版)
    """
    def __init__(self, in_channels=1, out_dim=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.time_stats = {}  # 记录各层耗时

        self.layers = [
            conv2D(in_channels=self.in_channels, out_channels=4, kernel_size=3, stride=1, padding=1), 
            # ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            conv2D(in_channels=4, out_channels=8, kernel_size=3, stride=1, padding=1), 
            # ReLU(),
            MaxPool2D(kernel_size=2, stride=2),
            Flatten(),
            Linear(in_dim=8 * 7 * 7, out_dim=64),  
            ReLU(),
            Linear(in_dim=64, out_dim=32),  
            ReLU(),
            Linear(in_dim=32, out_dim=self.out_dim),
        ]
        
        # 初始化计时器
        for i, layer in enumerate(self.layers):
            self.time_stats[f'layer_{i}_{layer.__class__.__name__}'] = {
                'forward': 0,
                'backward': 0
            }

    def __call__(self, X):
        return self.forward(X)
    
    
    def forward(self, X):
        """带时间统计的前向传播"""
        if len(X.shape) != 4:
            X = X.reshape(X.shape[0], self.in_channels, 28, 28)
            
        for i, layer in enumerate(self.layers):
            start_time = time.time()
            X = layer(X)
            self.time_stats[f'layer_{i}_{layer.__class__.__name__}']['forward'] += time.time() - start_time
            
        return X

    def backward(self, loss_grad):
        """带时间统计的反向传播"""
        for i, layer in enumerate(reversed(self.layers)):
            start_time = time.time()
            loss_grad = layer.backward(loss_grad)
            self.time_stats[f'layer_{len(self.layers)-1-i}_{layer.__class__.__name__}']['backward'] += time.time() - start_time

    def print_time_stats(self):
        """打印各层耗时统计"""
        print("\n=== 各层耗时分析 ===")
        for name, times in self.time_stats.items():
            print(f"{name.ljust(25)}: 前向 {times['forward']:.4f}s | 反向 {times['backward']:.4f}s")
        
        total_forward = sum(t['forward'] for t in self.time_stats.values())
        total_backward = sum(t['backward'] for t in self.time_stats.values())
        print(f"\n总耗时: 前向 {total_forward:.4f}s | 反向 {total_backward:.4f}s")

        # 初始化计时器
        for i, layer in enumerate(self.layers):
            self.time_stats[f'layer_{i}_{layer.__class__.__name__}'] = {
                'forward': 0,
                'backward': 0
            }


    def load_model(self, param_list):
        idx = 0
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for k in layer.params:
                    layer.params[k] = param_list[idx]
                    idx += 1

    def save_model(self, save_path):
        param_list = []
        for layer in self.layers:
            if hasattr(layer, 'params') and layer.params:
                for k in layer.params:
                    param_list.append(layer.params[k])
        cp.save(save_path, param_list)
