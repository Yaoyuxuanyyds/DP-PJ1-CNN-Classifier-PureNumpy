from abc import abstractmethod
import cupy as cp


class Optimizer:
    def __init__(self, init_lr, model) -> None:
        self.init_lr = init_lr
        self.model = model

    @abstractmethod
    def step(self):
        pass


class SGD(Optimizer):
    def __init__(self, init_lr, model):
        super().__init__(init_lr, model)
    
    def step(self):
        for layer in self.model.layers:
            if layer.optimizable == True:
                for key in layer.params.keys():
                    if layer.weight_decay:
                        layer.params[key][...] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    layer.params[key][...] = layer.params[key] - self.init_lr * layer.grads[key]


class MomentGD(Optimizer):
    def __init__(self, init_lr, model, mu = 0.9):
        super().__init__(init_lr, model)
        self.mu = mu
        self.v = []
    
    def step(self):
        layers_opt = []
        for layer in self.model.layers:
            if layer.optimizable == True:
                layers_opt.append(layer)
                self.v.append({})

        for i, layer in enumerate(layers_opt):
                for key in layer.params.keys():
                    # initialize v
                    if key not in self.v[i].keys():
                        self.v[i][key] = cp.zeros_like(layer.params[key])
                    # update v
                    self.v[i][key] = self.mu * self.v[i][key] + (1 - self.mu) * layer.grads[key]
                    # weight decay
                    if layer.weight_decay:
                        layer.params[key][...] *= (1 - self.init_lr * layer.weight_decay_lambda)
                    # update params
                    layer.params[key][...] = layer.params[key] - self.init_lr * self.v[i][key]