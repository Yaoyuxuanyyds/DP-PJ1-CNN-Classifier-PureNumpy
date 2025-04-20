from abc import abstractmethod
import cupy as cp

class scheduler():
    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.step_count = 0
    
    @abstractmethod
    def step():
        pass


class StepLR(scheduler):
    def __init__(self, optimizer, step_size=100, gamma=0.1, warmup_epoch=0, warmup_lr=1e-5) -> None:
        super().__init__(optimizer)
        self.step_size = step_size
        self.gamma = gamma
        self.warmup_epoch = warmup_epoch
        self.warmup_lr = warmup_lr
        self.base_lr = optimizer.init_lr  # 保存初始学习率
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1

        if self.step_count < self.warmup_epoch:
            # 线性 warmup，从 warmup_lr 增加到 base_lr
            lr = (self.base_lr - self.warmup_lr) / self.warmup_epoch * self.step_count + self.warmup_lr
        else:
            # 衰减阶段：按 epoch 计数来衰减
            decay_steps = (self.step_count - self.warmup_epoch) // self.step_size
            lr = self.base_lr * (self.gamma ** decay_steps)

        # 设置当前学习率（clip）
        self.optimizer.init_lr = max(lr, 1e-5)






class MultiStepLR(scheduler):
    def __init__(self, optimizer, milestones=[30, 60, 90], gamma=0.1) -> None:
        super().__init__(optimizer)
        self.milestones = milestones
        self.gamma = gamma

    def step(self) -> None:
        self.step_count += 1
        if self.step_count in self.milestones:
            self.optimizer.init_lr *= self.gamma


class ExponentialLR(scheduler):
    def __init__(self, optimizer, gamma=0.9) -> None:
        super().__init__(optimizer)
        self.gamma = gamma

    def step(self) -> None:
        self.optimizer.init_lr *= self.gamma
        self.optimizer.init_lr = max(self.optimizer.init_lr, 1e-5)
        