from abc import ABCMeta, abstractmethod

class NoGradientError(Exception):
    pass

class BaseLoss(metaclass=ABCMeta):
    def __init__(self, args):
        self.kwargs = args

    @abstractmethod
    def compute(self, outputs_model, inputs_data):
        pass
