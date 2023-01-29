from abc import ABCMeta, abstractmethod


class BasicLoss(metaclass=ABCMeta):
    def __init__(self, args):
        self.kwargs = args

    @abstractmethod
    def compute(self, outputs_model, inputs_data):
        pass
