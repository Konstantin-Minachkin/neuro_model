from abc import ABCMeta, abstractmethod

class regress_model(metaclass=ABCMeta):
    @abstractmethod
    def predict(self,x,y):
        pass

    @abstractmethod
    def train(self,x,y):
        pass

