from abc import ABC,abstractmethod
import pandas as pd

class Data(ABC):
    def __init__(self,dates):
        pass

    @abstractmethod
    def addFeature(self, ta_list, dates, price, high, low):
        pass

    @abstractmethod
    def appendDate(self,dates):
        pass

    @abstractmethod
    def createDateFrame(self,dates,datalist,tags):
        pass

    @abstractmethod
    def getInputMatrix(self):
        pass