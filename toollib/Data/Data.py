from abc import ABC,abstractmethod
import pandas as pd

class Data(ABC):
    def __init__(self,dates):
        pass

    @abstractmethod
    def addFeature(self, feature_name_list, dates, inputs):
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