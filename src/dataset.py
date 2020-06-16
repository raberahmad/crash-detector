import pandas as pd

class Dataset:
    def __init__(self, data):
        self.data = data



    def importData(self):
        toReturn = pd.read_excel(self.data, index_col=0)
        return toReturn

