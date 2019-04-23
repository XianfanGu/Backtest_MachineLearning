import csv

class Info:
    def __init__(self,pathname):
        self.info = self.readCSV(pathname)
    def readCSV(self,pathname):
        n = 0
        mydict = {}
        with open(pathname, mode='r') as infile:
            reader = csv.reader(infile)
            for rows in reader:
                if not n == 0:
                    mydict[rows[0]] = rows[2]
                else:
                    n = 1
        return mydict
    def get_symbol_list(self):
        return list(self.info.keys())