import numpy as np
import nnfs
from nnfs.datasets import vertical_data
import csv

nnfs.init()

X, y = vertical_data(samples=100, classes=3)

with open('data.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',', 
            quotechar='|', quoting=csv.QUOTE_MINIMAL)
    for i in range(len(X)):
        filewriter.writerow(X[i])

print(y)
