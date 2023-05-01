import csv
import numpy as np
import os
from collections import defaultdict

# Define paths and folder names
basepath = 'D:\Chenqi\KP Detection\dataset\RealSampleDatasetFineCam'

# Read annotation csv
data = np.loadtxt(os.path.join(basepath,'annotations.csv'), delimiter=',', dtype=str)
assert data.shape[0] % 5 == 0

# Initialize output np array
annotations = np.empty((int(data.shape[0]/5),11),list)
for ind, row in enumerate(data[::5]):
    fname = '/'+row[3]
    annotations[ind,0] = str(fname)
    annotations[ind,1] = data[ind*5,:][1]
    annotations[ind,2] = data[ind*5,:][2]
    annotations[ind,3] = data[ind*5+1,:][1]
    annotations[ind,4] = data[ind*5+1,:][2]
    annotations[ind,5] = data[ind*5+2,:][1]
    annotations[ind,6] = data[ind*5+2,:][2]
    annotations[ind,7] = data[ind*5+3,:][1]
    annotations[ind,8] = data[ind*5+3,:][2]
    annotations[ind,9] = data[ind*5+4,:][1]
    annotations[ind,10] = data[ind*5+4,:][2]

print(annotations)


# Initialize the csv file for keypoint locations
with open(os.path.join(basepath, 'keypoints_output.csv'),'w+',newline='') as traincsv:
    writer = csv.writer(traincsv)
    for row in annotations:
        writer.writerow(row)
