import json
import csv
from sys import base_prefix
import os

# Define paths and folder names
batch = [48,49]#[0,47]
for i in range(batch[0],batch[1]+1):
  basepath = f'D:\Chenqi\KP Detection\dataset\sim{str(i).zfill(2)}'

  # Gather basic information about the dataset
  for folder in os.listdir(basepath):
    if folder[:3]=='RGB':
      picfolder = folder
    if folder[:7]=='Dataset':
      jsonfolder = folder

  jsonfiles = os.listdir(os.path.join(basepath,jsonfolder))
  num_jsons = 0
  for jsonfile in jsonfiles:
    if jsonfile[:9] == 'captures_':
      num_jsons += 1

  # Initialize the csv file for keypoint locations
  traincsv = open(os.path.join(basepath, 'keypoints.csv'),'w',newline='')
  writer = csv.writer(traincsv)

  # Go through all the captures_XXX.json files and put into csv
  for k in range(0, num_jsons):
      # Read the json
      jsonname = f'captures_{str(k).zfill(3)}.json'
      f = open(os.path.join(basepath, jsonfolder, jsonname))
      captures = json.load(f)
      data = captures["captures"]

      # Go through each json file and write to csv
      for i in range(0, len(data)):
          keypoints = data[i]["annotations"][0]["values"][0]["keypoints"]
          filename = data[i]["filename"]
          filename = filename.replace(picfolder, '')

          entry = []
          entry.append(filename)

          for j in range(0, len(keypoints)):
              kp = keypoints[j]

              x = str(kp["x"])
              y = str(kp["y"])

              entry.append(x)
              entry.append(y)
          print(entry)
          writer.writerow(entry)
          f.flush()
      f.close()