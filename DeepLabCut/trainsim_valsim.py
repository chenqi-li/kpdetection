from cProfile import run
from distutils import dep_util
from distutils.command.config import config
from email import header
from imp import source_from_cache
from pydoc import doc
from cv2 import sort
import deeplabcut
import os
import pickle
import yaml
import shutil
import pandas as pd
import sys
from PIL import Image
from tqdm import tqdm
import csv
import warnings 
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load training parameters.')
    parser.add_argument('--mode', type=str)
    # Training parameters
    parser.add_argument('--sim_version', type=str)
    parser.add_argument('--run_num', type=int)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--train_test_split', type=float)
    parser.add_argument('--shuffle', type=int)
    args = parser.parse_args()




    sim_version = args.sim_version
    run_num = args.run_num
    train_test_split = args.train_test_split
    num_epoch = args.num_epoch
    root_dir = 'D:\\Chenqi\\KP Detection\\DeepLabCut'
    run_dir = f'sim{str(sim_version)}_run{str(run_num).zfill(2)}'
    deeplabcut.create_new_project(run_dir, 'chenqi', ['D:\\Chenqi\KP Detection\\dataset_video\\RealSampleDatasetFineCam\\20220711162306.avi'], working_directory=root_dir, copy_videos=True, multianimal=False)

    # Find the current directory for this run
    files = os.listdir(root_dir)
    for file in files:
        if file[0:len(run_dir)] == run_dir:
            run_dir = file

    # Load and edit the yaml file
    config_yaml_file = os.path.join(root_dir, run_dir, 'config.yaml')
    with open(config_yaml_file, 'rb') as f:
        config_yaml = yaml.load(f,Loader=yaml.SafeLoader)
    config_yaml['bodyparts'] = ['kp0', 'kp1', 'kp2', 'kp3', 'kp4']
    config_yaml['video_sets'] = {f'D:\\Chenqi\\KP Detection\\DeepLabCut\\{run_dir}\\videos\\RGB.avi': {'crop': '0, 4032, 0, 3040'}}
    config_yaml['TrainingFraction'] = [train_test_split]
    config_yaml['snapshotindex'] = 'all'
    with open(config_yaml_file, 'w') as yamlfile:
        data = yaml.dump(config_yaml, yamlfile, sort_keys=False)

    # Copy images into labeled_data
    source_root = os.path.join('D:\\Chenqi\\KP Detection\\dataset',f'sim{str(sim_version)}')
    for folder in os.listdir(source_root):
        if folder[0:3] == 'RGB':
            source = os.path.join(source_root, folder)
    target = os.path.join(root_dir,run_dir,'labeled-data','RGB')
    if not os.path.exists(target):
        os.makedirs(target)
        print('Copying and resizing dataset')
        imgExts = ["png", "jpg"]
        for path, dirs, files in os.walk(source):
            for fileName in tqdm(files):
                ext = fileName[-3:].lower()
                if ext not in imgExts:
                    continue
                filePath = os.path.join(source, fileName)
                im = Image.open(filePath)
                newIm = im.resize((224, 224))
                original_size = im.size
                newIm.save(os.path.join(target,fileName))

    # Create csv in dlc format if not done so
    if not os.path.exists(os.path.join(root_dir,run_dir,'labeled-data','RGB','CollectedData_chenqi.csv')):
        csv_source = os.path.join('D:\\Chenqi\\KP Detection\\dataset',f'sim{str(sim_version)}','keypoints.csv')    
        keypoints = pd.read_csv(csv_source, header=None)
        csv_target = open(os.path.join(root_dir,run_dir,'labeled-data','CollectedData_chenqi.csv'),'w+')
        writer = csv.writer(csv_target)
        #original_size = (1008, 760)
        keypoints.iloc[:,1:11:2] = keypoints.iloc[:,1:11:2]/original_size[0]*224
        keypoints.iloc[:,2:11:2] = keypoints.iloc[:,2:11:2]/original_size[1]*224
        keypoints.iloc[:,0] = 'labeled-data/RGB/'+keypoints.iloc[:,0].str[1:]
        my_header = pd.DataFrame([['scorer'] + ['chenqi']*10, ['bodyparts', 'kp0', 'kp0', 'kp1', 'kp1', 'kp2', 'kp2', 'kp3', 'kp3', 'kp4', 'kp4'], ['coords'] + ['x', 'y']*5])
        out_pd = pd.concat([my_header, keypoints])
        print(out_pd)
        out_pd.to_csv(os.path.join(root_dir,run_dir,'labeled-data','RGB','CollectedData_chenqi.csv'),',',index=False, header=False)
        deeplabcut.convertcsv2h5(config_yaml_file, scorer= 'chenqi', userfeedback=False)

    # Create training set
    deeplabcut.create_training_dataset(config_yaml_file, num_shuffles=args.shuffle)

    # Find dataset size from pkl file
    dataset_folder = os.path.join(root_dir, run_dir, 'training-datasets', 'iteration-0')
    for root, dirs, files in os.walk(dataset_folder):
        if not dirs:
            for file in files:
                if file[-6:] == 'pickle':
                    documentation_pickle_file = os.path.join(root, file)
    with open(documentation_pickle_file, 'rb') as f:
        documentation_pickle = pickle.load(f)
    train_size = documentation_pickle[-3].shape[0]
    val_size = documentation_pickle[-2].shape[0]
    split = documentation_pickle[-1]
    print(train_size, val_size, split)

    # Train and evaluate all the shuffles
    for shuffle_num in range(1,args.shuffle+1):
        deeplabcut.train_network(config_yaml_file,displayiters=int(train_size/10),saveiters=train_size*5,max_snapshots_to_keep=num_epoch, maxiters=train_size*num_epoch, shuffle=shuffle_num)
    # deeplabcut.evaluate_network(config_yaml_file, Shuffles=range(1,2))  #Shuffles=range(1,args.shuffle+1)) for dataset14 and before, Shuffles=range(1,2) for dataset15 and beyond to save time