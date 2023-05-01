


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
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load training parameters.')
    parser.add_argument('--mode', type=str)
    # Training parameters
    parser.add_argument('--sim_version', type=str)
    parser.add_argument('--run_num', type=int)
    parser.add_argument('--num_epoch', type=int)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--finetune_dataset', type=str)
    parser.add_argument('--train_test_split', type=float)
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
    source_root = os.path.join('D:\\Chenqi\\KP Detection\\dataset',args.finetune_dataset)
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
        csv_source = os.path.join('D:\\Chenqi\\KP Detection\\dataset',args.finetune_dataset,'keypoints.csv')    
        keypoints = pd.read_csv(csv_source, header=None)
        csv_target = open(os.path.join(root_dir,run_dir,'labeled-data','CollectedData_chenqi.csv'),'w+')
        writer = csv.writer(csv_target)
        #original_size = (1008, 760)
        keypoints.iloc[:,1:11:2] = keypoints.iloc[:,1:11:2]/original_size[0]*224
        keypoints.iloc[:,2:11:2] = keypoints.iloc[:,2:11:2]/original_size[1]*224
        keypoints.iloc[:,0] = 'labeled-data/RGB/'+keypoints.iloc[:,0].str[1:]
        my_header = pd.DataFrame([['scorer'] + ['chenqi']*10, ['bodyparts', 'kp0', 'kp0', 'kp1', 'kp1', 'kp2', 'kp2', 'kp3', 'kp3', 'kp4', 'kp4'], ['coords'] + ['x', 'y']*5])
        out_pd = pd.concat([my_header, keypoints])
        out_pd.to_csv(os.path.join(root_dir,run_dir,'labeled-data','RGB','CollectedData_chenqi.csv'),',',index=False, header=False)
        deeplabcut.convertcsv2h5(config_yaml_file, scorer= 'chenqi', userfeedback=False)

    # Find shuffle directories of ckpt
    shuffle_dirs = os.listdir(os.path.join(args.ckpt, 'dlc-models', 'iteration-0'))

    # Create training set
    deeplabcut.create_training_dataset(config_yaml_file, num_shuffles=len(shuffle_dirs))

    # Get checkpoint paths of all pretrained weights
    shuffle_ckpts = []
    for shuffle_dir in sorted(shuffle_dirs):
        files = os.listdir(os.path.join(root_dir, args.ckpt, 'dlc-models', 'iteration-0', shuffle_dir, 'train'))
        files = sorted(files, key=lambda t: -os.stat(os.path.join(root_dir, args.ckpt, 'dlc-models', 'iteration-0', shuffle_dir, 'train', t)).st_mtime)

        candidates = []
        for file in files:
            if '.data-00000-of-00001' in file:
                candidates.append(file)

        iterations = [int(c[c.find('snapshot-')+9:c.find('.')]) for c in candidates]
        max_iter = np.argmax(iterations)
        shuffle_ckpts.append(os.path.join(root_dir, args.ckpt, 'dlc-models', 'iteration-0', shuffle_dir, 'train', candidates[max_iter].split('.')[0]))
        # f = open(os.path.join(root_dir, args.ckpt, 'dlc-models', 'iteration-0', shuffle_dir, 'train', 'checkpoint'))
        # checkpoint = f.readlines()
        # last_ckpt_path =checkpoint[0].split(': ')[1][1:-2]
        # shuffle_ckpts.append(last_ckpt_path)

    print(shuffle_ckpts)

    # Change init_weight as pretrained weights
    subdirs = os.listdir(os.path.join(root_dir, run_dir, 'dlc-models', 'iteration-0'))
    for i, subdir in enumerate(sorted(subdirs)):
        pose_cfg_file = os.path.join(root_dir, run_dir, 'dlc-models', 'iteration-0', subdir, 'train', 'pose_cfg.yaml')
        
        with open(pose_cfg_file, 'rb') as f:
            pose_cfg = yaml.load(f, Loader=yaml.SafeLoader)
        pose_cfg['init_weights'] = shuffle_ckpts[i]
        print(os.path.exists(shuffle_ckpts[i]))
        with open(pose_cfg_file, 'w') as yamlfile:
            data = yaml.dump(pose_cfg, yamlfile, sort_keys=False)



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
    for shuffle_num in range(1,len(shuffle_dirs)+1):
        deeplabcut.train_network(config_yaml_file,displayiters=int(train_size/10),saveiters=train_size,max_snapshots_to_keep=num_epoch, maxiters=train_size*num_epoch, shuffle=shuffle_num)
    deeplabcut.evaluate_network(config_yaml_file, Shuffles=range(1,2)) #Shuffles=range(1,args.shuffle+1)) for dataset14 and before, Shuffles=range(1,2) for dataset15 and beyond to save time

















































































# from cProfile import run
# from distutils import dep_util
# from distutils.command.config import config
# from email import header
# from imp import source_from_cache
# from pydoc import doc
# from cv2 import sort
# import deeplabcut
# import os
# import pickle
# import yaml
# import shutil
# import pandas as pd
# import sys
# from PIL import Image
# from tqdm import tqdm
# import csv
# import warnings 
# import argparse
# from distutils.dir_util import copy_tree


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description='Load training parameters.')
#     parser.add_argument('--num_epoch', default=1, type=int)
#     parser.add_argument('--train_test_split', default=1.0, type=float)
#     parser.add_argument('--run_dir', default='sim01_run01-chenqi-2022-07-22', type=str)
#     # Evaluation parameters
#     parser.add_argument('--sim_version', default=0, type=int)
#     parser.add_argument('--run_num', default=0, type=int)
#     parser.add_argument('--real_dir', default='D:\Chenqi\KP Detection\dataset\SampleReal10', type=str)
#     parser.add_argument('--suffix', default='finetune', type=str)
#     args = parser.parse_args()

    # # for folder in os.listdir('D:\Chenqi\KP Detection\DeepLabCut'):
    # #     if folder[0:11] == f'sim{str(args.sim_version).zfill(2)}_run{str(args.run_num).zfill(2)}':
    # #         run_dir = folder

    # # Copy entire folder to prepare for real evaluation
    # root_dir = 'D:\\Chenqi\\KP Detection\\DeepLabCut'
    # run_dir = os.path.join(root_dir,args.run_dir)
    # run_dir_real = run_dir+'_real'+args.suffix
    # if not os.path.exists(os.path.join(root_dir,run_dir_real)):
    #     copy_tree(os.path.join(root_dir,run_dir), os.path.join(root_dir,run_dir_real))
    #     shutil.rmtree(os.path.join(root_dir,run_dir_real,'evaluation-results'))
    #     shutil.rmtree(os.path.join(root_dir,run_dir_real,'labeled-data'))
    #     os.makedirs(os.path.join(root_dir,run_dir_real,'labeled-data'))
    #     shutil.rmtree(os.path.join(root_dir,run_dir_real,'training-datasets'))


    # # Load and edit the yaml file
    # config_yaml_file = os.path.join(root_dir, run_dir_real, 'config.yaml')
    # with open(config_yaml_file, 'rb') as f:
    #     config_yaml = yaml.load(f,Loader=yaml.SafeLoader)
    # config_yaml['project_path'] = os.path.join(root_dir, run_dir_real)
    # with open(config_yaml_file, 'w') as yamlfile:
    #     data = yaml.dump(config_yaml, yamlfile, sort_keys=False)


    # # Copy images into labeled_data
    # source_root = args.real_dir
    # for folder in os.listdir(source_root):
    #     if folder[0:3] == 'RGB':
    #         source = os.path.join(source_root, folder)
    # target = os.path.join(root_dir, run_dir_real,'labeled-data','RGB')
    # if not os.path.exists(target):
    #     os.makedirs(target)
    #     print('Copying and resizing dataset')
    #     imgExts = ["png", "jpg"]
    #     for path, dirs, files in os.walk(source):
    #         for fileName in tqdm(files):
    #             ext = fileName[-3:].lower()
    #             if ext not in imgExts:
    #                 continue
    #             filePath = os.path.join(source, fileName)
    #             im = Image.open(filePath)
    #             newIm = im.resize((224, 224))
    #             original_size = im.size
    #             newIm.save(os.path.join(target,fileName))
    
    # # Create csv in dlc format if not done so
    # if not os.path.exists(os.path.join(root_dir,run_dir_real,'labeled-data','RGB','CollectedData_chenqi.csv')):
    #     csv_source = os.path.join(args.real_dir,'keypoints.csv')    
    #     keypoints = pd.read_csv(csv_source, header=None)
    #     #csv_target = open(os.path.join(root_dir,run_dir_real,'labeled-data','CollectedData_chenqi.csv'),'w+')
    #     #writer = csv.writer(csv_target)
    #     #original_size = (1008, 760)
    #     keypoints.iloc[:,1:11:2] = keypoints.iloc[:,1:11:2]/original_size[0]*224
    #     keypoints.iloc[:,2:11:2] = keypoints.iloc[:,2:11:2]/original_size[1]*224
    #     keypoints.iloc[:,0] = 'labeled-data/RGB/'+keypoints.iloc[:,0].str[1:]
    #     my_header = pd.DataFrame([['scorer'] + ['chenqi']*10, ['bodyparts', 'kp0', 'kp0', 'kp1', 'kp1', 'kp2', 'kp2', 'kp3', 'kp3', 'kp4', 'kp4'], ['coords'] + ['x', 'y']*5])
    #     out_pd = pd.concat([my_header, keypoints])
    #     out_pd.to_csv(os.path.join(root_dir,run_dir_real,'labeled-data','RGB','CollectedData_chenqi.csv'),',',index=False, header=False)
    #     deeplabcut.convertcsv2h5(config_yaml_file, scorer= 'chenqi', userfeedback=False)
    
    # # Find number of shuffles in the training folder
    # num_shuffle = len(os.listdir(os.path.join(root_dir, run_dir, 'dlc-models', 'iteration-0')))

    # # Create training set
    # deeplabcut.create_training_dataset(config_yaml_file,num_shuffles=num_shuffle)

    # # Load pose_cfg.yaml
    # subdirs = os.listdir(os.path.join(root_dir, run_dir_real, 'dlc-models', 'iteration-0'))
    # for subdir in subdirs:
    #     pose_cfg_file = os.path.join(root_dir, run_dir_real, 'dlc-models', 'iteration-0', subdir, 'train', 'pose_cfg.yaml')

    #     f = open(os.path.join(root_dir, run_dir_real, 'dlc-models', 'iteration-0', subdir, 'train', 'checkpoint'))
    #     checkpoint = f.readlines()
    #     last_ckpt_path = os.path.join(checkpoint[0].split(': ')[1])

    #     with open(pose_cfg_file, 'rb') as f:
    #         pose_cfg = yaml.load(pose_cfg_file, Loader=yaml.SafeLoader)
    #     print(pose_cfg)
    #     pose_cfg['init_weights'] = last_ckpt_path
    #     with open(pose_cfg_file, 'w') as yamlfile:
    #         data = yaml.dump(pose_cfg, pose_cfg_file, sort_keys=False)

    # # Find dataset size from pkl file
    # dataset_folder = os.path.join(root_dir, run_dir_real, 'training-datasets', 'iteration-0')
    # for root, dirs, files in os.walk(dataset_folder):
    #     if not dirs:
    #         for file in files:
    #             if file[-6:] == 'pickle':
    #                 documentation_pickle_file = os.path.join(root, file)
    # with open(documentation_pickle_file, 'rb') as f:
    #     documentation_pickle = pickle.load(f)
    # train_size = documentation_pickle[-3].shape[0]
    # val_size = documentation_pickle[-2].shape[0]
    # split = documentation_pickle[-1]
    # print(train_size, val_size, split)

    # deeplabcut.evaluate_network(config_yaml_file, plotting=True, Shuffles=range(1,num_shuffle+1))

    # # Modify evaluation results to get the error for entire dataset
    # evaluation_result_file = os.path.join(root_dir, run_dir_real, 'evaluation-results', 'iteration-0', 'CombinedEvaluation-results.csv')
    # eval_res = pd.read_csv(evaluation_result_file)
    # average_error = eval_res[' Train error(px)']*eval_res['%Training dataset']/100 + eval_res[' Test error(px)']*(1-eval_res['%Training dataset']/100)
    # average_error = pd.DataFrame(average_error,columns=['average_error'])
    # average_error = average_error.rename(columns={"0":"Pet"})
    # eval_res_new = pd.concat([eval_res, average_error],axis=1)
    # eval_res_new.to_csv(evaluation_result_file[:-4]+'_mod.csv')

    # # Remove checkpoints to save space 
    # shutil.rmtree(os.path.join(root_dir,run_dir_real, 'dlc-models'))

    # # sim_version = args.sim_version
    # # run_num = args.run_num
    # # train_test_split = args.train_test_split
    # # num_epoch = args.num_epoch
    # # root_dir = 'D:\\Chenqi\\KP Detection\\DeepLabCut'
    # # run_dir = f'sim{str(sim_version).zfill(2)}_run{str(run_num).zfill(2)}_debug'
    # # deeplabcut.create_new_project(run_dir, 'chenqi', ['D:\\Chenqi\KP Detection\\dataset_video\\RealSampleDatasetFineCam\\20220711162306.avi'], working_directory=root_dir, copy_videos=True, multianimal=False)

    # # # Find the current directory for this run
    # # files = os.listdir(root_dir)
    # # for file in files:
    # #     if file[0:len(run_dir)] == run_dir:
    # #         run_dir = file

    # # # Load and edit the yaml file
    # # config_yaml_file = os.path.join(root_dir, run_dir, 'config.yaml')
    # # with open(config_yaml_file, 'rb') as f:
    # #     config_yaml = yaml.load(f,Loader=yaml.SafeLoader)
    # # config_yaml['bodyparts'] = ['kp0', 'kp1', 'kp2', 'kp3', 'kp4']
    # # config_yaml['video_sets'] = {f'D:\\Chenqi\\KP Detection\\DeepLabCut\\{run_dir}\\videos\\RGB.avi': {'crop': '0, 4032, 0, 3040'}}
    # # config_yaml['TrainingFraction'] = [train_test_split]
    # # config_yaml['snapshotindex'] = 'all'
    # # with open(config_yaml_file, 'w') as yamlfile:
    # #     data = yaml.dump(config_yaml, yamlfile, sort_keys=False)

    # # # Copy images into labeled_data
    # # source_root = os.path.join('D:\\Chenqi\\KP Detection\\dataset',f'sim{str(sim_version).zfill(2)}')
    # # for folder in os.listdir(source_root):
    # #     if folder[0:3] == 'RGB':
    # #         source = os.path.join(source_root, folder)
    # # target = os.path.join(root_dir,run_dir,'labeled-data','RGB')
    # # if not os.path.exists(target):
    # #     os.makedirs(target)
    # #     print('Copying and resizing dataset')
    # #     imgExts = ["png", "jpg"]
    # #     for path, dirs, files in os.walk(source):
    # #         for fileName in tqdm(files):
    # #             ext = fileName[-3:].lower()
    # #             if ext not in imgExts:
    # #                 continue
    # #             filePath = os.path.join(source, fileName)
    # #             im = Image.open(filePath)
    # #             newIm = im.resize((224, 224))
    # #             original_size = im.size
    # #             newIm.save(os.path.join(target,fileName))

    # # # Create csv in dlc format if not done so
    # # if not os.path.exists(os.path.join(root_dir,run_dir,'labeled-data','RGB','CollectedData_chenqi.csv')):
    # #     csv_source = os.path.join('D:\\Chenqi\\KP Detection\\dataset',f'sim{str(sim_version).zfill(2)}','keypoints.csv')    
    # #     keypoints = pd.read_csv(csv_source, header=None)
    # #     csv_target = open(os.path.join(root_dir,run_dir,'labeled-data','CollectedData_chenqi.csv'),'w+')
    # #     writer = csv.writer(csv_target)
    # #     #original_size = (1008, 760)
    # #     keypoints.iloc[:,1:11:2] = keypoints.iloc[:,1:11:2]/original_size[0]*224
    # #     keypoints.iloc[:,2:11:2] = keypoints.iloc[:,2:11:2]/original_size[1]*224
    # #     keypoints.iloc[:,0] = 'labeled-data/RGB/'+keypoints.iloc[:,0].str[1:]
    # #     my_header = pd.DataFrame([['scorer'] + ['chenqi']*10, ['bodyparts', 'kp0', 'kp0', 'kp1', 'kp1', 'kp2', 'kp2', 'kp3', 'kp3', 'kp4', 'kp4'], ['coords'] + ['x', 'y']*5])
    # #     out_pd = pd.concat([my_header, keypoints])
    # #     out_pd.to_csv(os.path.join(root_dir,run_dir,'labeled-data','RGB','CollectedData_chenqi.csv'),',',index=False, header=False)
    # #     deeplabcut.convertcsv2h5(config_yaml_file, scorer= 'chenqi', userfeedback=False)

    # # # Create training set
    # # deeplabcut.create_training_dataset(config_yaml_file)

    # # # Find dataset size from pkl file
    # # dataset_folder = os.path.join(root_dir, run_dir, 'training-datasets', 'iteration-0')
    # # for root, dirs, files in os.walk(dataset_folder):
    # #     if not dirs:
    # #         for file in files:
    # #             if file[-6:] == 'pickle':
    # #                 documentation_pickle_file = os.path.join(root, file)
    # # with open(documentation_pickle_file, 'rb') as f:
    # #     documentation_pickle = pickle.load(f)
    # # train_size = documentation_pickle[-3].shape[0]
    # # val_size = documentation_pickle[-2].shape[0]
    # # split = documentation_pickle[-1]
    # # print(train_size, val_size, split)

    # # deeplabcut.train_network(config_yaml_file,displayiters=int(train_size/10),saveiters=train_size,max_snapshots_to_keep=num_epoch, maxiters=train_size*num_epoch)
    # # deeplabcut.evaluate_network(config_yaml_file)
