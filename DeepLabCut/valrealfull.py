from cProfile import run
from distutils import dep_util
from distutils.command.config import config
from email import header
from imp import source_from_cache
from pydoc import doc
from re import S
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
from distutils.dir_util import copy_tree
from deeplabcut.utils import auxiliaryfunctions, auxfun_multianimal
from pathlib import Path
from deeplabcut.pose_estimation_tensorflow.config import load_config
import numpy as np


def pairwisedistances(DataCombined, scorer1, scorer2, pcutoff=-1, bodyparts=None):
    """From DLC, Calculates the pairwise Euclidean distance metric over body parts vs. images"""
    mask = DataCombined[scorer2].xs("likelihood", level=1, axis=1) >= pcutoff
    if bodyparts == None:
        Pointwisesquareddistance = (DataCombined[scorer1] - DataCombined[scorer2]) ** 2
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        return RMSE, RMSE[mask]
    else:
        Pointwisesquareddistance = (
            DataCombined[scorer1][bodyparts] - DataCombined[scorer2][bodyparts]
        ) ** 2
        # print('gt', DataCombined[scorer1][bodyparts])
        # print('pd', DataCombined[scorer2][bodyparts])
        RMSE = np.sqrt(
            Pointwisesquareddistance.xs("x", level=1, axis=1)
            + Pointwisesquareddistance.xs("y", level=1, axis=1)
        )  # Euclidean distance (proportional to RMSE)
        # print(DataCombined[scorer1][bodyparts], DataCombined[scorer2][bodyparts])
        return RMSE, RMSE[mask]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load training parameters.')
    parser.add_argument('--num_epoch', default=1, type=int)
    parser.add_argument('--train_test_split', default=0.7, type=float)
    # parser.add_argument('--run_dir', default='sim01_run01-chenqi-2022-07-22', type=str)
    # Evaluation parameters
    parser.add_argument('--sim_version', default=0, type=int)
    parser.add_argument('--run_num', default=0, type=int)
    parser.add_argument('--real_dir', default='D:\Chenqi\KP Detection\dataset\RealSampleDatasetFineCam', type=str)
    parser.add_argument('--suffix', default='_real', type=str)

    args = parser.parse_args()

    for folder in os.listdir('D:\Chenqi\KP Detection\DeepLabCut'):
        if folder[0:11] == f'sim{str(args.sim_version).zfill(2)}_run{str(args.run_num).zfill(2)}':
            run_dir = folder
            break

    # Find the desired_shuffle_num and desired_iteration
    root_dir = os.path.join('D:\Chenqi\KP Detection\DeepLabCut', run_dir+args.suffix)
    shuffle_result = {}
    for path, dirs, files in os.walk(root_dir):
        for fileName in files:
            if fileName == 'CombinedEvaluation-results_mod.csv':
                file = os.path.join(root_dir, path, fileName)
    eval_result = pd.read_csv(file)
    dataset_size = int(eval_result['Training iterations:'][0])

    if eval_result.isnull().any()['average_error']:
        column_to_use = ' Train error(px)'
    else:
        column_to_use = 'average_error'

    sorted_eval_result = eval_result.sort_values(by=[column_to_use])

    for index, row in sorted_eval_result.iterrows():
        shuffle = int(row['Shuffle number'])
        error = float(row[column_to_use])
        iterations = int(row['Training iterations:'])
        epoch = int(iterations/dataset_size)
        # Use the first 4 shuffles only
        if shuffle > 4:
            continue
        if shuffle not in shuffle_result:
            shuffle_result[shuffle] = [error, iterations, epoch]

    total_error = 0
    desired_pair = []
    print(run_dir+args.suffix)
    for shuffle in sorted(shuffle_result):
        res = shuffle_result[shuffle]
        total_error += res[0]
        print(f'Shuffle{shuffle}: {round(res[0],3):.3f} @ e{str(res[2]).zfill(2)} i{str(res[1]).zfill(6)}')
        desired_pair.append((int(shuffle),int(res[1])))
    # desired_iteration_num = int(sorted_eval_result.iloc[0]['Training iterations:'])
    # desired_shuffle_num = int(sorted_eval_result.iloc[0]['Shuffle number'])
    print(f"Lowest error: {str(round(sorted_eval_result.iloc[0][column_to_use], 3)).zfill(3)} from shuffle {int(sorted_eval_result.iloc[0]['Shuffle number'])}")
    print(f"Average error: {str(round(total_error/len(shuffle_result),3)).zfill(3)}")



    # Get right directories
    root_dir = 'D:\\Chenqi\\KP Detection\\DeepLabCut'
    run_dir = os.path.join(root_dir,run_dir)
    run_dir_real = run_dir+args.suffix
    
    # Read yaml config file
    config_yaml_file = os.path.join(root_dir, run_dir, 'config.yaml')
    cfg = auxiliaryfunctions.read_config(config_yaml_file)

    # Get comparison body parts
    comparisonbodyparts = auxiliaryfunctions.IntersectionofBodyPartsandOnesGivenbyUser(
        cfg, 'all')
    
    # Get trainIndices
    trainingsetfolder = auxiliaryfunctions.GetTrainingSetFolder(cfg)
    trainFraction = cfg["TrainingFraction"][0]
    datafn, metadatafn = auxiliaryfunctions.GetDataandMetaDataFilenames(
        trainingsetfolder, trainFraction, shuffle, cfg
    )
    data, trainIndices, testIndices, trainFraction = auxiliaryfunctions.LoadMetadata(
        os.path.join(cfg["project_path"], metadatafn)
    )

    # Find number of shuffles in the training folder
    num_shuffle = len(os.listdir(os.path.join(root_dir, run_dir, 'dlc-models', 'iteration-0')))

    # Get scorer name
    shuffle = 1 #range(1,num_shuffle+1)
    trainingsetindex = 0
    if trainingsetindex == "all":
            TrainingFractions = cfg["TrainingFraction"]
    else:
        if (
            trainingsetindex < len(cfg["TrainingFraction"])
            and trainingsetindex >= 0
        ):
            TrainingFractions = [cfg["TrainingFraction"][int(trainingsetindex)]]
        else:
            raise Exception(
                "Please check the trainingsetindex! ",
                trainingsetindex,
                " should be an integer from 0 .. ",
                int(len(cfg["TrainingFraction"]) - 1),
            )

    # Evaluate
    every_error = []
    for trainFraction in TrainingFractions:
        modelprefix = ""
        modelfolder = os.path.join(cfg["project_path"],
            str(
                auxiliaryfunctions.get_model_folder(
                    trainFraction, shuffle, cfg, modelprefix=modelprefix
                )
            ),
        )
        # print(modelfolder)

        
        path_test_config = Path(modelfolder) / "test" / "pose_cfg.yaml"
        dlc_cfg = load_config(str(path_test_config))
        trainingsiterations = (dlc_cfg["init_weights"].split(os.sep)[-1]).split("-")[-1]
        DLCscorer, DLCscorerlegacy = auxiliaryfunctions.GetScorerName(
                cfg, shuffle, trainFraction, trainingsiterations, modelprefix=modelprefix)
        # GT Annotation
        for root, dirs, files in os.walk(os.path.join(run_dir_real,'training-datasets','iteration-0')):
            if not dirs:
                for file in files:
                    if file[-2:] == 'h5':
                        gt_anno_file = os.path.join(root, file)
        Data = pd.read_hdf(gt_anno_file)

        # PRED Annotation
        pred_anno_files = []
        for root, dirs, files in os.walk(os.path.join(run_dir_real,'evaluation-results','iteration-0')):
            for file in files:
                if file[-2:] == 'h5':
                    pred_anno_files.append(os.path.join(root,file))
        for pred_anno_file in pred_anno_files:
            # Find iteration and shuffle of best checkpoint
            shuffle_num = int(pred_anno_file.split('shuffle')[-1].split('_')[0])
            iteration_num = int(pred_anno_file[:-3].split('-')[-1])
            if (shuffle_num,iteration_num) not in desired_pair:
                continue
            
            # Read the individual errors
            DataMachine = pd.read_hdf(os.path.join(pred_anno_file))
            # print(DataMachine)
            DataCombined = pd.concat([Data.T, DataMachine.T], axis=0).T
            RMSE, RMSE_masked = pairwisedistances(
                DataCombined,
                cfg["scorer"],
                pred_anno_file.split('-snapshot')[0].split('\\')[-1], #DLCscorer[:-18]+'_'+str(pred_anno_file.split('-')[-1][:-3]),
                cfg["pcutoff"],
                comparisonbodyparts)
            every_error.append(RMSE.iloc[:].values.flatten())
    print('Mean, Standard Deviation: ',np.nanmean(every_error),np.nanstd(every_error))
    print('Box Whisker Quantiles: ', np.nanquantile(every_error, [0.00, 0.25, 0.50, 0.75, 1.00]))
    print('\n')
    #averagerror = np.nanmean(RMSE.iloc[:].values.flatten())
            
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

    # # Modify evaluation results to get the error for entire dataset
    # evaluation_result_file = os.path.join(root_dir, run_dir_real, 'evaluation-results', 'iteration-0', 'CombinedEvaluation-results.csv')
    # eval_res = pd.read_csv(evaluation_result_file)
    # average_error = eval_res[' Train error(px)']*eval_res['%Training dataset']/100 + eval_res[' Test error(px)']*(1-eval_res['%Training dataset']/100)
    # average_error = pd.DataFrame(average_error,columns=['average_error'])
    # average_error = average_error.rename(columns={"0":"Pet"})
    # eval_res_new = pd.concat([eval_res, average_error],axis=1)
    # eval_res_new.to_csv(evaluation_result_file[:-4]+'_mod.csv')