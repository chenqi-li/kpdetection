from distutils import dep_util
import deeplabcut
import os

video_path = 'D:\Chenqi\KP Detection\dataset_video\RealSampleDatasetFineCam'
video_files = [os.path.join(video_path, file) for file in os.listdir(video_path)]

deeplabcut.create_new_project('real_00', 'chenqi', video_files, working_directory='D:\Chenqi\KP Detection\DeepLabCut', copy_videos=True, multianimal=False)

config_path = "D:\\Chenqi\\KP Detection\\DeepLabCut\\real_00-chenqi-2022-07-16\\config.yaml"


#deeplabcut.extract_frames(config_path)

#deeplabcut.label_frames(config_path)

#deeplabcut.check_labels(config_path)

deeplabcut.create_training_dataset(config_path, num_shuffles=4)

#deeplabcut.train_network(config_path,displayiters=1,saveiters=100)

#deeplabcut.evaluate_network(config_path,Shuffles=[1], plotting=True)
