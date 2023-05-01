import os
import pickle

# filename = 'D:\\Chenqi\\KP Detection\\DeepLabCut\\real_00-chenqi-2022-07-16\\training-datasets\\iteration-0\\UnaugmentedDataSet_real_00Jul16\\Documentation_data-real_00_95shuffle1.pickle'

# with open(filename, 'rb') as f:
#     shuffle1 = pickle.load(f)


# print(shuffle1)


# filename = 'D:\\Chenqi\\KP Detection\\DeepLabCut\\real_00-chenqi-2022-07-16\\training-datasets\\iteration-0\\UnaugmentedDataSet_real_00Jul16\\Documentation_data-real_00_95shuffle2.pickle'

# with open(filename, 'rb') as f:
#     shuffle1 = pickle.load(f)


# print(shuffle1)


# from scipy.io import loadmat
# matfile = 'D:\\Chenqi\\KP Detection\\DeepLabCut\\real_00-chenqi-2022-07-16\\training-datasets\\iteration-0\\UnaugmentedDataSet_real_00Jul16\\real_00_chenqi95shuffle1.mat'
# annots = loadmat(matfile)
# print(annots)








filename = 'D:\\Chenqi\\KP Detection\\DeepLabCut\\sim02_run00-chenqi-2022-07-16\\training-datasets\\iteration-0\\UnaugmentedDataSet_sim02_run00Jul16\\Documentation_data-sim02_run00_70shuffle1.pickle'

with open(filename, 'rb') as f:
    shuffle1 = pickle.load(f)


print(shuffle1)


filename = 'D:\\Chenqi\\KP Detection\\DeepLabCut\\sim02_run01-chenqi-2022-07-22\\training-datasets\\iteration-0\\UnaugmentedDataSet_sim02_run01Jul22\\Documentation_data-sim02_run01_70shuffle2.pickle'

with open(filename, 'rb') as f:
    shuffle1 = pickle.load(f)


print(shuffle1)