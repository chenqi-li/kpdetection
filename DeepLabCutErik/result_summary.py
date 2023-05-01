import os
import pandas as pd
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, default='sim02_run01-chenqi-2022-07-22_real')
    args = parser.parse_args()

    root_dir = os.path.join('D:\Chenqi\KP Detection\DeepLabCut', args.dir)
    print(root_dir)
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
        
        if shuffle not in shuffle_result:
            shuffle_result[shuffle] = [error, iterations, epoch]

    total_error = 0
    for shuffle in sorted(shuffle_result):
        res = shuffle_result[shuffle]
        total_error += res[0]
        print(f'Shuffle{shuffle}: {round(res[0],3):.3f} @ e{str(res[2]).zfill(2)} i{str(res[1]).zfill(6)}')
    print(f"Lowest error: {str(round(sorted_eval_result.iloc[0][column_to_use], 3)).zfill(3)} from shuffle {int(sorted_eval_result.iloc[0]['Shuffle number'])}")
    print(f"Average error: {str(round(total_error/len(shuffle_result),3)).zfill(3)}")
    print('\n')
