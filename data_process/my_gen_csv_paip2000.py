import os
import pandas as pd
import pickle
from libs.dataPreprocess.my_data import write_csv_based_on_dir, get_patient_ids_from_dir, \
    split_patient_ids, split_dataset_by_predefined_pat_id


data_type = 'paip2000'
data_version = 'level1_299_299'
dir_path = '/home/stu/data_share/PAIP2020/patches/classification/wsi_level1_299_299'

filename_csv = os.path.join(os.path.abspath('..'),
               'datafiles', f'{data_type}_{data_version}.csv')
dict_mapping = {'0': 0, '1': 1}
write_csv_based_on_dir(filename_csv, dir_path, dict_mapping, match_type='header')
print('write csv file complete.')

list_patient_id = get_patient_ids_from_dir(dir_path)
print('loading patient ids from dir complete.')
list_patient_id_train, list_patient_id_valid, list_patient_id_test\
    = split_patient_ids(list_patient_id, valid_ratio=0.1, test_ratio=0.1, random_seed=1234)
print('split patient ids complete.')


filename_train = os.path.join(os.path.abspath('..'),
                'datafiles', f'{data_type}_{data_version}_train.csv')
filename_valid = os.path.join(os.path.abspath('..'),
                'datafiles', f'{data_type}_{data_version}_valid.csv')
filename_test = os.path.join(os.path.abspath('..'),
                'datafiles', f'{data_type}_{data_version}_test.csv')

split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_train, filename_train)
print(f'write training dataset file {filename_train} complete.')
split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_valid, filename_valid)
print(f'write validation dataset file {filename_valid} complete.')
split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_test, filename_test)
print(f'write test dataset file {filename_test} complete.')


for file_csv in [filename_train, filename_valid, filename_test]:
    df = pd.read_csv(file_csv)
    print(len(df))
    num_class = df['labels'].nunique(dropna=True)
    for label in range(num_class):
        df1 = df[df['labels'] == label]
        print(str(label), len(df1))


'''

paip2000_test_level0_299_299
2175606
0 1426721
1 748885
316839
0 215435
1 101404
207583
0 135830
1 71753

paip2000_test_level1_299_299
140158
0 92414
1 47744
20303
0 13872
1 6431
13688
0 9040
1 4648

'''


print('OK')