import os
import pandas as pd
import pickle
from libs.dataPreprocess.my_data import write_csv_based_on_dir, get_patient_ids_from_dir, \
    split_patient_ids, split_dataset_by_predefined_pat_id


data_type = 'camelyon16'
data_version = 'v1'
dir_path = '/home/stu/data_share/CAMELYON16/training/patches/classification'

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

# filename_pkl_train = os.path.join(os.path.abspath('.'),
#                     'pat_id_pkl', 'split_patid_train.pkl')
# with open(filename_pkl_train, 'wb') as file:
#     pickle.dump(list_patient_id_train, file)

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



print('OK')