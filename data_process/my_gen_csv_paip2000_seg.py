import os
from libs.dataPreprocess.my_data_seg import write_csv_based_on_dir
from libs.dataPreprocess.my_data import get_patient_ids_from_dir, \
    split_patient_ids, split_dataset_by_predefined_pat_id


data_type = 'paip2000_seg'
data_version = 'wsi_level0_384_384__num4000'

dir_images = f'/home/stu/data_share/PAIP2020/patches/segmentation/{data_version}/images'
dir_masks = f'/home/stu/data_share/PAIP2020/patches/segmentation/{data_version}/masks'

filename_csv = os.path.join(os.path.abspath('..'),
                            'datafiles', f'{data_type}_{data_version}.csv')

write_csv_based_on_dir(filename_csv, dir_images, dir_masks)
print('write csv file complete.')

list_patient_id = get_patient_ids_from_dir(dir_images)
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

split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_train, filename_train, field_columns=['images', 'masks'])
print(f'write training dataset file {filename_train} complete.')
split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_valid, filename_valid, field_columns=['images', 'masks'])
print(f'write validation dataset file {filename_valid} complete.')
split_dataset_by_predefined_pat_id(filename_csv, list_patient_id_test, filename_test, field_columns=['images', 'masks'])
print(f'write test dataset file {filename_test} complete.')


print('OK')