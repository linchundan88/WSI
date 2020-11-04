import os
import random
import pandas as pd
import csv
import sklearn

def write_csv_based_on_dir(filename_csv, base_dir, dict_mapping, match_type='header',
       list_file_ext=['.BMP', '.PNG', '.JPG', '.JPEG', '.TIFF', '.TIF']):

    assert match_type in ['header', 'partial', 'end'], 'match type is error'

    if os.path.exists(filename_csv):
        os.remove(filename_csv)
    os.makedirs(os.path.dirname(filename_csv), exist_ok=True)

    if not base_dir.endswith('/'):
        base_dir = base_dir + '/'

    with open(filename_csv, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow(['images', 'labels'])

        for dir_path, subpaths, files in os.walk(base_dir, False):
            for f in files:
                img_file_source = os.path.join(dir_path, f)
                (filedir, tempfilename) = os.path.split(img_file_source)
                (filename, extension) = os.path.splitext(tempfilename)
                if extension.upper() not in list_file_ext:
                    print('file ext name:', f)
                    continue

                if not filedir.endswith('/'):
                    filedir += '/'

                for (k, v) in dict_mapping.items():
                    if match_type == 'header':
                        dir1 = os.path.join(base_dir, k)
                        if not dir1.endswith('/'):
                            dir1 += '/'

                        if dir1 in filedir:
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'partial':
                        if '/' + k + '/' in filedir:
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break
                    elif match_type == 'end':
                        if filedir.endswith('/' + k + '/'):
                            print(f'writing record:{img_file_source}')
                            csv_writer.writerow([img_file_source, v])
                            break


def split_dataset(filename_csv, valid_ratio=0.1, test_ratio=None,
                  shuffle=True, random_state=None, field_columns=['images', 'labels']):

    if filename_csv.endswith('.csv'):
        df = pd.read_csv(filename_csv)
    elif filename_csv.endswith('.xls') or filename_csv.endswith('.xlsx'):
        df = pd.read_excel(filename_csv)

    if shuffle:
        df = sklearn.utils.shuffle(df, random_state=random_state)

    if test_ratio is None:
        split_num = int(len(df)*(1-valid_ratio))
        data_train = df[:split_num]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        data_valid = df[split_num:]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels
    else:
        split_num_train = int(len(df) * (1 - valid_ratio - test_ratio))
        data_train = df[:split_num_train]
        train_files = data_train[field_columns[0]].tolist()
        train_labels = data_train[field_columns[1]].tolist()

        split_num_valid = int(len(df) * (1 - test_ratio))
        data_valid = df[split_num_train:split_num_valid]
        valid_files = data_valid[field_columns[0]].tolist()
        valid_labels = data_valid[field_columns[1]].tolist()

        data_test = df[split_num_valid:]
        test_files = data_test[field_columns[0]].tolist()
        test_labels = data_test[field_columns[1]].tolist()

        return train_files, train_labels, valid_files, valid_labels, test_files, test_labels


def get_patient_ids_from_dir(dir):
    list_patient_id = []

    for dir_path, subpaths, files in os.walk(dir, False):
        patient_id = dir_path.split('/')[-1]
        if patient_id not in list_patient_id:
            list_patient_id.append(patient_id)

    return list_patient_id


def split_patient_ids(list_patient_id, valid_ratio=0.1, test_ratio=0.1,
                      random_seed=18888):
    random.seed(random_seed)
    random.shuffle(list_patient_id)

    if test_ratio is None:
        split_num = int(len(list_patient_id) * (1 - valid_ratio))
        list_patient_id_train = list_patient_id[:split_num]
        list_patient_id_valid = list_patient_id[split_num:]

        return list_patient_id_train, list_patient_id_valid
    else:
        split_num_train = int(len(list_patient_id) * (1 - valid_ratio - test_ratio))
        list_patient_id_train = list_patient_id[:split_num_train]
        split_num_valid = int(len(list_patient_id) * (1 - test_ratio))
        list_patient_id_valid = list_patient_id[split_num_train:split_num_valid]
        list_patient_id_test = list_patient_id[split_num_valid:]

        return list_patient_id_train, list_patient_id_valid, list_patient_id_test

def file_belong_patient_ids(file, list_patient_ids):
    if os.path.isfile(file):
        file_dir, filename = os.path.split(file)
    else:
        file_dir = file

    tmp_s = file_dir.split('/')[-1]

    for patient_id in list_patient_ids:
        if tmp_s == patient_id:
            return True

    return False


def split_dataset_by_predefined_pat_id(filename_csv, list_patient_ids, filename_csv_save, field_columns=['images', 'labels']):

    assert filename_csv.endswith('.csv'), f"{filename_csv} file error."
    if filename_csv.endswith('.csv'):
        df = pd.read_csv(filename_csv)
    # elif filename_csv.endswith('.xls') or filename_csv.endswith('.xlsx'):
    #     df = pd.read_excel(filename_csv)

    files, labels = [], []
    with open(filename_csv_save, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile, delimiter=',')
        csv_writer.writerow([field_columns[0], field_columns[1]])

        for _, row in df.iterrows():
            file = row[field_columns[0]]
            label = row[field_columns[1]]

            if file_belong_patient_ids(file, list_patient_ids):
                print(f'writing {file}')
                csv_writer.writerow([file, label])

    return files, labels

