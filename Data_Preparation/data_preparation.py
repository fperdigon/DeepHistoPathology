# ============================================================
#
#  Deep Histo Pathology
#  Data preparation
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
# ===========================================================

import urllib.request
import os
import zipfile

def load_img_path(data_path, folders_list):
    X_data_list = []
    y_data_list = []

    for folder in folders_list:
        # Getting negative patch
        if os.path.exists(data_path + '/' + folder):
            for file in os.listdir(data_path + '/' + folder + '/0'):
                tmp_img_path = data_path + '/' + folder + '/0/' + file
                if os.path.isfile(tmp_img_path):
                    X_data_list.append(tmp_img_path)
                    y_data_list.append(0)

            # Getting positive patch
            for file in os.listdir(data_path + '/' + folder + '/1'):
                tmp_img_path = data_path + '/' + folder + '/1/' + file
                if os.path.isfile(tmp_img_path):
                    X_data_list.append(tmp_img_path)
                    y_data_list.append(1)
        else:
            print(folder + ': folder do not exist')

    return [X_data_list, y_data_list]


def prepare_IDC_Data(data_folder, data_split=None, extract_anyways=False):
    url = 'http://andrewjanowczyk.com/wp-static/IDC_regular_ps50_idx5.zip'
    name = 'IDC_regular_ps50_idx5.zip'
    uzip_folder = 'IDC_Dataset'

    # Download the data
    if not os.path.exists(data_folder + '/' + name):
        print('Downloading data ...')
        urllib.request.urlretrieve(url, data_folder + '/' + name)
        print('Data downloaded ...')
    else:
        print('Data downloaded already ...')

    # UnZip
    if (not os.path.exists(data_folder + '/' + uzip_folder)) and (not extract_anyways):
        print('Extracting data ...')
        with zipfile.ZipFile(data_folder + '/' + name, 'r') as zip_ref:
            zip_ref.extractall(data_folder + '/' + uzip_folder)
        print('Data extracted ...')
    else:
        print('Data extracted already ...')


    # Read Data split from files
    # Same split used by Roa et all https://github.com/choosehappy/public/tree/master/DL%20tutorial%20Code/6-idc
    if data_split is None:
        train_list_file = './Data_Preparation/cases_train.txt'
        val_list_file = './Data_Preparation/cases_val.txt'
        test_list_file = './Data_Preparation/cases_test.txt'
    else:
        [train_list_file, val_list_file, test_list_file] = data_split

    train_list = []
    with open(train_list_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            train_list.append(line.strip())

    val_list = []
    with open(val_list_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            val_list.append(line.strip())

    test_list = []
    with open(test_list_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            test_list.append(line.strip())

    # Path list and labels

    [X_train_list, y_train_list] = load_img_path(data_folder + '/' + uzip_folder, train_list)

    [X_val_list, y_val_list] = load_img_path(data_folder + '/' + uzip_folder, val_list)

    [X_test_list, y_test_list] = load_img_path(data_folder + '/' + uzip_folder, test_list)

    return [X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list]


if __name__ == "__main__":
    data_file = '../data'
    train_list_file = './cases_train.txt'
    val_list_file = './cases_val.txt'
    test_list_file = './cases_test.txt'
    data_split = [train_list_file, val_list_file, test_list_file]

    prepare_IDC_Data(data_file, data_split=data_split)
