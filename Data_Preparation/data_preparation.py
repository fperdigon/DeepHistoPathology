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


def prepare_IDC_Data(data_file, data_split=None, extract_anyways=False):
    url = 'http://andrewjanowczyk.com/wp-static/IDC_regular_ps50_idx5.zip'
    name = 'IDC_regular_ps50_idx5.zip'
    uzip_folder = 'IDC_Dataset'

    # Download the data
    if not os.path.exists(data_file + '/' + name):
        print('Downloading data ...')
        urllib.request.urlretrieve(url, data_file + '/' + name)
        print('Data downloaded ...')
    else:
        print('Data downloaded already ...')

    # UnZip
    if (not os.path.exists(data_file + '/' + uzip_folder)) and (not extract_anyways):
        print('Extracting data ...')
        with zipfile.ZipFile(data_file + '/' + name, 'r') as zip_ref:
            zip_ref.extractall(data_file + '/' + uzip_folder)
        print('Data extracted ...')
    else:
        print('Data extracted already ...')


    # Read Data split from files
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



    print('DB')


if __name__ == "__main__":
    data_file = '../data'
    train_list_file = './cases_train.txt'
    val_list_file = './cases_val.txt'
    test_list_file = './cases_test.txt'
    data_split = [train_list_file, val_list_file, test_list_file]

    prepare_IDC_Data(data_file, data_split=data_split)
