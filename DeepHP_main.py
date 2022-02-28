# -*- coding: utf-8 -*-

#============================================================
#
#  Deep HistoPathology (DeepHP)
#  Main
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import _pickle as pickle

import Data_Preparation.data_preparation as data_preparation
import DeepHP.dl_pipeline as dl_pipeline
import Utils.metrics as metrics
import Utils.utils as utils

if __name__ == "__main__":
    data_file = './data'
    train_list_file = './Data_Preparation/cases_train.txt'
    val_list_file = './Data_Preparation/cases_val.txt'
    test_list_file = './Data_Preparation/cases_test.txt'
    data_split = [train_list_file, val_list_file, test_list_file]

    Dataset_paths = data_preparation.prepare_IDC_Data(data_file, data_split=data_split)
    Dataset_np = dl_pipeline.dataset_np(Dataset_paths)

    dl_pipeline.train_dl(Dataset_np)
    dl_pipeline.test_dl(Dataset_np)

    # Load results
    with open('results.pkl', 'rb') as input:
        [test_set_GT, test_pred_keras] = pickle.load(input)

    metrics.classification_metrics(test_set_GT, test_pred_keras)

    metrics.generate_roc_plus_auc(test_set_GT, test_pred_keras)
    metrics.confusion_matrix_plot(test_set_GT, test_pred_keras, classes=['Non-IDC', 'IDC'])


