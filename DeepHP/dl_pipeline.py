#============================================================
#
#  Deep HistoPathology (DeepHP)
#  DL Pipeline
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import os
import numpy as np
import datetime

import DeepHP.dl_models as dl_models
from imblearn.over_sampling import RandomOverSampler

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping, TensorBoard

from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from keras.utils import to_categorical

batch_counter = 0
batch_step = 100

experiment_label = 'RGB_norm_paper_model'


def balace_data(X, y):
    """
    Balance the dataset using Oversampling technique
    :param X: Feature vector
    :param y: Labels vector
    :return: Balanced dataset
    """
    sampling_strategy = "not majority"
    ros = RandomOverSampler(sampling_strategy=sampling_strategy, random_state=1234)
    X = np.expand_dims(np.array(X), 1)
    y = np.array(y)
    X_res, y_res = ros.fit_resample(X, y)
    X_res = np.squeeze(X_res, 1)
    return [X_res, y_res]


def normalize_rgb(img, std=True):
    """
    Normalization for RGB images
    :param img: input numpy image
    :param std: std normalization
    :return: Normalized numpy image
    """
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]

    R = R - np.mean(R)  # zero-center
    G = G - np.mean(G)  # zero-center
    B = B - np.mean(B)  # zero-center
    if std:
        epsilon = 1e-100
        R_std = np.std(R)  # normalize
        R = R / (R_std + epsilon)  # epsilon is used in order to avoid by zero division
        G_std = np.std(G)  # normalize
        G = G / (G_std + epsilon)  # epsilon is used in order to avoid by zero division
        B_std = np.std(B)  # normalize
        B = B / (B_std + epsilon)  # epsilon is used in order to avoid by zero division

        img = np.stack((R, G, B), axis=-1)
        img = img.astype(dtype=np.float16)
    return img


def load_img_from_path(img_path):
    """
    Load an image, fix the issue with images smaller than 50x50
    :param img_path: image path
    :return: image in numpy format
    """
    img_patch = np.zeros((50, 50, 3), dtype=np.uint8)
    img = load_img(img_path, grayscale=False)  # Import Images in keras way
    img = img_to_array(img)
    img = img.astype(dtype=np.uint8)
    im_dim0 = img.shape[0]
    im_dim1 = img.shape[1]
    img_patch[0:im_dim0, 0:im_dim1, :] = img
    # img_patch = normalize_rgb(img_patch)
    return img_patch


def dataset_np(Dataset_paths):
    """
    Generates Dataset in np format
    :param Dataset_paths: datasets paths for each set
    :return: Dataset in numpy format
    """
    [X_train_list, y_train_list, X_val_list, y_val_list, X_test_list, y_test_list] = Dataset_paths

    [X_train_list, y_train_list] = balace_data(X_train_list, y_train_list)

    [X_val_list, y_val_list] = balace_data(X_val_list, y_val_list)

    X_train = []
    for img_path in X_train_list:
        img_tmp = load_img_from_path(img_path)
        X_train.append(img_tmp)

    X_train = np.array(X_train)
    y_train = to_categorical(np.array(y_train_list))

    X_val = []
    for img_path in X_val_list:
        img_tmp = load_img_from_path(img_path)
        X_val.append(img_tmp)

    X_val = np.array(X_val)
    y_val = to_categorical(np.array(y_val_list))

    X_test = []
    for img_path in X_test_list:
        img_tmp = load_img_from_path(img_path)
        X_test.append(img_tmp)

    X_test = np.array(X_test)
    y_test = to_categorical(np.array(y_test_list))

    return [X_train, y_train, X_val, y_val, X_test, y_test]


def train_dl(Dataset, DA=False):
    [train_set, train_set_GT, val_set, val_set_GT, test_set, test_set_GT] = Dataset

    # Results with DA are always lower
    # ====================
    # DATA AUGMENTATION
    # ====================

    if DA:
        print('Data Augmentation: In progress')
        data_gen_args = dict(featurewise_center=False,
                             featurewise_std_normalization=False,
                             rotation_range=20.,
                             width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.0,
                             horizontal_flip=True,
                             vertical_flip=True,
                             fill_mode="nearest")

        datagen = ImageDataGenerator(**data_gen_args)

        seed = 1

        datagen.fit(train_set, augment=False, seed=seed)
        print('Data Augmentation: Done')

    # ==================
    # LOAD THE DL MODEL
    # ==================
    # model = dl_models.deepHP_model_BC_V0_DW_V0()
    # model = dl_models.deepHP_model_BC_V0_DW_V1()
    # model = dl_models.deepHP_model_BC_V0_inception()
    model = dl_models.deepHP_model_BC_V0_1_inception()  # BEST RESULTS
    # model = dl_models.deepHP_model_BC_V0_inception_non_bn()
    # model = dl_models.deepHP_model_BC_V0_inception_NO_FC()
    # model = dl_models.deepHP_model_BC_V1_inception()
    # model = dl_models.deepHP_model_BC_v1_DP_inception()

    model.summary()

    epochs = int(1e5)  # 100000
    batch_size = 32
    # best results Adam ...
    lr = 1e-3
    # lr = 1e-4
    minimum_lr = 1e-10

    model.compile(loss=keras.losses.categorical_crossentropy,
                  # optimizer=keras.optimizers.Adadelta(),
                  optimizer=keras.optimizers.Adam(lr=lr),  # Best results
                  # optimizer=keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False),
                  metrics=['accuracy'])

    # checkpoint
    model_filepath = "weights.best.hdf5"
    checkpoint = ModelCheckpoint(model_filepath,
                                 monitor='val_acc',
                                 verbose=1,
                                 save_best_only=True,
                                 mode='max',  # on acc has to go max
                                 save_weights_only=True)

    reduce_lr = ReduceLROnPlateau(monitor='val_acc',
                                  factor=0.5,
                                  min_delta=0.005,
                                  mode='max',  # on acc has to go max
                                  patience=3,
                                  min_lr=minimum_lr,
                                  verbose=1)

    early_stop = EarlyStopping(monitor="val_acc",  # "val_loss"
                               min_delta=0.005,
                               mode='max',  # on acc has to go max
                               patience=10,
                               verbose=1)

    global experiment_label
    tb_log_dir = './TBoard_Graph/' + experiment_label

    if os.path.isdir(tb_log_dir):
        tb_log_dir = './TBoard_Graph/' + str(datetime.datetime.now())

    tboard = TensorBoard(log_dir=tb_log_dir, histogram_freq=0,
                         write_graph=False, write_grads=False,
                         write_images=False, embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None)

    # Command to run tensorboard
    # tensorboard --logdir=./TBoard_Graph

    if DA:
        # Results with DA are always lower
        # fits the model on batches with real-time data augmentation:
        model.fit_generator(datagen.flow(train_set, train_set_GT, batch_size=batch_size),
                            steps_per_epoch=int(len(train_set) / batch_size), epochs=epochs,
                            validation_data=(val_set, val_set_GT),
                            verbose=1,
                            max_queue_size=10, workers=32, use_multiprocessing=True,
                            callbacks=[early_stop, reduce_lr, checkpoint, tboard])
    else:
        model.fit(x=train_set, y=train_set_GT,
                  validation_data=(val_set, val_set_GT),
                  batch_size=batch_size, epochs=epochs,
                  verbose=1,
                  callbacks=[early_stop, reduce_lr, checkpoint, tboard])

    K.clear_session()


def test_dl(Dataset):
    [train_set, train_set_GT, val_set, val_set_GT, test_set, test_set_GT] = Dataset

    batch_size = 32

    # ==================
    # LOAD THE DL MODEL
    # ==================
    # model = dl_models.deepHP_model_BC_V0_DW_V0()
    # model = dl_models.deepHP_model_BC_V0_DW_V1()
    # model = dl_models.deepHP_model_BC_V0_inception()
    model = dl_models.deepHP_model_BC_V0_1_inception()  # BEST RESULTS
    # model = dl_models.deepHP_model_BC_V0_inception_non_bn()
    # model = dl_models.deepHP_model_BC_V0_inception_NO_FC()
    # model = dl_models.deepHP_model_BC_V1_inception()
    # model = dl_models.deepHP_model_BC_v1_DP_inception()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.1),
                  metrics=['accuracy'])

    # checkpoint
    model_filepath = "weights.best.hdf5"
    # load weights
    model.load_weights(model_filepath)

    test_pred_keras = model.predict(test_set, batch_size=batch_size, verbose=1)

    cc = test_pred_keras[:, 1]
    ncc = test_pred_keras[:, 0]

    print('DB')

    global experiment_label
    # Pickle the values
    import _pickle as pickle
    with open('results_' + experiment_label + '_.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([test_set_GT, test_pred_keras], output)

    with open('results.pkl', 'wb') as output:  # Overwrites any existing file.
        pickle.dump([test_set_GT, test_pred_keras], output)

    K.clear_session()


def inference_dl(inference_set, model_filepath="weights.best.hdf5"):

    batch_size = 8

    # ==================
    # LOAD THE DL MODEL
    # ==================
    # model = dl_models.deepHP_model_BC_V0_DW_V0()
    # model = dl_models.deepHP_model_BC_V0_DW_V1()
    # model = dl_models.deepHP_model_BC_V0_inception()
    model = dl_models.deepHP_model_BC_V0_1_inception()  # BEST RESULTS
    # model = dl_models.deepHP_model_BC_V0_inception_non_bn()
    # model = dl_models.deepHP_model_BC_V0_inception_NO_FC()
    # model = dl_models.deepHP_model_BC_V1_inception()
    # model = dl_models.deepHP_model_BC_v1_DP_inception()

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=0.1),
                  metrics=['accuracy'])

    # checkpoint

    # load weights
    model.load_weights(model_filepath)

    pred_keras = model.predict(inference_set, batch_size=batch_size, verbose=1)

    K.clear_session()

    return pred_keras