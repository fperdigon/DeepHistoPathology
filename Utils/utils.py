#============================================================
#
#  Deep HistoPathology (DeepHP)
#  Utility functions
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import os
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes


def hps_image_reconst_from_patches(dir, DataSet_path):

        ### load all patches path ###

        negative_patches_file_list = []
        positive_patches_file_list = []
        for file in os.listdir(DataSet_path + dir + '/0'):
            tmp_img_path = DataSet_path + dir + '/0/' + file
            if os.path.isfile(tmp_img_path):
                negative_patches_file_list.append(tmp_img_path)

        for file in os.listdir(DataSet_path + dir + '/1'):
            tmp_img_path = DataSet_path + dir + '/1/' + file
            if os.path.isfile(tmp_img_path):
                positive_patches_file_list.append(tmp_img_path)

        patches = negative_patches_file_list + positive_patches_file_list

        ### find the maximum size ###

        x_size = 0
        y_size = 0

        patch_size = (50, 50)

        for p in patches:
            x_val = p.split('/')[-1].split('_')[2].split('x')[1]
            x_val = int(x_val)

            y_val = p.split('/')[-1].split('_')[3].split('y')[1]
            y_val = int(y_val)

            if x_size < x_val:
                x_size = x_val
            if y_size < y_val:
                y_size = y_val

        x_size = x_size + 50
        y_size = y_size + 50

        img_big = np.zeros((y_size, x_size, 3), dtype=np.uint8)
        img_big = img_big + 240

        class_patches = np.zeros((y_size, x_size, 3), dtype=np.uint8)

        for p in patches:
            x_val = p.split('/')[-1].split('_')[2].split('x')[1]
            x_val = int(x_val)

            y_val = p.split('/')[-1].split('_')[3].split('y')[1]
            y_val = int(y_val)

            img = load_img(p, grayscale=False)  # Import Images in keras way
            img = img_to_array(img)  # Import Images in keras way
            img = img.astype(dtype=np.uint8)
            # plt.imshow(img)
            # plt.show()
            im_dim0 = img.shape[1]
            im_dim1 = img.shape[0]

            dx = x_val + im_dim0
            dy = y_val + im_dim1

            img_big[y_val:dy, x_val:dx, :] = img

            if p in positive_patches_file_list:
                class_patches[y_val:dy, x_val:dx, :] = 255

        return [img_big, class_patches]


def save_reconst_hps_to_png(img, save_folder, hps_slide):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # Plot the reconstructed image
    plt.figure()
    plt.imshow(img)
    plt.axis('off')
    plt.tight_layout()
    fig1 = plt.gcf()
    plt.show(block=False)
    fig1.savefig(save_folder + '/' + hps_slide + '_hps.png', dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    return save_folder + '/' + hps_slide + '_hps.png'


def load_img_generate_patch_array(imp_path, patch_size=(50, 50)):

    im_data = load_img(imp_path, grayscale=False)  # Import Images in keras way
    im_data = img_to_array(im_data)  # Import Images in keras way
    im_data = im_data.astype(dtype=np.uint8)

    patch_per_row = int(im_data.shape[0] / patch_size[0])
    patch_per_column = int(im_data.shape[1] / patch_size[1])

    patches_array = []
    for i in range(patch_per_row):
        for j in range(patch_per_column):
            x = i * patch_size[0]
            dx = x + patch_size[0]
            y = j * patch_size[1]
            dy = y + patch_size[1]
            patches_array.append(im_data[x:dx, y:dy, :])

    patches_array = np.asarray(patches_array)
    out = {'patch_size': patch_size,
           'patch_per_row': patch_per_row,
           'patch_per_column': patch_per_column,
           'patches_array': patches_array}
    return out

# def save_HeatMap(heatMap, name_img = "heatmap.png"):
#     """
#     Save heatMap
#
#     :param Image_list:
#     :return:
#     """
#
#     print("Saving images heatmap (" + name_img + ")")
#
#     ## TODO: Use matplotlib for this
#     img = denormalize2uint8(heatMap * (-1)) # To make the positive in red and negative in blue
#     image_cm = cv2.applyColorMap(np.array(img, dtype=np.uint8), cv2.COLORMAP_JET)
#     img_out = array_to_img(image_cm)#, data_format='channels_first')
#     img_out.save(name_img)














