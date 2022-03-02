# -*- coding: utf-8 -*-

#============================================================
#
#  Deep HistoPathology (DeepHP)
#  Test Full HPS Image reconstruction and cancer detection
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#===========================================================

import Data_Preparation.data_preparation as data_preparation
import DeepHP.dl_pipeline as dl_pipeline
import Utils.utils as utils
import matplotlib.pyplot as plt
import os


if __name__ == "__main__":
    data_folder = './data'
    recons_folder = 'reconst_data'
    train_list_file = './Data_Preparation/cases_train.txt'
    val_list_file = './Data_Preparation/cases_val.txt'
    test_list_file = './Data_Preparation/cases_test.txt'

    # TODO: Add Dataset download dataset function

    # Reading hps from test to generate images
    test_list = []
    with open(test_list_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            test_list.append(line.strip())

    # Generate HPS image from patches
    hps_path_list = []
    for hps in test_list:
        df = data_folder + '/' + data_preparation.uzip_folder
        if os.path.exists(df + '/' + hps):
            [img_big, class_patches] = utils.hps_image_reconst_from_patches(hps, df)
            hps_path = utils.save_reconst_hps_to_png(img=img_big, hps_slide=hps, save_folder=recons_folder)
            hps_path_list.append(hps_path)
        else:
            print(hps + ' folder does not exist')

    # Predict Cancer on the HPS images and Generate Heatmap
    for hps_path in hps_path_list:
        # Generate patches from HPS image
        patch_size = (50, 50)
        out_dict = utils.load_img_generate_patch_array(hps_path, patch_size)
        patch_size = out_dict['patch_size']
        patch_per_row = out_dict['patch_per_row']
        patch_per_column = out_dict['patch_per_column']
        patches_array = out_dict['patches_array']
        org_hps = out_dict['org_hps']

        # Apply RGB normalization to each patch
        for i in len(patches_array):
            patches_array[i] = dl_pipeline.normalize_rgb(patches_array[i])

        # Deep Learning Model Prediction
        predictions = dl_pipeline.inference_dl(patches_array)

        # Get cancer cell prediction
        cancer_cells_prediction = predictions[:, 1]

        # Heatmap generation
        heatmap = utils.heatmap_img_from_predictions(org_hps, predictions, patch_size, patch_per_row, patch_per_column)

        # Generating images

        def transparent_cmap(cmap, alpha_th=70, N=255):
            "Copy colormap and set alpha values"

            mycmap = cmap
            mycmap._init()
            alpha_values = np.linspace(0, 1, N + 4)
            alpha_values[alpha_values < alpha_th / N] = 0
            alpha_values[alpha_values >= alpha_th / N] = 1
            mycmap._lut[:, -1] = alpha_values
            return mycmap


        # MAke the figure
        f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.07]})

        # Axis 0
        a0.imshow(org_hps)
        a0.axis('off')

        # Axis 1
        a1.imshow(org_hps)
        mycmap = transparent_cmap(cmap=plt.get_cmap('jet'), alpha_th=int(0.2 * 255), N=255)
        a1.imshow(org_hps, cmap=mycmap)
        a1.axis('off')

        # # Axis 2
        # import matplotlib.image as mpimg
        #
        # img = mpimg.imread('jet_colorbar.png')
        # a2.imshow(img)
        # a2.axis('off')

        fig1 = plt.gcf()
        plt.show(block=False)
        # plt.show(block=True)
        print('Saving image ...')
        fig1.savefig(hps_path + '_predictions.jpg', dpi=500, bbox_inches='tight', pad_inches=0)
        print('Saving image DONE.')
        plt.close()

        print(dir + ' Procesed ...')



