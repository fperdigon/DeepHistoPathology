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
import matplotlib
import matplotlib.gridspec as gridspec
import os
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, grey_dilation


if __name__ == "__main__":
    data_folder = './data'
    recons_folder = 'reconst_data'
    train_list_file = './Data_Preparation/cases_train.txt'
    val_list_file = './Data_Preparation/cases_val.txt'
    test_list_file = './Data_Preparation/cases_test.txt'

    # Reading hps from test to generate images
    test_list = []
    with open(test_list_file) as fp:
        Lines = fp.readlines()
        for line in Lines:
            test_list.append(line.strip())

    # Generate HPS image from patches
    hps_path_list = []

    pbar = tqdm(total=len(test_list))
    pbar.set_description(desc=' Generating HPS slide image')
    for hps in test_list:
        df = data_folder + '/' + data_preparation.uzip_folder
        # Determines if the hps folder exists
        if os.path.exists(df + '/' + hps):
            # determine if the image was previously reconstructed
            if not os.path.exists(data_folder + '/' + recons_folder + '/' + hps + '_hps.png'):
                [img_big, class_patches] = utils.hps_image_reconst_from_patches(hps, df)
                hps_path = utils.save_reconst_hps_to_png(img=img_big,
                                                         hps_slide=hps,
                                                         save_folder=data_folder + '/' + recons_folder)
                hps_path_list.append(hps_path)
            else:
                hps_path_list.append(data_folder + '/' + recons_folder + '/' + hps + '_hps.png')
        else:
            print(hps + ' folder does not exist')
        pbar.update(1)
    pbar.close()

    # Predict Cancer on the HPS images and Generate Heatmap
    pbar = tqdm(total=len(hps_path_list))
    pbar.set_description(desc=' Generating HPS Predictions heatmaps')
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
        # for i in range(patches_array.shape[0]):
        #     patches_array[i, :, :, :] = dl_pipeline.normalize_rgb(patches_array[i, :, :, :])

        # Deep Learning Model Prediction
        predictions = dl_pipeline.inference_dl(patches_array, model_filepath='weights.best.hdf5')
        # predictions = dl_pipeline.inference_dl(patches_array, model_filepath='norm_weights.best.hdf5')

        # Get cancer cell prediction
        cancer_cells_prediction = predictions[:, 1]

        # Heatmap generation
        heatmap = utils.heatmap_img_from_predictions(org_hps, cancer_cells_prediction, patch_size, patch_per_row, patch_per_column)

        # Dilate operation to reduce the fading intrudiced by the gaussian filter
        for i in range(10):
            heatmap = grey_dilation(heatmap, footprint=np.ones((3, 3)))

        # Apply a Gaussian filter to smooth the heatmap
        heatmap = gaussian_filter(heatmap, sigma=25)

        # Generating images
        def transparent_cmap(cmap, alpha_th=70, N=255):
            "Copy colormap and set alpha values"
            mycmap = cmap.__copy__()
            mycmap._init()
            alpha_values = np.linspace(0, 1, N + 4)
            alpha_values[alpha_values < alpha_th / N] = 0
            alpha_values[alpha_values >= alpha_th / N] = 1
            mycmap._lut[:, -1] = alpha_values
            return mycmap


        # Make the figure
        f = plt.figure()
        gs = gridspec.GridSpec(nrows=1, ncols=3, left=0.1, bottom=0.25, right=0.95, top=0.95,
                               wspace=0.05, hspace=0., width_ratios=[1, 1, 0.03])
        a0 = plt.subplot(gs[0])
        a1 = plt.subplot(gs[1])
        a2 = plt.subplot(gs[2])

        # Using subplots (It works as well)
        # f, (a0, a1, a2) = plt.subplots()
        # f, (a0, a1, a2) = plt.subplots(1, 3, gridspec_kw={'width_ratios': [1, 1, 0.03]})
        # plt.tight_layout()

        # Axis 0
        a0.imshow(org_hps)
        a0.axis('off')

        # Axis 1
        a1.imshow(org_hps)
        mycmap = transparent_cmap(cmap=plt.get_cmap('jet'), alpha_th=int(0.5 * 255), N=255)
        aa1 = a1.imshow(heatmap, cmap=mycmap,  alpha=0.3)
        #a1.imshow(heatmap)
        a1.axis('off')

        # # Axis 2
        norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
        cbar = f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap='jet'),
                          ax=a2, pad=.05, extend='neither', fraction=1)
        for t in cbar.ax.get_yticklabels():
            t.set_fontsize(5)
        a2.axis('off')

        fig1 = plt.gcf()
        plt.show(block=False)
        # plt.show(block=True)
        print('\nSaving image ...')
        fig1.savefig(hps_path + '_predictions.jpg', dpi=500, bbox_inches='tight', pad_inches=0.5)
        print('\nSaving image DONE.')
        plt.close()

        pbar.update(1)

    pbar.close()



