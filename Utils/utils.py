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














