#============================================================
#
#   Metrics
#
#  author: Francisco Perdigon Romero
#  email: fperdigon88@gmail.com
#  github id: fperdigon
#
#============================================================


from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, balanced_accuracy_score, \
    recall_score, classification_report, auc, roc_curve, roc_auc_score

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LinearSegmentedColormap

import itertools

import numpy as np

# def generate_k_fold_ROC_curve(list_test_set_GT, list_test_pred_keras):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import roc_curve, auc
#     import numpy as np
#     from scipy import interp
#
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     for i in range(len(list_test_set_GT)):
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(list_test_set_GT[i][:, 0],
#                                          list_test_pred_keras[i][:, 0])
#
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#         i += 1
#
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#              label='Chance', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#     plt.show()


def classification_metrics(y_test, y_pred):
    n_classes = len(np.unique(y_test.argmax(axis=1)))

    acc = accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), normalize=True)

    if n_classes <= 2:
        precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        f1_score_m = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='weighted') #'binary'??
        balanced_acc = balanced_accuracy_score(y_test.argmax(axis=1), y_pred.argmax(axis=1))
        fpr, tpr, thresholds = roc_curve(y_test[:, 0], y_pred[:, 0])
        auc_metric = auc(fpr, tpr)
    else:
        precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        f1_score_m = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='macro')
        # precision = precision_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
        # recall = recall_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
        # f1_score = f1_score(y_test.argmax(axis=1), y_pred.argmax(axis=1), average='micro')
        balanced_acc = 0
        y_scores = y_pred[y_pred.argmax(axis=1)]
        #auc_metric = roc_auc_score(y_test.argmax(axis=1),y_scores) # TODO: Check if this works
        auc_metric = 0

    print('')
    print('################## CLASSIFICATION REPORT ##################')
    print('')
    print('Accuracy: %.2f' % acc)
    print('Balanced accuracy: %.2f' % balanced_acc)
    print('f1-score: %.2f' % f1_score_m)
    print('Precision: %.2f' % precision)
    print('Recall (Sensitivity): %.2f' % recall)
    print('Area under curve (AUC): %.2f' % auc_metric)

    print('')
    print('')
    print(classification_report(y_test.argmax(axis=1), y_pred.argmax(axis=1)))


    metrics_dict = {}
    metrics_dict['accuracy'] = acc
    metrics_dict['baccuracy'] = balanced_acc
    metrics_dict['f1-score'] = f1_score
    metrics_dict['auc'] = auc_metric
    metrics_dict['precision'] = precision
    metrics_dict['recall'] = recall

    return metrics_dict


def confusion_matrix_plot(y_test, y_pred, classes=None, custom_colors=False):

    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    title = 'Confusion matrix'

    if custom_colors:
        # Custom colors colormap
        colors = [(0.925, 0.7, 0.7), (0.83, 0.925, 0.906)]
        cmap_name = 'my_cmap'
        n_bin = 2048
        cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    else:
        cmap = plt.cm.Blues

    if classes == None:
        n_classes = len(np.unique(y_test.argmax(axis=1)))
        classes = [str(i) for i in range(n_classes)]

    print('Generating confusion matrix ...')
    print('')

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # Generate a non normalized confusion matrix
    print('Non Normalized confusion matrix')
    print('')
    print(cm)
    print('')


    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Next three lines solves the squeezed image problem recalculating the limits
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()

    plt.show(block=False)
    #plt.show()
    plt.savefig('fig_' + title + '.png', dpi=300)
    plt.close()

    # Generate a non normalized confusion matrix
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    print('Normalized confusion matrix')
    print('')
    print(cm_n)
    print('')

    plt.figure()
    plt.imshow(cm_n, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(0, 1)  # set the colorbar boudaries
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    fmt = '.2f'

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm_n.shape[0]), range(cm_n.shape[1])):
        plt.text(j, i, format(cm_n[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm_n[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    # Next three lines solves the squeezed image problem recalculating the limits
    ax = plt.gca()
    ax.relim()
    ax.autoscale_view()

    plt.tight_layout()

    plt.show(block=False)
    plt.savefig('fig_' + title + '_normalized.png', dpi=300)
    plt.close()

    print('Confusion matrix done')

    print('')


def confusion_matrix_normalized(y_test, y_pred):

    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    # Generate a non normalized confusion matrix
    cm_n = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return cm_n


def confusion_matrix_plot_custom_cm(cm_n, classes=None):  # this version is for plotting one already generated CM
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.colors import LinearSegmentedColormap
    import numpy as np
    import itertools

    matplotlib.rcParams['font.sans-serif'] = "Arial"
    matplotlib.rcParams['font.family'] = "sans-serif"

    title = 'Confusion matrix'


    # Custom colors colormap
    colors = [(0.925, 0.7, 0.7), (0.83, 0.925, 0.906)]
    cmap_name = 'my_cmap'
    n_bin = 2048
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bin)

    # cmap = plt.cm.Blues


    if classes == None:
        n_classes = cm_n.shape[0]
        classes = [str(i+1) for i in range(n_classes)]

    print('Generating confusion matrix ...')
    print('')

    print('Normalized confusion matrix')
    print('')
    print(cm_n)
    print('')

    plt.figure()
    plt.imshow(cm_n, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    plt.clim(0, 1) # set the colorbar boudaries
    tick_marks = np.arange(cm_n.shape[0])
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'

    thresh = cm_n.max() / 2.
    for i, j in itertools.product(range(cm_n.shape[0]), range(cm_n.shape[1])):
        # plt.text(j, i, format(cm_n[i, j], fmt),
        #          horizontalalignment="center",
        #          color="white" if cm_n[i, j] > thresh else "black")

        plt.text(j, i, format(cm_n[i, j], fmt),
                 horizontalalignment="center",
                 color="black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    plt.tight_layout()

    plt.show(block=False)
    plt.savefig(title + '_normalized.png', dpi=300)
    plt.close()

    print('Confusion matrix done')

    print('')

def generate_roc_plus_auc(y_test, y_pred_keras):

    print('Generating ROC courve ...')

    fpr_keras, tpr_keras, thresholds_keras = roc_curve(y_test[:,0], y_pred_keras[:,0])
    auc_keras = auc(fpr_keras, tpr_keras)

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show(block=False)
    plt.savefig('fig_ROC_curve.png', dpi=300)
    plt.close()

    # Zoom in view of the upper left corner.
    plt.figure()
    plt.xlim(0, 0.2)
    plt.ylim(0.8, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_keras, tpr_keras, label='AUC = {:.3f}'.format(auc_keras))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve (zoomed in at top left)')
    plt.legend(loc='best')
    plt.show(block=False)
    plt.savefig('fig_ROC_curve(zoomed_in_at_top_left).png', dpi=300)
    plt.close()

    print(' ')
    print('AUC = {:.3f}'.format(auc_keras))
    print(' ')

    print('ROC courve done')
    print('')

# def generate_k_fold_ROC_curve(list_test_set_GT, list_test_pred, toPNG=False, PNGName='ROC.png'):
#     import matplotlib.pyplot as plt
#     from sklearn.metrics import roc_curve, auc
#     import numpy as np
#     from scipy import interp
#
#     plt.figure()
#
#     tprs = []
#     aucs = []
#     mean_fpr = np.linspace(0, 1, 100)
#
#     for i in range(len(list_test_set_GT)):
#         # Compute ROC curve and area the curve
#         fpr, tpr, thresholds = roc_curve(list_test_set_GT[i][:, 0],
#                                          list_test_pred[i][:, 0])
#
#         tprs.append(interp(mean_fpr, fpr, tpr))
#         tprs[-1][0] = 0.0
#         roc_auc = auc(fpr, tpr)
#         aucs.append(roc_auc)
#         plt.plot(fpr, tpr, lw=1, alpha=0.3,
#                  label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#
#         i += 1
#
#     plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
#              label='Chance', alpha=.8)
#
#     mean_tpr = np.mean(tprs, axis=0)
#     mean_tpr[-1] = 1.0
#     mean_auc = auc(mean_fpr, mean_tpr)
#     std_auc = np.std(aucs)
#     plt.plot(mean_fpr, mean_tpr, color='b',
#              label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
#              lw=2, alpha=.8)
#
#     std_tpr = np.std(tprs, axis=0)
#     tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
#     tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
#     plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
#                      label=r'$\pm$ 1 std. dev.')
#
#     plt.xlim([-0.05, 1.05])
#     plt.ylim([-0.05, 1.05])
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('Receiver operating characteristic')
#     plt.legend(loc="lower right")
#
#     if toPNG:
#         plt.savefig(PNGName)
#     else:
#         plt.show()