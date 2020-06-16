import matplotlib
matplotlib.use('Agg') # Remove this line for showing graphics in the display
import matplotlib.pyplot as plt
import numpy as np
import itertools
import os

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def plot_ROC_and_Report(DB, expID, date, condition, zscore_norm, filter_sizes, filters, fpr, tpr, roc_auc, report, cnf_matrix, threshold):
    directory = '../results/' + DB + '/' + DB + '_' + date + '/' + expID + '/'
    expID = expID + '_TH' + str(threshold)
    if not os.path.exists(directory):
        os.makedirs(directory)
    if condition == 'validation':
        if zscore_norm == False:
            text_file = open(directory + 'ROC_Validation_' + expID + ".txt", "w")
        else:
            text_file = open(directory + 'ROC_ZNormValidation_' + expID + ".txt", "w")
    elif condition == 'test':
        if zscore_norm == False:
            text_file = open(directory + 'ROC_Test_' + expID + ".txt", "w")
        else:
            text_file = open(directory + 'ROC_ZNormTest_' + expID + ".txt", "w")
    elif condition == 'testIDs':
        if zscore_norm == False:
            text_file = open(directory + 'ROC_TestIDs_' + expID + ".txt", "w")
        else:
            text_file = open(directory + 'ROC_ZNormTestIDs_' + expID + ".txt", "w")

    text2write = report + '\nArea Under the Curve (AUC): ' + str(roc_auc) + '\nThreshold: ' + str(
        threshold) + '\n'
    text_file.write(text2write)
    text_file.close()

    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-')
    print('REPORT:\n', report)
    print("\nArea Under the Curve (AUC):", roc_auc)
    print('-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-.-\n')

    fig = plt.figure()
    ax = fig.add_subplot(111, adjustable='box', aspect=1)
    fig, ax = plt.subplots(figsize=(10, 10))
    if condition == 'validation':
        fig.suptitle('ROC Validation', fontsize=18, fontweight='bold')
    elif condition == 'test':
        fig.suptitle('ROC Test', fontsize=18, fontweight='bold')
    elif condition == 'testIDs':
        fig.suptitle('ROC TestIDs', fontsize=18, fontweight='bold')

    # fig.subplots_adjust(top=0.85)
    ax.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    ax.legend(loc='lower right', fontsize = 16)
    ax.plot([-0.1, 1.1], [-0.1, 1.1], 'r--')
    ax.set_title(expID, fontsize = 18)
    ax.set_xlabel('False Positive Rate', fontsize = 16)
    ax.set_ylabel('True Positive Rate', fontsize = 16)
    ax.text(0.99, 0.17, 'number of filters: ' + str(filters),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='blue', fontsize=14)
    ax.text(0.99, 0.12, 'filter sizes: ' + str(filter_sizes),
            verticalalignment='bottom', horizontalalignment='right',
            transform=ax.transAxes,
            color='blue', fontsize=14)
    ax.axis([-0.01, 1.01, -0.01, 1.01])
    # ax.set_aspect('equal', 'datalim')

    if condition == 'validation':
        plt.savefig(directory + 'ROC_Validation_' + expID + '.png')
        plt.clf()
        plt.close()
        np.set_printoptions(precision=2)
        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes={0,1}, normalize=True,title='Normalized Validation Confusion Matrix')
        plt.savefig(directory + 'CM_Validation_' + expID + '.png')


    elif condition == 'test':
        plt.savefig(directory + 'ROC_Test_' + expID + '.png')
        plt.clf()
        plt.close()
        np.set_printoptions(precision=2)
        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes={0,1}, normalize=True,title='Normalized Test Confusion Matrix')
        plt.savefig(directory + 'CM_Test_' + expID + '.png')

    elif condition == 'testIDs':
        plt.savefig(directory + 'ROC_TestIDs_' + expID + '.png')
        plt.clf()
        plt.close()
        np.set_printoptions(precision=2)
        # Plot normalized confusion matrix
        plot_confusion_matrix(cnf_matrix, classes={0,1}, normalize=True,title='Normalized TestIDs Confusion Matrix')
        plt.savefig(directory + 'CM_TestIDs_' + expID + '.png')

    plt.clf()
    plt.close()
    #plt.show()
