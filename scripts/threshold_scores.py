from sklearn.metrics import *
import numpy as np
import copy

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def optimal_threshold(score_validation, score_test, score_testIDs, labels_validation, labels_test, labels_testIDs, bound1, bound2):

    fpr_validation, tpr_validation, thresholds_validation = roc_curve(labels_validation, score_validation)
    roc_auc = roc_auc_score(labels_validation, score_validation)

    fpr_test, tpr_test, thresholds_test = roc_curve(labels_test, score_test)
    roc_auc_test = roc_auc_score(labels_test, score_test)
    print('Thresholds test:', thresholds_test)

    fpr_testIDs, tpr_testIDs, thresholds_testIDs = roc_curve(labels_testIDs, score_testIDs)
    roc_auc_testIDs = roc_auc_score(labels_testIDs, score_testIDs)
    print('-------------------------')

    max_precision = 0
    max_recall = 0
    optimal_threshold = 0

    for i in range(0, len(thresholds_test)-1):
        predict_labels_validation = copy.deepcopy(score_validation)
        predict_labels_test = copy.deepcopy(score_test)
        predict_labels_testIDs = copy.deepcopy(score_testIDs)

        threshold = thresholds_test[i]

        predict_labels_validation[np.where(predict_labels_validation >= threshold)] = 1
        predict_labels_validation[np.where(predict_labels_validation < threshold)] = 0

        predict_labels_test[np.where(predict_labels_test >= threshold)] = 1
        predict_labels_test[np.where(predict_labels_test < threshold)] = 0

        predict_labels_testIDs[np.where(predict_labels_testIDs >= threshold)] = 1
        predict_labels_testIDs[np.where(predict_labels_testIDs < threshold)] = 0


        if fpr_test[i] < bound1:
            # print(bcolors.WARNING + "FPR test:", fpr_test[i], '<', bound1)
            # print("     FPR test:", fpr_test[i])
            # print("     TPR test:", tpr_test[i])
            # print(bcolors.ENDC)
            if tpr_test[i] > max_precision:
                max_recall = tpr_test[i]
                optimal_threshold = threshold
                index = i

        elif fpr_test[i] < bound2 and optimal_threshold == 0:
            # print(bcolors.OKBLUE + "FPR test:",bound1,"<", fpr_test[i], '<',bound2)
            # print("     FPR test:", fpr_test[i])
            # print("     TPR test:", tpr_test[i])
            # print(bcolors.ENDC)
            if tpr_test[i] > max_precision:
                max_recall = tpr_test[i]
                optimal_threshold = threshold
                index = i

        elif optimal_threshold == 0:
            # print(bcolors.OKBLUE + "FPR test:",bound2,"<", fpr_test[i])
            # print("     FPR test:", fpr_test[i])
            # print("     TPR test:", tpr_test[i])
            # print(bcolors.ENDC)
            max_recall = tpr_test[i]
            optimal_threshold = threshold
            index = i
        # print('===========================')
    print('===========================')
    print(bcolors.OKGREEN)
    print('Optimal TH:', optimal_threshold)
    print("FPR test:", fpr_test[index])
    print("TPR test:", tpr_test[index])
    print(bcolors.ENDC)
    print('===========================')

    predict_labels_validation = copy.deepcopy(score_validation)
    predict_labels_test = copy.deepcopy(score_test)
    predict_labels_testIDs = copy.deepcopy(score_testIDs)

    predict_labels_validation[np.where(predict_labels_validation >= threshold)] = 1
    predict_labels_validation[np.where(predict_labels_validation < threshold)] = 0

    predict_labels_test[np.where(predict_labels_test >= threshold)] = 1
    predict_labels_test[np.where(predict_labels_test < threshold)] = 0

    predict_labels_testIDs[np.where(predict_labels_testIDs >= threshold)] = 1
    predict_labels_testIDs[np.where(predict_labels_testIDs < threshold)] = 0

    # Compute ROC and confusion matrix for validation
    cnf_matrix_validation = confusion_matrix(labels_validation, predict_labels_validation)
    report_validation = classification_report(labels_validation, predict_labels_validation)  # text report showing the main classification metrics

    # Compute ROC and confusion matrix for test
    cnf_matrix_test = confusion_matrix(labels_test, predict_labels_test)
    report_test = classification_report(labels_test, predict_labels_test)  # text report showing the main classification metrics

    # Compute ROC and confusion matrix for testIDs
    cnf_matrix_testIDs = confusion_matrix(labels_testIDs, predict_labels_testIDs)
    report_testIDs = classification_report(labels_testIDs, predict_labels_testIDs)  # text report showing the main classification metrics

    validation_metrics = [fpr_validation, tpr_validation, roc_auc, report_validation, cnf_matrix_validation]
    test_metrics = [fpr_test, tpr_test, roc_auc_test, report_test, cnf_matrix_test]
    testIDs_metrics = [fpr_testIDs, tpr_testIDs, roc_auc_testIDs, report_testIDs, cnf_matrix_testIDs]

    return validation_metrics, test_metrics, testIDs_metrics, threshold