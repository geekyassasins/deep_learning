# Main script to train one user model
# Tensorflow: 1.3.0
# Keras: 2.1.3

# Parameters
# Syntax running in terminal example
# python process_all.py --use_database PulseID --chunk_time 1 --batch_size 16 --num_epochs 20 --random_shuffles 150
# --restart_ID_model True --use_class_weights False --ini_spk 1 --n_spks 1 --num_filters [6,6,6] --filter_sizes [50,30,20]
# --earlystop True  --patience 7 --expID usr1_ct1_ep20_bs16_rs150_Es7_0320_nf666_fs503020 --nusr_train 35
# ==================================================

# ---------- TO AVOID WARNINGS LIKE: -------- #
# The TensorFlow library wasn't compiled to use SSE4.1 instructions
# os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# ------------------------------------------- #

# -------------------- IMPORTS -------------------- #
from atomic_test import atomic_test
from Model_PulseID import get_model
from plot_ROC_and_Report import plot_ROC_and_Report
from split_db_TROIKA import split_db_TROIKA
from split_db_pulseID import split_db_pulseID
from threshold_scores import optimal_threshold

import math
import collections                                          # for the circular queue
import datetime
import json
import matplotlib
matplotlib.use('Agg')                                       # matplotlib backend ('Agg') for .png
import numpy as np
import os
import sys
import tensorflow as tf

from sklearn.metrics import *
from sklearn.utils import class_weight

from keras.backend.tensorflow_backend import set_session
from keras.models import clone_model
from keras.optimizers import SGD

# --------------- CUDA configuration -------------- #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"                    # use de GPU number 1
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True                      # Use a reasonable and fair memory percentage (to be respectful with others)
# allow_soft_placement = True                               # For automatic cuda device asignation
# config.gpu_options.per_process_gpu_memory_fraction = 0.3  # Manually cuda's memory limitation
set_session(tf.compat.v1.Session(config=config))

# ---------------- Reproducibility ---------------- #
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

#  Data loading params
tf.flags.DEFINE_string("use_database", "PulseID", "Database (PulseID|TROIKA) on run experiments (default: PulseID) ")
tf.flags.DEFINE_float("percentage_test", 0.15, "Data percentage for testing (in terms of users) (default: 0.15)")
tf.flags.DEFINE_float("percentage_validation", 0.25, "Data percentage for developing (in terms of samples -- chunks --  per usuari) (default: 0.25")
tf.flags.DEFINE_integer("chunk_time", 2,"Time (in seconds) to perform the minimum chunk of pulse signal to process")
#tf.flags.DEFINE_string("preload_file", "../clean_data.pickle", "Data source for the positive data.")


# Model Hyperparameters
tf.flags.DEFINE_integer("nusr_train", 25, "number of users for training + validation, it includes target ID and world impostor IDs (default: 25)")
tf.flags.DEFINE_integer("n_spks", 1, "Compute the model sequentially from ini_spk to n_spks initials IDs (default: 1)")
tf.flags.DEFINE_integer("ini_spk", 1, "Compute the model sequentially from ini_spk to n_spks initials IDs (default: 1)")
tf.flags.DEFINE_boolean("use_class_weights", False, "Use class weights estimated from training samples for weighting the loss function (default: False)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.2, "Dropout keep probability (default: 0.5)")
tf.flags.DEFINE_float("dropout_keep_prob2", 0.8, "Dropout2 keep probability (default: 0.8)")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda (default: 0.0)")
tf.flags.DEFINE_boolean("restart_ID_model", True, "Restart hyperparameters for each subject iteration, otherwise use previous parameters from subject ID-1 in a sequential way (default: True)")
tf.flags.DEFINE_string("num_filters", "[6,6,6]", "Number of filters of each Conv1D (it MUST have the same number of elements than filter_sizes")
tf.flags.DEFINE_string("filter_sizes", "[100,50,25]", "Size of the Conv1D's filters (it MUST have the same number of elements than num_filters")

# Training parameters
tf.flags.DEFINE_integer("random_shuffles", 5, " (default: 5)")
tf.flags.DEFINE_integer("batch_size", 8, "Batch Size (default: 8)")
tf.flags.DEFINE_integer("num_epochs", 10, "Number of training epochs (default: 10)")
tf.flags.DEFINE_integer("patience", 3, "Minimum number of evaluations before activating early stopping (default: 3)")
tf.flags.DEFINE_boolean("earlystop", True, "Stop training early when the validation AUC does not improve")
tf.flags.DEFINE_integer("pool_factor", 1, "pool_factor = size of the output of pool 'layer'")
#tf.flags.DEFINE_float("early_stopping_improvement_threshold", 0.0005, "The improvement that the validation score must have, otherwise early stopping will trigger. If validation score has an allways increasing plateau, early stopping will trigger (default: 0.005)")

# Misc Parameters
tf.flags.DEFINE_string("expID", "_Equal_Superepochs", "Identification name given to the experiment, used for results folder (default: _Equal_Superepochs)" )
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement (default: True)")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices (default: False)")

FLAGS = tf.flags.FLAGS
import sys
FLAGS(sys.argv)
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")
allow_soft_placement = FLAGS.allow_soft_placement

# ------------------------------------------- #

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def main(argv):

    # --------------------- Initializations ----------------------- #
    now = datetime.datetime.now()
    date = str('{:04d}'.format(now.year)) + str('{:02d}'.format(now.month)) + str('{:02d}'.format(now.day))

    # -------- Flags ---------- #
    chunk_time = FLAGS.chunk_time
    db = FLAGS.use_database
    expID = FLAGS.expID
    patience = FLAGS.patience
    percentage_test = FLAGS.percentage_test
    percentage_validation = FLAGS.percentage_validation
    use_class_weights = FLAGS.use_class_weights
    target_id = FLAGS.ini_spk

    # ---------- NN ------------ #
    restart_ID_model = FLAGS.restart_ID_model
    filters = eval(FLAGS.num_filters)
    filter_sizes = eval(FLAGS.filter_sizes)

    # ---- Training options ---- #
    batch_size = FLAGS.batch_size
    earlystop = FLAGS.earlystop
    epochs = FLAGS.num_epochs
    ini_spk = FLAGS.ini_spk
    nusr_train = FLAGS.nusr_train
    n_spks = FLAGS.n_spks
    pool_factor = FLAGS.pool_factor
    random_shuffles = FLAGS.random_shuffles

    if not os.path.exists('../models'):
        os.makedirs('../models')

    # -------------------------- Main ----------------------------- #
    print('=-=-=-=-=-==-=-=-=-=-=-=-=-=-=-=-=-==-=-=-=-=-=-=-')
    print(bcolors.OKBLUE + 'Chunk_Time:' + bcolors.ENDC, chunk_time)

    if db == 'PulseID':
        fs = 200
        path_df = './DataFrame_PulseID.pkl'
        signal = 'PPG_GDF'
        df, num_subjects, num_segments = split_db_pulseID(path_df=path_df, fs=fs, chunk_time=chunk_time,
                                                          percentage_validation=percentage_validation,
                                                          percentage_test=percentage_test, nusr_train=nusr_train,
                                                          target=target_id)


    elif db == 'TROIKA':
        fs = 125
        signal = 'PPG'
        path_df = './DataFrame_TROIKA.pkl'
        df, num_subjects, num_segments = split_db_TROIKA(path_df=path_df, fs=fs, chunk_time=chunk_time,
                                                         percentage_validation=percentage_validation,
                                                         percentage_test=percentage_test, nusr_train=nusr_train,
                                                         target=target_id)
    #print(df.shape)
    data_train, labels_train, data_validation, labels_validation, data_test, labels_test, data_testIDs, labels_testIDs, data_zscore = atomic_test(
        df=df, signal=signal,
        num_subjects=num_subjects,
        num_segments=num_segments, target=target_id,
        percentage_test=percentage_test)                    # data_train & labels_train are BALANCED
    #print(data_train.shape[1])
    data_train = data_train.reshape(data_train.shape[0], 1)                        # Reshaping necessary to train

    if restart_ID_model != True:
        model = get_model(filters=filters, kernel_size=filter_sizes, x_train=data_train, cnn_stride=1, pool_factor=pool_factor)
        print(bcolors.WARNING+'Keeping previous trained parameters from previous models'+bcolors.ENDC)


    for i in range(ini_spk,n_spks+1): # round(num_subjects * percentage_test) + 1):
        # Different model for each subject ID
        if restart_ID_model == True:
            model = get_model(filters=filters, kernel_size=filter_sizes, x_train=data_train, cnn_stride=1, pool_factor=pool_factor)
            print(bcolors.WARNING+'Reseting NN model per userID '+bcolors.ENDC,i)
            sgd = SGD(lr=0.0001, momentum=0.0, decay=0.0, nesterov=True)
            model.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
            model.summary()
        print('*****************************************')
        print(bcolors.OKGREEN+'ID:'+bcolors.ENDC, i)
        target_id=i
        labels_validation = []
        labels_test = []
        score_validation = []
        score_test = []

        # For the Circular Queue:
        buffer_auc = collections.deque(maxlen=patience+1)                       #circular queue declaration
        buffer_models = collections.deque(maxlen=patience + 1)
        buffer_score_validation = collections.deque(maxlen=patience + 1)
        buffer_score_test = collections.deque(maxlen=patience + 1)
        buffer_score_testIDs = collections.deque(maxlen=patience + 1)
        buffer_labels_validation = collections.deque(maxlen=patience + 1)
        buffer_labels_test = collections.deque(maxlen=patience + 1)
        buffer_labels_testIDs = collections.deque(maxlen=patience + 1)

        for i in range(patience + 1):
            buffer_auc.append(0.0)

        print(bcolors.WARNING + 'Entering at Random_Shuffle for'+ bcolors.ENDC)
        for r in range(1, random_shuffles + 1):
            print('------------------------------------------')

            data_train, labels_train, data_validation, labels_validation, data_test, labels_test, data_testIDs, labels_testIDs, data_zscore = atomic_test(
                df=df, signal=signal,
                num_subjects=num_subjects,
                num_segments=num_segments,
                target=target_id,
                percentage_test=percentage_test)

            data_train = data_train.reshape(data_train.shape[0], 1, 1)
            data_test = data_test.reshape(data_test.shape[0],1, 1)
            data_testIDs = data_testIDs.reshape(data_testIDs.shape[0], 1, 1)
            data_validation = data_validation.reshape(data_validation.shape[0], 1, 1)
            #up=np.array(labels_validation,dtype='int')
            develop_weights = class_weight.compute_class_weight("balanced",(np.unique(labels_validation)), (labels_validation))
            develop_weights_vec = []
            #develop_weights=develop_weights.reshape(2,1,1)
            #print((develop_weights[0]))
            for j in range(0, len(labels_validation)):
                if labels_validation[j] == 0:
                    develop_weights_vec.append(develop_weights[0])
                elif labels_validation[j] == 1:
                    develop_weights_vec.append(develop_weights[1])
            develop_weights_vec = np.array(develop_weights_vec)

            class_weights = {0: 1., 1: 150.} # Weights in case of unbalanced validation data
            #  1 appearence of class 0 equals 150 appearences of class 1
            #data_validation=[0]*19
            data_train=np.resize(data_train,(19,19,1))
            labels_train=labels_train[19:]
            #print(batch_size)
            data_validation=np.resize(data_validation,(2,19,1))
            #labels_train=np.reshape(1,19)
            labels_train=np.reshape(labels_train,(19,1))
            if use_class_weights == True:
                history = model.fit(x=data_train, y=labels_train, batch_size=batch_size, epochs=epochs, verbose=0,
                                   callbacks=None,
                                    validation_split=0.0,
                                    validation_data=(data_validation, labels_validation, develop_weights_vec),
                                    shuffle=True,
                                    class_weight=class_weights,
                                    sample_weight=None,
                                    initial_epoch=0)  # default values
            else:
                """history = model.fit(x=data_train, y=labels_train, batch_size=batch_size, epochs=epochs, verbose=0,
                                   
                                    validation_split=0.0,
                                    validation_data=(data_validation, labels_validation, develop_weights_vec),
                                    shuffle=True,
                                    class_weight=None,
                                    sample_weight=None,
                                    initial_epoch=0)"""  # default values

            #print(data_test.shape)
            data_test=np.resize(data_test,(7,19,1))
            data_testIDs=np.resize(data_testIDs,(7,19,1))
            score_validation = model.predict(x=data_validation, batch_size=1, verbose=0)
            score_test = model.predict(x=data_test, batch_size=1, verbose=0)
            score_testIDs = model.predict(x=data_testIDs, batch_size=1, verbose=0)

            score_validation = score_validation.reshape(score_validation.shape[0])
            score_test = score_test.reshape(score_test.shape[0])
            score_testIDs = score_testIDs.reshape(score_testIDs.shape[0])

            labels_validation = labels_validation.reshape(labels_validation.shape[0])
            labels_test = labels_test.reshape(labels_test.shape[0])
            labels_testIDs = labels_testIDs.reshape(labels_testIDs.shape[0])

            roc_auc = roc_auc_score(labels_validation,score_validation)
            roc_auc_test = roc_auc_score(labels_test,score_test)
            print('random shuffle iteration: ', r ,'Validation AUC: ', roc_auc, 'Test AUC: ',roc_auc_test)
            model_copy = clone_model(model) # necesary to clone to create a copy of the whole model. If we do not do this,
                                            # the model is passed as pointer by default
            model_copy.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])
            print("\n---------------------------------------\nmodel_copy summary: \n")
            model_copy.summary()
            print("\n---------------------------------------\n")

            # Circular queue
            # If the AUC decreases 'patience' consecutive stop training and save the model & scores for the last 'best' model
            # (actual model -patience steps)
            buffer_auc.append(roc_auc) #buffer_auc[0 --> patience]
            buffer_models.append(model_copy)
            buffer_score_validation.append(score_validation)
            buffer_score_test.append(score_test)
            buffer_score_testIDs.append(score_testIDs)
            buffer_labels_validation.append(labels_validation)
            buffer_labels_test.append(labels_test)
            buffer_labels_testIDs.append(labels_testIDs)

            # Boundaries to set the maximum FPR value
            fprbound1= 0.2 # more restrictive
            fprbound2= 0.3 # "checkpoint" value

            if (buffer_auc.index(max(buffer_auc)) == 0 or r == random_shuffles) and earlystop == True:
                if buffer_auc.index(max(buffer_auc)) == 0:
                    index_modelok = 0
                elif r == random_shuffles:
                    index_modelok = -1

                if not os.path.exists('../models/'+ db + '/' + date + '/'):
                    os.makedirs('../models/' + db + '/' + date + '/')

                buffer_models[index_modelok].save('../models/'+db+'/'+date+ '/model_'+db+'_ES' + str(patience) + '_' + expID + '.h5')

                if not os.path.exists('../results/'+ db + '/' + 'scores_ES' + str(patience) + '_'+date+'/'):
                    os.makedirs('../results/'+ db + '/' + 'scores_ES' + str(patience) + '_'+date+'/')
                filename = '../results/'+ db + '/' + 'scores_ES' + str(patience) + '_'+date+'/' + expID + '_scores'

                # Computing the optimal threshold for a given scores
                validation_metrics, test_metrics, testIDs_metrics, threshold = optimal_threshold(buffer_score_validation[index_modelok],
                                                                                                 buffer_score_test[index_modelok],
                                                                                                 buffer_score_testIDs[index_modelok],
                                                                                                 buffer_labels_validation[index_modelok],
                                                                                                 buffer_labels_test[index_modelok],
                                                                                                 buffer_labels_testIDs[index_modelok],
                                                                                                 fprbound1,
                                                                                                 fprbound2)

                # Saving scores in a .json file
                scores = {'score_validation': buffer_score_validation[index_modelok].tolist(),
                          'score_test': buffer_score_test[index_modelok].tolist(),
                          'score_testIDs': buffer_score_testIDs[index_modelok].tolist(),
                          'labels_validation': buffer_labels_validation[index_modelok].tolist(),
                          'labels_test': buffer_labels_test[index_modelok].tolist(),
                          'labels_testIDs': buffer_labels_testIDs[index_modelok].tolist(),
                          'threshold': threshold.tolist()}

                with open(filename + '.json', 'w') as fp:
                    json.dump(scores, fp)

                # pickle.dump((buffer_score_validation[index_modelok], buffer_score_test[index_modelok], buffer_labels_validation[index_modelok],
                #              buffer_predict_labels_validation[index_modelok], buffer_labels_test[index_modelok]),open(filename,"wb"))

                break

        if earlystop == False:
            if not os.path.exists('../models/' + db + '/' + date+'/'):
                os.makedirs('../models/' + db + '/' + date + '/')
            model.save('../models/' + db + '/' + date + '/model_' + db + '_' + expID + '.h5')
            if not os.path.exists('../results/' + db + '/' + 'scores_'+date+'/'):
                os.makedirs('../results/' + db + '/' + 'scores_'+ date + '/')
            filename = '../results/' + db + '/' + 'scores_' + date + '/' + expID +  '_scores'

            # Computing the optimal threshold for a given scores
            validation_metrics, test_metrics, testIDs_metrics, threshold = optimal_threshold(score_validation,
                                                                                             score_test,
                                                                                             score_testIDs,
                                                                                             labels_validation,
                                                                                             labels_test,
                                                                                             labels_testIDs,
                                                                                             fprbound1,
                                                                                             fprbound2)
            # Saving the scores in a .json file
            scores = {'score_validation': score_validation.tolist(),
                      'score_test': score_test.tolist(),
                      'score_testIDs': score_testIDs.tolist(),
                      'labels_validation': labels_validation.tolist(),
                      'labels_test': labels_test.tolist(),
                      'labels_testIDs': labels_testIDs.tolist(),
                      'threshold': threshold.tolist()}


            with open(filename+'.json', 'w') as fp:
                json.dump(scores, fp)


        # Unpacking variables
        fpr_validation = validation_metrics[0]
        tpr_validation = validation_metrics[1]
        roc_auc = validation_metrics[2]
        report_validation = validation_metrics[3]
        cnf_matrix_validation = validation_metrics[4]

        fpr_test = test_metrics[0]
        tpr_test = test_metrics[1]
        roc_auc_test = test_metrics[2]
        report_test = test_metrics[3]
        cnf_matrix_test = test_metrics[4]

        fpr_testIDs = testIDs_metrics[0]
        tpr_testIDs = testIDs_metrics[1]
        roc_auc_testIDs = testIDs_metrics[2]
        report_testIDs = testIDs_metrics[3]
        cnf_matrix_testIDs = testIDs_metrics[4]


        print('FINAL ITERATION -> Validation AUC: ', roc_auc, 'Test AUC: ', roc_auc_test, 'TestIDs AUC: ', roc_auc_testIDs)

        plot_ROC_and_Report(db, expID, date, 'validation', False, filter_sizes, filters, fpr_validation, tpr_validation,
                            roc_auc, report_validation, cnf_matrix_validation, threshold)

        plot_ROC_and_Report(db, expID , date, 'test', False, filter_sizes, filters, fpr_test, tpr_test,
                            roc_auc_test, report_test,cnf_matrix_test, threshold)

        plot_ROC_and_Report(db, expID, date, 'testIDs', False, filter_sizes, filters, fpr_testIDs, tpr_testIDs,
                            roc_auc_testIDs, report_testIDs, cnf_matrix_testIDs, threshold)

if __name__ == "__main__":
    main(sys.argv[1:])
