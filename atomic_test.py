import numpy as np
import sys


def atomic_test(df, signal, num_subjects, num_segments, target, percentage_test):
    print(df.shape)
    data_train = []
    labels_train = []
    data_train_candidate = []
    labels_train_candidate = []
    data_validation = []
    labels_validation = []
    data_test = []
    labels_test = []
    data_testIDs = []
    labels_testIDs = []
    data_zscore = []

    for i, row in df.iterrows():
        #print(df.iloc[i]['Split'])
        if target > round(num_subjects * percentage_test) and target < 0:
            sys.exit('ERROR -> Please insert a valid target.')

        elif df.iloc[i]['ID'] == target:
            if df.iloc[i]['Split'] == 'Validation':
                data_validation.append(df.iloc[i][signal])
                labels_validation.append(1)
            elif df.iloc[i]['Split'] == 'Train':
                data_train.append(df.iloc[i][signal])
                labels_train.append(1)
            elif df.iloc[i]['Split'] == 'Test':
                data_test.append(df.iloc[i][signal])
                labels_test.append(1)
            elif df.iloc[i]['Split'] == 'TestIDs':
                data_testIDs.append(df.iloc[i][signal])
                labels_testIDs.append(1)

        elif df.iloc[i]['ID'] != target:
            if df.iloc[i]['Split'] == 'Validation':
                data_validation.append(df.iloc[i][signal])
                labels_validation.append(0)
            elif df.iloc[i]['Split'] == 'Train':
                data_train_candidate.append(df.iloc[i][signal])
                labels_train_candidate.append(0)
            elif df.iloc[i]['Split'] == 'Test':
                data_test.append(df.iloc[i][signal])
                labels_test.append(0)
            elif df.iloc[i]['Split'] == 'TestIDs':
                data_testIDs.append(df.iloc[i][signal])
                labels_testIDs.append(0)
            elif df.iloc[i]['Split'] == 'Zscore':
                data_zscore.append(df.iloc[i][signal])

    # ============== data_train always the same shape ================ #
    # ----- Repeat data_train samples to reach max(num_segments) ----- #
    num_develop = max(num_segments) - len(data_train)
    counter = 0
    
    long = len(data_train)
    #print(max(num_segments))
    while ((max(num_segments) % len(data_train)) != 0) and counter < int(max(num_segments) / long):
        counter += 1
        data_train_length = len(data_train)
        for i in range(0, data_train_length):
            if ((max(num_segments) - num_develop) % len(data_train)) == 0:
                break
            data_train.append(data_train[i])
            labels_train.append(1)

    # -------------------- Add the random samples -------------------- #
    rand_candidates = np.random.random_integers(low=0, high=len(data_train_candidate) - 1, size=(len(data_train),))
    for i in range(0, len(rand_candidates)):
        index = rand_candidates[i]
        data_train.append(data_train_candidate[index])
        labels_train.append(0)

    data_train = np.array(data_train)
    labels_train = np.array(labels_train)
    data_validation = np.array(data_validation)
    labels_validation = np.array(labels_validation)
    data_test = np.array(data_test)
    labels_test = np.array(labels_test)
    data_testIDs = np.array(data_testIDs)
    labels_testIDs = np.array(labels_testIDs)
    data_zscore = np.array(data_zscore)

    return data_train, labels_train, data_validation, labels_validation, data_test, labels_test, data_testIDs, labels_testIDs, data_zscore
