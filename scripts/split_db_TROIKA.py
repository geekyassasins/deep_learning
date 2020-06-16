import numpy as np
import pandas as pd
from math import floor

def split_db_TROIKA(path_df, fs, chunk_time, percentage_validation, percentage_test, nusr_train, target):
    # ----------------------- Declarations ------------------------ #
    num_segments = []
    chunk_size = fs * chunk_time

    df = pd.read_pickle(path_df) # Load Dataframe

    # --------------- Dataframe adaptation for CNN ---------------- #
    columns = ['ID', 'Time', 'PPG1', 'PPG2', 'PPG1_GDF', 'PPG2_GDF', 'PPG1_zscore', 'PPG2_zscore']
    empty_np = np.empty(shape=(1, chunk_size))
    newDF = pd.DataFrame([[empty_np, empty_np, empty_np, empty_np, empty_np, empty_np, empty_np, empty_np]],
                         columns=columns)

    for i in range(df.shape[0]):  # Iteration for each subject
        subject_to_work = df.iloc[i]
        for s in range(floor(subject_to_work[
                                 'Time'].size / chunk_size)):  # Iteration for each each chunk of data (usually around 23)
            newDF2 = pd.DataFrame([[subject_to_work['ID'],
                                    subject_to_work['Time'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG1'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG2'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG1_GDF'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG2_GDF'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG1_zscore'][(s) * chunk_size:(s + 1) * chunk_size],
                                    subject_to_work['PPG2_zscore'][(s) * chunk_size:(s + 1) * chunk_size]]],
                                  columns=columns)
            newDF = newDF.append(newDF2)
    num_subjects = i - 1
    df = newDF[1:]
    df = df.reset_index(drop=True)

    # Result is a table with:
    # - each subject appearing as many times as chunk_size fits into experiment length
    # - in each row: subject id + chunk number + all the signals that fit into that chunk

    # ----------------- Split data in Train/Develop/Test ------------------ #
    df.insert(1, "Split", "")
    segments = df['ID'].tolist()
    for k in range(1, num_subjects + 1):
        num_segments.append(segments.count(k))

    cont_validation = 1
    previous_id = df.iloc[0]['ID']
    cont_test = 0
    cont_testIDs = 0
    usr = 0

    print('num segments', num_segments)

    for i, row in df.iterrows():
        if previous_id != df.iloc[i]['ID']:
            previous_id = df.iloc[i]['ID']
            cont_validation = 1
            cont_test = 0
            usr = usr + 1

        if df.iloc[i]['ID'] <= nusr_train:
            if df.iloc[i]['ID'] == target:
                if cont_test <= round(num_segments[usr] * percentage_test):
                    df.loc[i, 'Split'] = 'Test'
                    cont_test += 1
                elif cont_testIDs <= round(num_segments[usr] * percentage_test):
                    df.loc[i, 'Split'] = 'TestIDs'
                    cont_testIDs += 1
                elif cont_validation <= round(num_segments[usr] * percentage_validation):
                    df.loc[i, 'Split'] = 'Validation'
                    cont_validation += 1
                else:
                    df.loc[i, 'Split'] = 'Train'

            else:
                if cont_test <= round(2 * num_segments[usr] * percentage_test):
                    df.loc[i, 'Split'] = 'Test'
                    cont_test += 1
                elif cont_validation <= round(num_segments[usr] * percentage_validation):
                    df.loc[i, 'Split'] = 'Validation'
                    cont_validation += 1
                else:
                    df.loc[i, 'Split'] = 'Train'

        elif df.iloc[i]['ID'] > nusr_train:
            df.loc[i, 'Split'] = 'TestIDs'

    # df.to_csv('Splitted_DF_TROIKA.csv', sep='\t', encoding='utf-8')	# Export DataFrame to .csv
    print('------------------------------------')

    return df, num_subjects, num_segments
