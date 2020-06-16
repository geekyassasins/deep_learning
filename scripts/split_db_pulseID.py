import numpy as np
import pandas as pd
from math import floor

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def split_db_pulseID(path_df, fs, chunk_time, percentage_validation, percentage_test, nusr_train, target):
    # ----------------------- Declarations ------------------------ #
    num_segments = []
    chunk_size = fs * chunk_time

    # --------------------- Load Dataframe ------------------------ #
    df = pd.read_pickle("./DataFrame_PulseID.pkl")
    #print(df.shape)


    # --------------- Dataframe adaptation for CNN ---------------- #
    columns = ['ID', 'Experiment', 'Time', 'PPG', 'PPG_GDF']
    empty_np = np.empty(shape=(1, chunk_size))
    newDF = pd.DataFrame([[empty_np, empty_np, empty_np, empty_np, empty_np]],
                         columns=columns)
    tmp_signalDF = pd.DataFrame([[empty_np, empty_np, empty_np, empty_np, empty_np]],
                         columns=columns)
    i=0
    #print(df.shape[0])

    #DATAFRAME ADAPTATION
    while i < df.shape[0]:  # Iteration for each experiment and user
        subject_to_work = df.iloc[i]

        if subject_to_work['Experiment'] > 2:
            previous_id = subject_to_work['ID']
            while (i < df.shape[0] and previous_id == df.iloc[i]['ID']):
                signalDF2 = pd.DataFrame([[df.iloc[i]['ID'],
                                    df.iloc[i]['Experiment'],
                                    df.iloc[i]['Time'],
                                    df.iloc[i]['PPG'],
                                    df.iloc[i]['PPG_GDF']]],
                                  columns=columns)

                i+=1
                tmp_signalDF = tmp_signalDF.append(signalDF2)


            signalDF = tmp_signalDF[1:]
            signalDF = signalDF.reset_index(drop=True)

            tmp_signalDF = pd.DataFrame([[empty_np, empty_np, empty_np, empty_np, empty_np]],
                                        columns=columns)

            if signalDF.shape[0] < 4 & signalDF.shape[0] > 2:
                print(signalDF.iloc[0]['Time'][-1])
                experiment_id_group = 345
                subject_to_work = pd.DataFrame([[signalDF.iloc[0]['ID'],
                                             experiment_id_group,
                                             np.concatenate((signalDF.iloc[0]['Time'],
                                                     signalDF.iloc[1]['Time'] + signalDF.iloc[0]['Time'][-1],
                                                     signalDF.iloc[2]['Time'] + signalDF.iloc[0]['Time'][-1] + signalDF.iloc[1]['Time'][-1]),axis=0),
                                             np.concatenate((signalDF.iloc[0]['PPG'],
                                                     signalDF.iloc[1]['PPG'],
                                                     signalDF.iloc[2]['PPG']),axis=0),
                                             np.concatenate((signalDF.iloc[0]['PPG_GDF'],
                                                     signalDF.iloc[1]['PPG_GDF'],
                                                     signalDF.iloc[2]['PPG_GDF']),axis=0)]],
                                          columns=columns)


            elif signalDF.shape[0] >= 4:
                experiment_id_group = 3457
                subject_to_work = pd.DataFrame([[signalDF.iloc[0]['ID'],
                                                 experiment_id_group,
                                                 np.concatenate((signalDF.iloc[0]['Time'],
                                                                 signalDF.iloc[1]['Time'] + signalDF.iloc[0]['Time'][-1],
                                                                 signalDF.iloc[2]['Time'] + signalDF.iloc[0]['Time'][-1] + signalDF.iloc[1]['Time'][-1],
                                                                 signalDF.iloc[3]['Time'] + signalDF.iloc[2]['Time'][-1] + signalDF.iloc[0]['Time'][-1] + signalDF.iloc[1]['Time'][-1]), axis=0),
                                                 np.concatenate((signalDF.iloc[0]['PPG'],
                                                                 signalDF.iloc[1]['PPG'],
                                                                 signalDF.iloc[2]['PPG'],
                                                                 signalDF.iloc[3]['PPG']), axis=0),
                                                 np.concatenate((signalDF.iloc[0]['PPG_GDF'],
                                                                 signalDF.iloc[1]['PPG_GDF'],
                                                                 signalDF.iloc[2]['PPG_GDF'],
                                                                 signalDF.iloc[3]['PPG_GDF']), axis=0)]],
                                               columns=columns)

            # ------------------------------------------------------------------------------------------------
            signalDF = pd.DataFrame([[empty_np, empty_np, empty_np, empty_np, empty_np]], columns=columns)


        if int(subject_to_work['Experiment']) > 2:
            #print((chunk_size))
            for s in range(floor((subject_to_work['Time'].size) / chunk_size)):  # Iteration for each each chunk of data
                newDF2 = pd.DataFrame([[subject_to_work['ID'].values[0],
                                        subject_to_work['Experiment'].values[0],
                                        subject_to_work['Time'][0][(s) * chunk_size:(s + 1) * chunk_size],
                                        subject_to_work['PPG'][0][(s) * chunk_size:(s + 1) * chunk_size],
                                        subject_to_work['PPG_GDF'][0][(s) * chunk_size:(s + 1) * chunk_size]]],
                                      columns=columns)
                newDF = newDF.append(newDF2)
                #print(newDF.shape)
        else:
            for s in range(floor(subject_to_work['Time'].size / chunk_size)):  # Iteration for each each chunk of data
                newDF2 = pd.DataFrame([[subject_to_work['ID'],
                                        subject_to_work['Experiment'],
                                        subject_to_work['Time'][(s) * chunk_size:(s + 1) * chunk_size],
                                        subject_to_work['PPG'][(s) * chunk_size:(s + 1) * chunk_size],
                                        subject_to_work['PPG_GDF'][(s) * chunk_size:(s + 1) * chunk_size]]],
                                      columns=columns)
                newDF = newDF.append(newDF2)
            i+=1
    #print(newDF.shape)
    #df = newDF[1:]
    #df = df.reset_index(drop=True)
    num_subjects = df.iloc[:-1]['ID']
    #print(len(num_subjects))

#########################################################################################
   # ----------------- Split data in Train/Test ------------------ #
    df.insert(1, "Split", "")
    segments = df['ID'].tolist()
    for k in range(0, len(num_subjects+1)):
        num_segments.append(segments.count(k))

    cont_validation = 1
    previous_id = df.iloc[0]['ID']
    cont_test = 0

    for i, row in df.iterrows():

        if previous_id != df.iloc[i]['ID']:
            previous_id = df.iloc[i]['ID']
            cont_validation = 1
            cont_test = 0

        if df.iloc[i]['ID'] == target:
            if df.iloc[i]['Experiment'] == 1:
                df.loc[i, 'Split'] = 'Test'
                cont_test += 1
            elif df.iloc[i]['Experiment'] == 2:
                df.loc[i, 'Split'] = 'TestIDs'
                cont_test += 1
            elif cont_validation <= round((num_segments[int(df.iloc[i]['ID'])]-cont_test) * percentage_validation):
                df.loc[i, 'Split'] = 'Validation'
                cont_validation += 1
            else:
                df.loc[i, 'Split'] = 'Train'

        elif df.iloc[i]['ID'] <= nusr_train or df.iloc[i]['ID'] == 99:
            if df.iloc[i]['Experiment'] == 1 or df.iloc[i]['Experiment'] == 2:
                df.loc[i, 'Split'] = 'Test'
                cont_test += 1
            elif cont_validation <= round((num_segments[int(df.iloc[i]['ID'])]-cont_test) * percentage_validation):
                df.loc[i, 'Split'] = 'Validation'
                cont_validation += 1
            else:
                df.loc[i, 'Split'] = 'Train'

        elif df.iloc[i]['ID'] > nusr_train and df.iloc[i]['ID'] < 99:
            df.loc[i, 'Split'] = 'TestIDs'

        elif df.iloc[i]['ID'] > nusr_train and df.iloc[i]['ID'] > 100:
            df.loc[i, 'Split'] = 'Zscore'

    # df.to_csv('Splitted_DF_PulseID.csv', sep='\t', encoding='utf-8')  # Export DataFrame to .csv
    return df, num_subjects, num_segments
