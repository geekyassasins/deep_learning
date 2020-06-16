import os
import scipy.io
from scipy import ndimage
import numpy as np
import pandas as pd

# ----------------------- Declarations ------------------------ #
path = '' # DataBase .mat directory
lstFiles = []
lst = []
fs = 125
chunk_size = fs * 10  # 10 second chunk size
ppg1_zscore = []
ppg2_zscore = []
ppg1_gdf = []

# ------- Get all .mat directories inside one folder ---------- #
for root, dirs, files in os.walk(path, topdown=True):
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        if (extension == ".mat"):
            lstFiles.append(os.path.join(root, fichero))  # lstFiles cointains every .mat full path
lstFiles.sort()

# ----------------- Creating Panda Dataframe ------------------ #
df = pd.DataFrame(columns=['Subject', 'ID', 'Time', 'PPG1', 'PPG2', 'PPG1_GDF', 'PPG2_GDF'])
totaltime = 0

for i in np.arange(0, len(lstFiles)):
    time_stamp = []
    parts = lstFiles[i].split('/')  # parts is a vector which every position is one folder level of the path
    id_num = parts[-1].split('_')
    id_num = int(
        id_num[1].lstrip('S'))
    mat = scipy.io.loadmat(lstFiles[i])  #
    mat_data = mat['sig']  # label of PPG signal data in .mat
    ppg1 = np.array(mat_data[1, :])
    ppg2 = np.array(mat_data[2, :])
    time1 = len(ppg1) / fs
    for j in range(len(ppg1)):
        time_stamp.append(j / fs)

    time_np = np.array(time_stamp)
    totaltime += time1
    print('measure', i, '->', 'ppg1 size:', len(ppg1), '->', int(time1 // 60), 'min', round(time1 % 60, 2), 'sec')
    print("Partial Time (s):", time1)
    print("Total Time (s):", totaltime)

    # ------------------ Gaussian filtering-------------------- #
    ppg1_gdf = ndimage.gaussian_filter1d(ppg1, sigma=1, order=0)
    ppg2_gdf = ndimage.gaussian_filter1d(ppg2, sigma=1, order=0)

    df.loc[i] = [parts[-1], id_num, time_np, ppg1, ppg2, ppg1_gdf, ppg2_gdf]  # Add line to the DataFrame

print('--------------------------------------------')
print("TOTAL TIME:", int(totaltime // 60), 'min', round(totaltime % 60, 2), 'sec')

# ------------------ Z-score normalization -------------------- #
for row in df.itertuples():

    ppg1_col = row[4]  # CHANGE IF NECESSARY
    ppg2_col = row[5]  # CHANGE IF NECESSARY
    ppg1_mean = np.mean(ppg1_col)  # z-score normalization
    ppg1_std = np.std(ppg1_col)
    zscore1 = np.array((ppg1_col - ppg1_mean) / ppg1_std)
    ppg1_zscore.append(zscore1)

    ppg2_mean = np.mean(ppg2_col)
    ppg2_std = np.std(ppg2_col)
    zscore2 = np.array((ppg2_col - ppg1_mean) / ppg1_std)
    ppg2_zscore.append(zscore2)

df.loc[:, 'PPG1_zscore'] = ppg1_zscore
df.loc[:, 'PPG2_zscore'] = ppg2_zscore

print(df)
df.to_csv('DataFrame_TROIKA.csv', sep='\t', encoding='utf-8')	# Export DataFrame to .csv
df.to_pickle('DataFrame_TROIKA.pkl')  # Export DataFrame to pickle

# -------------------------- Plots ---------------------------- #
# subject = 3  # subject to plot
# plt.plot(df.iloc[subject]['Time'], df.iloc[subject]['PPG1'],'C0', label='PPG1', linewidth=0.5)
# plt.plot(df.iloc[subject]['Time'], df.iloc[subject]['PPG2'],'C1', label='PPG2', linewidth=0.5)
# plt.plot(df.iloc[subject]['Time'], df.iloc[subject]['PPG1_zscore'],'C2', label='PPG1_zscore', linewidth=0.5)
# plt.plot(df.iloc[subject]['Time'], df.iloc[subject]['PPG2_zscore'],'C3', label='PPG2_zscore', linewidth=0.5)
# plt.plot(df.iloc[subject]['Time'], df.iloc[subject]['PPG1_GDF'], 'C4', label='PPG1_filtered gdf', linewidth=0.5)
# plt.xlabel('Time [sec]')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.show()

#
# PPG_dataset_TROIKA
#
# ECG	-> mat_data[0,:]
# PPG1-> mat_data[1,:]
# PPG2-> mat_data[2,:]
# X-Accel-> mat_data[3,:]
# Y-Accel-> mat_data[4,:]
# Z-Accel-> mat_data[5,:]
