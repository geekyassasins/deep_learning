import os
from scipy import signal, ndimage
import numpy as np
import pandas as pd

# --------------------- Initilalizations ---------------------- #
path = '' # DataBase .mat directory
lstFiles = []
lst = []
fitxers_test = []
fitxers = []
experiment = []
fs = 200
chunk_size = fs*10 #10 second chunk size
ppg_zscore = []
ppg_gdf = []

# ------- Get all .txt directories inside one folder ---------- #
for root, dirs, files in os.walk(path , topdown=True):
    for fichero in files:
        (nombreFichero, extension) = os.path.splitext(fichero)
        if(extension == ".txt"):
            lstFiles.append(os.path.join(root, fichero))		# lstFiles cointains every .txt full path
lstFiles.sort()

#  Creating Panda Dataframe  #
df = pd.DataFrame(columns=['Subject', 'ID', 'Experiment', 'Time', 'PPG', 'PPG_GDF'])

expermiments2use = ['1.txt', '2.txt', '3.txt', '4.txt', '5.txt', '7.txt']

for i in range(0,len(lstFiles)):
    parts = lstFiles[i].split('/')
    if parts[-2] == 'Pulse':
        experiment = parts[-1].split('_')
        if experiment[-1] in expermiments2use:			
            fitxers.append(lstFiles[i])


### CREATION OF training DATAFRAME ###
for i in np.arange(0, len(fitxers)):
    time_stamp = []
    parts = fitxers[i].split('/')								# parts is a vector which every position is one folder level of the path
    filename = parts[-1].split('_')
    if filename[0].lstrip('S') != 'Noisy':
        id_num = int(filename[0].lstrip('S'))
    exps = filename[-1].split('.')
    ppg_signal = np.loadtxt(fitxers[i], usecols=1)
    print('PPG_Signal', ppg_signal)
    time1 = len(ppg_signal)/fs
    for j in range(len(ppg_signal)):
        time_stamp.append(j/fs)

    time_np = np.array(time_stamp)
    print('measure', i,'->','ppg_signal size:',len(ppg_signal),'->',int(time1//60),'min',round(time1%60,2),'sec')

    # ------------------ Gaussian filtering-------------------- #
    ppg_gdf = ndimage.gaussian_filter1d(ppg_signal, sigma=1, order=0)

    df.loc[i] = [parts[-1], id_num, int(exps[0]), time_np, ppg_signal, ppg_gdf]		# Add line to the DataFrame



df.to_csv('DataFrame_PulseID.csv', sep='\t', encoding='utf-8')	# Export DataFrame to .csv
df.to_pickle('DataFrame_PulseID.pkl')	# Export DataFrame to pickle


