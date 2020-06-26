import math
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio
from scipy import signal
from scipy.signal import butter, lfilter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV

def compute_power_spectral_density(windowed_signal):
    ret,db_list = [],[]
    windowed_signal = list(map(list, zip(* windowed_signal)))
    PSD_FREQ = np.array([[4, 8]])
    # Welch parameters
    sliding_window = 7500
    SAMPLING_FREQUENCY = 250
    overlap = 0.667
    n_overlap = int(sliding_window * overlap)
    # compute psd using Welch method
    freqs, power = signal.welch(windowed_signal, fs=SAMPLING_FREQUENCY,
                                nperseg=sliding_window, noverlap=n_overlap)
    for psd_freq in PSD_FREQ:
        tmp = (freqs >= psd_freq[0]) & (freqs < psd_freq[1])
        a_ndarray=power[:, tmp].mean(1)
        h_list=a_ndarray.tolist()
        for i in h_list:
            db_list.append(10*math.log(i,10))
        ret.append(h_list)
    return ret

def get_trainxy(datapath,idspath):
    data = scio.loadmat(datapath)
    idsdata = scio.loadmat(idspath)
    train_x,train_y,filtedData,delete_channels=[],[],[],[]
    for i in range(30):
        b, a = signal.butter(8, [0.008, 0.4], 'bandpass')
        onechannel = signal.filtfilt(b, a, data['eeg_data'][i]) #1-50Hz滤波
        filtedData.append(onechannel)
    filtedData_t = list(map(list, zip(* filtedData)))  #100000*30
    for j in range(len(idsdata['idsNaN'])):
        x = filtedData_t[j * 2500:j * 2500 + 7500]
        res = compute_power_spectral_density(x)
        if idsdata['idsNaN'][j]==0 :
            train_x.append(res[0])
    for i in range(len(data['resTime'])):
            train_y.append([data['resTime'][i][0]])
    x = StandardScaler().fit_transform(train_x)
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(x)
    train_x = principalComponents.tolist()
    return train_x, train_y


def get_testx(datapath,idspath):
    data = scio.loadmat(datapath)
    idsdata = scio.loadmat(idspath)
    train_x,train_y,filtedData,delete_channels=[],[],[],[]
    for i in range(30):
        b, a = signal.butter(8, [0.008, 0.4], 'bandpass')
        onechannel = signal.filtfilt(b, a, data['eeg_data'][i]) #1-50Hz滤波
        filtedData.append(onechannel)
    filtedData_t = list(map(list, zip(* filtedData)))  #100000*30
    for j in range(len(idsdata['idsNaN'])):
        x = filtedData_t[j * 2500:j * 2500 + 7500]
        res = compute_power_spectral_density(x)
        if idsdata['idsNaN'][j]==0 :
            train_x.append(res[0])
    x = StandardScaler().fit_transform(train_x)
    pca = PCA(n_components=6)
    principalComponents = pca.fit_transform(x)
    train_x = principalComponents.tolist()
    return train_x

X_train, Y_train, res = [], [], []
test_idspath='D:\\学习学习学习\\人机交互\\idsNaN\\070207.mat'
test_datapath='D:\\学习学习学习\\人机交互\\【打包下载】天翼云盘\\test_data\\070207.mat'
idspath = 'D:\\学习学习学习\\人机交互\\idsNaN\\'
datapath = 'D:\\学习学习学习\\人机交互\\【打包下载】天翼云盘\\train_data\\'
#filenames = ['061031.mat']
filenames = ['061031.mat', '061101.mat', '061102.mat', '061130.mat', '070102.mat', '070105.mat', '070117.mat','060227.mat', '060308.mat', '060706.mat', '060707.mat', '060710.mat', '060711.mat', '060725.mat']
for i in filenames:
    curx,cury=get_trainxy(datapath+i,idspath+i)
    X_train.extend(curx)
    Y_train.extend(cury)
X_test=get_testx(test_datapath,test_idspath)

model = LassoCV()
model.fit(X_train,Y_train)
predict = model.predict(X_test)
#clf = linear_model.LinearRegression()
#clf.fit(X_train,Y_train)
#predict = clf.predict(X_test)
#res.extend([x[0] for x in predict])
name=['resTime']
result = pd.DataFrame(columns=name,data=predict)
result.to_csv('submission15.csv')

