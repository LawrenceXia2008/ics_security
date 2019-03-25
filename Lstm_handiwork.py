import pandas as pd
import numpy as np
import datetime
from prettytable import PrettyTable
import os
from keras.layers import RepeatVector
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from sklearn import preprocessing
from bokeh.plotting import figure, output_file, show
# from keras.utils import plot_model
# from bokeh.io import export_svgs
import matplotlib.pyplot as plt
# from Code \
import LSTM_Settings_load,metric_utils
import json
import argparse
#####################################################  读入数据  ##########################################################
dataset_train_station="E:\\Myfiles\\Self\\PRJ\\NEWPRJ\\SECPRJ\\Dataset&code\\SWaT_Dataset_Normal_v0.csv"
"""../dataset/SWaT_Dataset_Normal_v0.csv"""

df_train=pd.read_csv(dataset_train_station,sep=',',header=1)
dataset_test_station="E:\\Myfiles\\Self\\PRJ\\NEWPRJ\\SECPRJ\\Dataset&code\\SWaT_Dataset_Attack_v0.csv"
"""../dataset/SWaT_Dataset_Attack_v0.csv"""
df_test=pd.read_csv(dataset_test_station,sep=',',header=1)
#####################################################  初始化  ##########################################################
df_mat_train=df_train.as_matrix(columns=None)
df_mat_train=np.array(df_mat_train[16000:,1:52],dtype=np.float64)
df_mat_test=df_test.as_matrix(columns=None)
df_mat_test[:,-1]=np.where(df_mat_test[:,-1]=="Normal",0,1)
global Ground_truth
Ground_truth=df_mat_test[:,-1]
df_mat_test=np.array(df_mat_test[:,1:52],dtype=np.float64)
###################################################  数据归一化  ########################################################
def Z_ScoreNorm(column, _train_, train_scaler):
    if _train_ == 1:
        #     if np.std(column)==0:
        #         return [(1.0/(1 + np.exp(column[row]))) for row in range(column.shape[0])]
        #     else:
        return preprocessing.scale(column)
    else:
        return np.array(train_scaler.transform(column.reshape(-1, 1))).reshape(-1)
def dataset_norm(dataset, _train_, train_scaler):
    if _train_ == 1:
        print("正在进行训练集归一化")
        scaler = []
        for i in range(dataset.shape[1]):
            scaler.append(preprocessing.StandardScaler().fit(dataset[:, i].reshape(-1, 1)))
        return np.array([Z_ScoreNorm(dataset[:, column], 1, []) for column in range(dataset.shape[1])],
                        np.float64).T, scaler
    else:
        print("正在进行测试集归一化")
        return np.array(
            [Z_ScoreNorm(dataset[:, column], 0, train_scaler[column]) for column in range(dataset.shape[1])],np.float64).T
def dataset_norm_minmax(train_set,test_set):
    print("正在使用最大最小值归一化方法对训练集和测试集进行归一化")
    train_set1=[]
    test_set1=[]
    for i in range(train_set.shape[1]):
        min_max_scaler = preprocessing.MinMaxScaler()
        c=min_max_scaler.fit_transform(train_set[:, i].reshape(-1,1)).reshape(-1)
        d=min_max_scaler.transform(test_set[:,i].reshape(-1,1)).reshape(-1)
        train_set1.append(c)
        test_set1.append(d)
    return np.array(train_set1,dtype=np.float64).T,np.array(test_set1,dtype=np.float64).T

########################################################################################################################
def dataset_extraction(dataset,sample_size,input_win,predict_win,step,sensor_id0,sensor_ide):
    row=dataset.shape[0]
    # overlap_rate=0.5
    # step=int((1-overlap_rate)*sample_size)
    sample_num=(row-sample_size)//step+1
    row_modified=sample_num*sample_size
    horizon_win=sample_size-input_win-predict_win
    tmp_result=dataset
    '''sensor_time_dim∈[1,52]，[1,51]是传感器变量，将时间戳作为最后第52column'''
    time_stamp=np.array([i for i in range(0,row)],np.float64)[:,np.newaxis]
    tmp_result=np.concatenate((tmp_result,time_stamp),axis=1)
    sensor_scope=[sensor_id for sensor_id in range(sensor_id0-1,sensor_ide)]
    sensor_scope.extend([-1])
    sensor_time_dim=len(sensor_scope)
    '''sample的overlap滚动'''
    sample_generation=[]
    def gen_sample(total_array):
        for start, stop in zip(range(0, row-sample_size+1,step), range(sample_size, row+1,step)):
            yield total_array[start:stop,:]
    for sample_arr in gen_sample(tmp_result):
        sample_generation.extend(sample_arr)
    sample_generation=np.array(sample_generation,np.float64)

    sample_divided_mat=[]
    for sample_id in range(0, sample_num):
        sample_divided_mat.extend(sample_generation[sample_id * sample_size:(sample_id + 1)*sample_size,sensor_scope][np.newaxis,:])
    sample_divided_mat=np.array(sample_divided_mat)
    # (67,1000,52)
    # 得到预测部分:
    predict_generation=sample_divided_mat[:,-predict_win:,0:len(sensor_scope)]
    # 得到输入部分：
    input_generation=sample_divided_mat[:,:input_win,0:len(sensor_scope)]
    # 得到输入序列shape：

    return input_generation,predict_generation


def dataset_skip_norm_orno(mat_train,mat_test,__test_norm_with_train__):
    '''
    :param 对于settings['test_norm_with_train']参数:
    :param 为1时我们使用训练集参数归一化测试集，为0时我们使用测试集自己的参数归一化测试集:
    :return:
    '''

    if __test_norm_with_train__==True:
        normed_train_set = "../LSTM_Experiments/normed_data_with_train.npz"
        if os.path.exists(normed_train_set):
            print("Find cache file %s" % normed_train_set)
            c=np.load('../LSTM_Experiments/normed_data_with_train.npz')
            mat_train = c['mat_train']
            mat_test = c['mat_test']
        else:
            mat_train,mat_test=dataset_norm_minmax(mat_train,mat_test)
            # mat_train, train_scaler1 = dataset_norm(mat_train, 1, [])
            # mat_test = dataset_norm(mat_test, 0, train_scaler1)
            np.savez("../LSTM_Experiments/normed_data_with_train.npz",mat_train=mat_train,mat_test=mat_test)
        print("归一化结束")
        return mat_train,mat_test
    else:
        normed_train_set = "../LSTM_Experiments/normed_data_together.npz"
        if os.path.exists(normed_train_set):
            print("Find cache file %s" % normed_train_set)
            c = np.load('../LSTM_Experiments/normed_data_together.npz')
            mat_train = c['mat_train']
            mat_test = c['mat_test']
        else:
            aa=np.append(mat_train,mat_test, axis=0)
            aa, aa = dataset_norm_minmax(aa, aa)
            mat_train=aa[:mat_train.shape[0],:]
            mat_test=aa[mat_train.shape[0]:,:]
            print(mat_train.shape)
            print(mat_test.shape)
            # mat_train, train_scaler1 = dataset_norm(mat_train, 1, [])
            # mat_test = dataset_norm(mat_test, 0, train_scaler1)
            np.savez("../LSTM_Experiments/normed_data_together.npz", mat_train=mat_train, mat_test=mat_test)
        print("归一化结束")
        return mat_train, mat_test


def getPara():
    print("请按照如下格式输入预处理参数6个："+"\n请以空格隔开并在输入最后一个数字后两次回车"+"\n(sample_size,input_win,predict_win,step,sensor_id0,sensor_ide)")
    res = []
    inputLine = input()  # 以字符串的形式读入一行
# 如果不为空字符串作后续读入
    while inputLine != '':
        listLine = inputLine.split(' ')  # 以空格划分就是序列的形式了
        listLine = [int(e) for e in listLine]  # 将序列里的数由字符串变为int类型
        res.extend(listLine)
        inputLine = input()
    return res

def main(df_mat_train,df_mat_test,sample_size,input_win,predict_win,step,sensor_id0,sensor_ide,__test_norm_with_train__,pathc):
    train_data, test_data=dataset_skip_norm_orno(df_mat_train,df_mat_test,__test_norm_with_train__)
    train_input,train_predict=dataset_extraction(train_data,sample_size,input_win,predict_win,step,sensor_id0,sensor_ide)
    test_input,test_predict=dataset_extraction(test_data,sample_size,input_win,predict_win,step,sensor_id0,sensor_ide)
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')  # 现在
    time_result = pathc+r"/"+nowTime + r"LSTM_pp_result.npz"
    dianshu_tr = sample_size + (train_input.shape[0] - 1) * step
    dianshu_te = sample_size + (test_input.shape[0] - 1) * step

    ground_truth=Ground_truth[:dianshu_te]
    np.savez(time_result, train_input=train_input, train_predict= train_predict,test_input=test_input, test_predict= test_predict,ground_truth=ground_truth)


    return dianshu_tr,dianshu_te,sample_size,input_win,predict_win,step,time_result,train_input.shape[0],test_input.shape[0],sensor_id0,sensor_ide


if __name__ == '__main__':
    '''输入提示：(df_mat_train,df_mat_test,sample_size,input_win,predict_win,step,sensor_id0,sensor_ide)'''
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')  # 现在
    parser = LSTM_Settings_load.options_parser()
    settings=vars(parser.parse_args())

    if settings['settings_file']:
        settings = LSTM_Settings_load.load_settings_from_file(settings)
    # windows系统不同
    pathc="../LSTM_Experiments/"+r"_"+nowTime+r"_/"+r"LSTM_Cache"
    pathm="../LSTM_Experiments/" + r"_" + nowTime + r"_/" + r"LSTM_Model"
    pathp="../LSTM_Experiments/" + r"_" + nowTime + r"_/" + r"LSTM_Plot"
    os.makedirs(pathc)
    os.makedirs(pathm)
    os.makedirs(pathp)
    dianshu_tr,dianshu_te,sample_size, input_win, predict_win, step, time_result, train_sample_num,test_sample_num, sensor_id0, sensor_ide\
        =main(df_mat_train,df_mat_test,settings['sample_size'],settings['input_win'],settings['predict_win'],settings['step'],settings['sensor_id0'],settings['sensor_ide'],settings['test_norm_with_train'],pathc)
    x = PrettyTable()
    x.field_names = ["SAVE INFO:"+time_result, "time point number", "sample number", "horizon","input length", "predict length", "step length", "sensor start", "sensor end"]
    x.add_row(["training set", dianshu_tr,train_sample_num, sample_size-input_win-predict_win,input_win,predict_win, step, sensor_id0, sensor_ide])
    x.add_row(["test set", dianshu_te, test_sample_num, sample_size-input_win-predict_win,input_win,predict_win,step, sensor_id0, sensor_ide])
    print('预处理结束')
    print(x)
    ###############################################################   LSTM部分   ######################################################################
    tmp=np.load(time_result)
    train_input=tmp["train_input"][:,:,0:sensor_ide - sensor_id0 + 1]
    train_predict=tmp["train_predict"][:,:,0:sensor_ide - sensor_id0 + 1]
    test_input=tmp["test_input"][:,:,0:sensor_ide - sensor_id0 + 1]
    test_predict=tmp["test_predict"][:,:,0:sensor_ide - sensor_id0 + 1]
    ground_truth=tmp["ground_truth"]

    input_shape_0=input_win
    input_shape_1=sensor_ide-sensor_id0+1
    hidden_dim=settings['hidden_units']
    select_only_last_state = True  # Ture or False  Ture的时候loss会出现NAN，可能原因有很多，百度
    # m = Sequential()
    # if select_only_last_state:
    #     m.add(LSTM(hidden_dim, input_shape=(input_shape_0, input_shape_1), return_sequences=False))
    #     m.add(RepeatVector(input_shape_0))
    # else:
    #     m.add(LSTM(hidden_dim, input_shape=(input_shape_0, input_shape_1), return_sequences=True))
    # m.add(Dropout(p=0.1))
    # m.add(LSTM(input_shape_1, return_sequences=True, activation='linear'))
    '''模型设计'''
    model = Sequential()
    model.add(LSTM(hidden_dim, input_shape=(input_shape_0, input_shape_1),return_sequences=True))
    model.add(LSTM(hidden_dim,return_sequences=True))
    model.add(LSTM(hidden_dim))
    model.add(Dropout(0.5))
    model.add(RepeatVector(input_shape_0))
    model.add(LSTM(hidden_dim, return_sequences=True))
    model.add(LSTM(hidden_dim, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(input_shape_1, return_sequences=True, activation='linear'))
    model.add(TimeDistributed(Dense(input_shape_1)))
    model.compile(loss='mse', optimizer='rmsprop')
    filepath = pathm+r'/'+nowTime+'model-ep{epoch:03d}-loss{loss:.5f}-val_loss{val_loss:.5f}.h5'
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    history=model.fit(train_input,train_predict, batch_size=settings['batch_size'], nb_epoch=settings['nb_epochs'],callbacks=[checkpoint], validation_split=0.2)
    '''模型设计'''

    fig = plt.figure()  # 新建一张图
    fig = plt.figure()
    plt.plot(history.history['loss'], label='training loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(loc='upper right')
    fig.savefig(pathm+r'/'+'LSTM_' + nowTime + '_loss.png')

    # plot_model(model,to_file='../LSTM_Model/'+nowTime+'.png')
    ###############################################################   模型评估部分   ######################################################################
    test_predict_by_mach1 = model.predict(test_input)
    train_predict_by_mach1=model.predict(train_input)
    time_result_plot = pathp+r'/'+ nowTime + r"LSTM_等待画图.npz"
    np.savez(time_result_plot,test_predict_by_mach=test_predict_by_mach1,train_predict_by_mach=test_predict_by_mach1)
    print("正在进行训练集和测试集预测,预测结果已经存为"+time_result_plot)
    # 将得到的训练和测试的预测部分进行串联
    train_predict_by_mach=np.concatenate(train_predict_by_mach1, axis=0)
    test_predict_by_mach= np.concatenate(test_predict_by_mach1, axis=0)
    train_predict_true=np.concatenate(train_predict, axis=0)
    test_predict_true=np.concatenate(test_predict, axis=0)
    np.savez(time_result_plot, test_predict_by_mach=test_predict_by_mach, train_predict_by_mach=train_predict_by_mach,train_predict_true=train_predict_true,test_predict_true=test_predict_true)
    print("正在进行训练集和测试集预测,预测结果已经存为" + time_result_plot)
    print("各种shape")
    print([train_predict_by_mach.shape,test_predict_by_mach.shape,train_predict_true.shape,test_predict_true.shape,ground_truth.shape])
    test_error_distribution=metric_utils.metrics_and_diagnosis(train_predict_true, train_predict_by_mach,test_predict_true,test_predict_by_mach, ground_truth,input_win)
    # 本地存储异常信息
    # Code.metric_utils.error_consecutive_save(train_predict_true, train_predict_by_mach,test_predict_true,test_predict_by_mach, ground_truth,input_win)
    # 本地绘制异常曲线图
    # Code.metric_utils.error_consecutive_plot(train_predict_true, train_predict_by_mach, test_predict_true,test_predict_by_mach, ground_truth, input_win)
    ###############################################################   绘图检验部分   ######################################################################
    ###############################################################   文件记录部分   ######################################################################



    ###############################################################   文件记录部分   ######################################################################
