import argparse
import json
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import scipy.io as scio
import random
# from keras.layers import Input, LSTM, Dense, ConvLSTM2D, BatchNormalization, Conv3D, TimeDistributed
from sklearn import metrics
import tensorflow as tf
# from sklearn import preprocessing
# import pandas as pd
from sklearn import metrics

# from keras.models import load_model
import conditional_test_add_pure_merge_interval_try

from keras.models import load_model
import re
import pdb

# GPU Limitation
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))

# My Implementation
import models


'''function'''

def NN_predict(model,dataX,dataY):
    dataX = dataX.reshape(dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1)
    dataY = dataY.reshape(dataY.shape[0], dataY.shape[1], dataY.shape[2], dataY.shape[3], 1)

    dataX_zeros=np.zeros((dataX.shape[0], dataX.shape[1], dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)
    dataX_decoder_zeros = np.zeros((dataX.shape[0], 1, dataX.shape[2], dataX.shape[3], 1))  # (smaple,1,20,51,1)

    if dataY.shape[1]==1:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_zeros,dataX_decoder_zeros], batch_size=32, verbose=2)  #
    else:
        [predict_label1, predict_label2] = model.predict([dataX, dataX_decoder_zeros], batch_size=32, verbose=2)  #
    predict_label1 = predict_label1[:, ::-1, :, :, :]  # 把时间轴取反

    data_predict_NN = predict_label2.reshape(
        predict_label2.shape[0] * predict_label2.shape[1] * predict_label2.shape[2], predict_label2.shape[3])
    data_rcstr_NN = predict_label1.reshape(predict_label1.shape[0] * predict_label1.shape[1] * predict_label1.shape[2],
                                           predict_label1.shape[3])
    data_predict_true = dataY.reshape(dataY.shape[0] * dataY.shape[1] * dataY.shape[2], dataY.shape[3])
    data_rcstr_true = dataX.reshape(dataX.shape[0] * dataX.shape[1] * dataX.shape[2], dataX.shape[3])

    return data_predict_NN, data_rcstr_NN, data_predict_true, data_rcstr_true

def options_parser():
    parser = argparse.ArgumentParser(description='Train a neural network to handle real-valued data.')
    # meta-option
    parser.add_argument('--settings_file', help='json file of settings, overrides everything else', type=str, default='')
    parser.add_argument('--pre_path', help='preprocessing_timepath', type=str, default='')
    parser.add_argument('--model', help='result_timepath', type=str, default='')
    return parser

def load_settings_from_file(settings):
    # settings可以是任何一个txt形式的字典文件
    settings_path = settings['settings_file'] + ".txt"
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r',encoding='utf-8'))
    # check for settings missing in file
    return settings_loaded



def get_settings_and_files():
    parser = options_parser()
    settings_raw = vars(parser.parse_args())

    if settings_raw['settings_file']:
        settings = load_settings_from_file(settings_raw)

    result_path = "../seconddata/" + settings_raw['pre_path'] + "ConvLstm预处理结果(conditional版本)/"
    reslist = os.listdir(result_path)  # 列出文件夹下所有的目录与文件
    total_result = []
    for i in range(0, len(reslist)):
        if reslist[i].endswith('.npz'):
            path = os.path.join(result_path, reslist[i])
        if os.path.isfile(path):
            total_result.append(np.load(path)['result'][()])
    choice_result = total_result[0]
    train_input = choice_result['train_input']
    train_predict = choice_result["train_predict"]
    test_input = choice_result["test_input"]
    test_predict = choice_result["test_predict"]
    ground_truth=choice_result["ground_truth"]
    params = settings
    network = models.__dict__[settings_raw['model']]()
    input_win=test_input.shape[1]*test_input.shape[2]
    return params, input_win, network, ground_truth, train_input, train_predict, test_input, test_predict



def sort_model_by_time(model_path):
    models = os.listdir(model_path)
    if not models:
        return
    else:
        files = sorted(models, key=lambda x: os.path.getmtime(os.path.join(model_path, x)))
        return files


def one_model_operation_conditional(encoder_model,past_decoder_model,fu_decoder_model,model_path):
    encoder_model.load_weights(model_path, by_name=True)
    past_decoder_model.load_weights(model_path, by_name=True)
    fu_decoder_model.load_weights(model_path, by_name=True)
    _,_, train_input, train_predict, test_input, test_predict = get_settings_and_files()
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = NN_predict(encoder_model,
                                                                                        past_decoder_model,
                                                                                        fu_decoder_model, train_input,
                                                                                        train_predict)
    test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = NN_predict(encoder_model, past_decoder_model,
                                                                                    fu_decoder_model, test_input,
                                                                                    test_predict)
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true

def one_model_operation_unconditional(model,model_path,params, train_input, train_predict,test_input,test_predict):
    # model_graph=tf.Graph()
    # model_graph = tf.Graph()
    # sess = tf.Session(graph=model_graph)
    # K.set_session(sess)
    # K.get_session().run(tf.global_variables_initializer())

    with tf.Graph().as_default() as model_graph:
        with tf.Session() as model_session:
            model = model(params,train_input, train_predict)
            print('********************  已经加载网络  ********************')
            model.load_weights(model_path, by_name=True)
            print('********************  加载网络了之后加载了权重  ********************')
            _,_,_,_, train_input, train_predict, test_input, test_predict = get_settings_and_files()
    # with model_graph.as_default():
    #     with model_session.as_default():
            print('********************  开始预测第一个  ********************')
            train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true = NN_predict(model,train_input,train_predict)
            test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true = NN_predict(model, test_input, test_predict)
            print('********************  已经全都预测完  ********************')
    model_session=None
    model_graph=None
    # model_session.close()
    # K.clear_session()
    return train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true,test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true


def test_in_one_model(models_path, fuse):
    params, input_win, model, ground_truth, train_input, train_predict, test_input, test_predict=get_settings_and_files()

    '''预测大循环'''
    ground_truth = ground_truth[:-input_win]
    # print('######ground_truth.shape######',ground_truth.shape)
    train_predict_NN, train_rcstr_NN, train_predict_true, train_rcstr_true, test_predict_NN, test_rcstr_NN, test_predict_true, test_rcstr_true= one_model_operation_unconditional(model,models_path,params,train_input, train_predict,test_input,test_predict)

    '''对齐时间窗'''
    train_predict_true=train_predict_true[:-input_win]
    train_predict_NN=train_predict_NN[:-input_win]
    test_predict_NN = test_predict_NN[:-input_win]
    test_predict_true = test_predict_true[:-input_win]

    train_rcstr_true=train_rcstr_true[input_win:]
    train_rcstr_NN=train_rcstr_NN[input_win:]
    test_rcstr_NN = test_rcstr_NN[input_win:]
    test_rcstr_true = test_rcstr_true[input_win:]

    # print('######test_rcstr_true.shape######',test_rcstr_true.shape)


    '''预测+重构'''
    if fuse == 'A':
        alpha = params['rcstr_weight']
        beta = params['predict_weight']
        train_predict_NN = beta * train_predict_NN + alpha * train_rcstr_NN
        train_predict_true = beta * train_predict_true + alpha * train_rcstr_true
        test_predict_NN = beta * test_predict_NN + alpha *test_rcstr_NN
        test_predict_true = beta * test_predict_true + alpha * test_rcstr_true
    elif fuse == 'B':
        pass


        # '''post processing'''
        #WADI 72个特征
        # for j in range(train_predict_NN.shape[1]):
        #     if (j == 0 or j == 1 or j == 2 or j == 5 or j == 16 or j == 17 or j == 18 or j == 19 or j == 20
        #         or j == 21 or j == 22 or j == 23 or j == 24 or j == 37 or j == 38 or j == 39 or j == 40
        #         or j == 41 or j == 42 or j == 43 or j == 43 or j == 45 or j == 59 or j == 61 or j == 63 or j == 64 or j == 71):
        #         continue
        #     else:
        #         train_predict_NN[:, j] = np.around(train_predict_NN[:, j])
        #         test_predict_NN[:, j] = np.around(test_predict_NN[:, j])
        # '''post processing'''

        # '''post processing'''
        # # SWAT 40个特征
        # for j in range(train_predict_NN.shape[1]):
        #     if (j == 0 or j == 1 or j == 5 or j == 13 or j == 14 or j == 21 or j == 22 or j == 23 or j == 29
        #         or j == 30 or j == 31 or j == 32 or j == 35 or j == 36):
        #         continue
        #     else:
        #         train_predict_NN[:, j] = np.around(train_predict_NN[:, j])
        #         test_predict_NN[:, j] = np.around(test_predict_NN[:, j])
        # '''post processing'''

    start_t1 = '2015-12-28 10:02:00 AM'
    end_t1 = '2016-1-2 2:59:59 PM'
    # start_t1 = '2017-10-9 06:02:00 PM'
    # end_t1 = '2017-10-11 06:00:00 PM'

    '''单点'''
    weight_or_not = True
    test_obj = conditional_test_add_pure_merge_interval_try.analysis(train_rcstr_true, train_rcstr_NN, test_rcstr_true, test_rcstr_NN, train_predict_true, train_predict_NN, test_predict_true, test_predict_NN,
     ground_truth, start_t1, end_t1, input_win, weight_or_not=weight_or_not,
                                                  weight_type=1,
                                                  norm_or_not=False,rcstr_weight=params['rcstr_weight'],predict_weight=params['predict_weight'],
                                                  fuse = 'A', specific="1221")  # 这里的False指的是是否使用归一化误差
    
    max_wgt_error_value=np.max(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma)
    near_max_wgt_error_value=np.percentile(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma,99.9)
    min_wgt_error_value=np.min(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma)
    near_min_wgt_error_value=np.percentile(test_obj.wgt_error_out_ewma if weight_or_not==True else test_obj.nwgt_error_out_ewma,0.1)
    print('max',max_wgt_error_value)
    print('min',min_wgt_error_value)
    print('near_min',near_min_wgt_error_value)
    print('near_max',near_max_wgt_error_value)
    
    period_specific='period2'
    roc_img_path=os.path.join(os.path.dirname(models_path),'{0}_{1}_roc曲线'.format('加权' if weight_or_not==True else '不加权',period_specific))
    my_dict = test_obj.threshold_grid_search_no_plotting(weight_or_not=weight_or_not, number=100, lower=near_min_wgt_error_value, upper=near_max_wgt_error_value,
                                                             consecutive_or_not=False, fine_or_not=True,roc_curve=False, roc_img_path=roc_img_path)
    print(my_dict)  # 输出每一个模型最好的结果
    # model_dict["模型"].append(my_dict["模型"])
    # model_dict["阈值"].append(my_dict["阈值"])
    # model_dict["最好的F1值"].append(my_dict["最好的F1值"])
    # model_dict["此时的Precision"].append(my_dict["此时的Precision"])
    # model_dict["此时的Recall"].append(my_dict["此时的Recall"])

    # '''event_baesd'''
    # test_obj1 = conditional_test_add_pure.analysis(train_predict_true, train_predict_NN, test_predict_true,
    #                                                   test_predict_NN,
    #                                                   ground_truth, start_t1, end_t1, 120, weight_or_not=True,
    #                                                   weight_type=1,
    #                                                   norm_or_not=False,
    #                                                   specific=lastk_models_name[i])  # 这里的False指的是是否使用归一化误差
    # my_dict1 = test_obj1.threshold_grid_search_no_plotting(weight_or_not=True, number=100, lower=0.01, upper=0.5,
    #                                                      consecutive_or_not=True, fine_or_not=True)
    # print(my_dict1) # 输出每一个模型最好的结果
    # model_dict1["模型"].append(my_dict1["模型"])
    # model_dict1["阈值"].append(my_dict1["阈值"])
    # model_dict1["最好的F1值"].append(my_dict1["最好的F1值"])
    # model_dict1["此时的Precision"].append(my_dict1["此时的Precision"])
    # model_dict1["此时的Recall"].append(my_dict1["此时的Recall"])

    '''输出预测误差'''
    # from sklearn import metrics
    # seq_anomaly = list(np.argwhere(ground_truth == 1).reshape(-1))
    # seq_norm = list(np.argwhere(ground_truth == 0).reshape(-1))
    # test_pure_anomaly_error = metrics.mean_squared_error(test_predict_NN[seq_anomaly, :],
    #                                                      test_predict_true[seq_anomaly, :])
    # test_pure_norm_error = metrics.mean_squared_error(test_predict_NN[seq_norm, :],
    #                                                   test_predict_true[seq_norm, :])
    # my_dict2 = {
    #     '测试集中纯正常的预测误差': test_pure_norm_error,
    #     '测试集中纯异常的预测误差': test_pure_anomaly_error,
    #     '测试集中异常与正常的平均预测误差gap': test_pure_anomaly_error - test_pure_norm_error
    # }
    # print(my_dict2)



    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<单点>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # # 累计多个模型中最好的F1值
    # f1max = max(model_dict["最好的F1值"])
    # best = np.argwhere(model_dict["最好的F1值"] == f1max)[0][0]
    # # 全部结果输出

    # for _ in range(lastk):
    #     my_dict = {"模型": model_dict["模型"][_], "阈值": model_dict["阈值"][_],"最好的F1值": model_dict["最好的F1值"][_],
    #                "此时的Precision": model_dict["此时的Precision"][_], "此时的Recall": model_dict["此时的Recall"][_]}
    #     print(my_dict)
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<best>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("最好的情况" + str({"模型": model_dict["模型"][best], "阈值": model_dict["阈值"][best],"最好的F1值": model_dict["最好的F1值"][best],
    #                      "此时的Precision": model_dict["此时的Precision"][best], "此时的Recall": model_dict["此时的Recall"][best]}))


    # print("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<event-based>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # # 累计多个模型中最好的F1值
    # f1max1 = max(model_dict1["最好的F1值"])
    # best1 = np.argwhere(model_dict1["最好的F1值"] == f1max1)[0][0]
    # # 全部结果输出
    #
    # for _ in range(lastk):
    #     my_dict1 = {"模型": model_dict1["模型"][_], "阈值": model_dict1["阈值"][_],"最好的F1值": model_dict1["最好的F1值"][_],
    #                "此时的Precision": model_dict1["此时的Precision"][_], "此时的Recall": model_dict1["此时的Recall"][_]}
    #     print(my_dict1)
    # print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<best>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # print("最好的情况" + str({"模型": model_dict1["模型"][best1], "阈值": model_dict1["阈值"][best1],"最好的F1值": model_dict1["最好的F1值"][best1],
    #                      "此时的Precision": model_dict1["此时的Precision"][best1], "此时的Recall": model_dict1["此时的Recall"][best1]}))
    #

    # print("\n\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<预测误差>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    # for _ in range(lastk):
    #     my_dict2 = {"测试集中纯正常的预测误差": model_dict2["测试集中纯正常的预测误差"][_],
    #                "测试集中纯异常的预测误差": model_dict2["测试集中纯异常的预测误差"][_],
    #                 "测试集中异常与正常的平均预测误差gap": model_dict2["测试集中异常与正常的平均预测误差gap"][_]}
    #     print(my_dict2)


if __name__ == '__main__':
    # Instruction:
    # 1.根据不同情况有不同的backwards_num 如果picture正常画出，backwards_num=2，没有=1
    # 2.换了不同的网络要换不同的模型 只改动import和as之间就可以
    # 3.teacher_forcing_or_not
    # pdb.set_trace()
    # model_path="../resultdata/11_15_14_33conditional训练结果(conditional版本)/模型"
    # 无下采样
    # model_path="../resultdata/05_16_17_09conditional训练结果(conditional版本)/模型/05_16_17_09model-ep126-loss0.15498-val_loss0.12952.h5"
    # 有下采样
    
    '''
        Fuse：
        - Type: 
        - 表示融合的方式，可选的选项有：
            -- 'A' 同MCCED详细文档.docx所述，表示先将重构和预测相加再求误差
            -- 'B' 同MCCED详细文档.docx所述，表示重构和预测先各自求误差，再相加
            -- 'predict' 只使用预测
            -- 'reconstruct' 只使用重构
    '''
    model_path="../resultdata/05_16_21_39conditional训练结果(conditional版本)/模型/05_16_21_39model-ep148-loss0.31598-val_loss0.17372.h5"
    test_in_one_model(models_path = model_path, fuse = 'C')


