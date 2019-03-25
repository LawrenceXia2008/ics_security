# coding=UTF-8
import pandas as pd
import numpy as np
import datetime
from bokeh.plotting import figure, output_file, show
from bokeh.io import export_png
from sklearn.metrics import average_precision_score, recall_score, precision_score, f1_score
def ewma_vectorized(data, alpha, offset=None, dtype='float64', order='C', out=None):
    """
    Calculates the exponential moving average over a vector.
    Will fail for large inputs.
    :param data: Input data
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param offset: optional
        The offset for the moving average, scalar. Defaults to data[0].
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the input. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if data.ndim > 1:
        # flatten input
        data = data.reshape(-1, order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if offset is None:
        offset = data[0]

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # scaling_factors -> 0 as len(data) gets large
    # this leads to divide-by-zeros below
    row_size = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create cumulative sum array
    np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                dtype=dtype, out=out)
    np.cumsum(out, dtype=dtype, out=out)

    # cumsums / scaling
    out /= scaling_factors[-2::-1]

    if offset != 0:
        offset = np.array(offset, copy=False).astype(dtype, copy=False)
        # add offsets
        out += offset * scaling_factors[1:]

    return out
def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype='float64', order='C', out=None):
    """
    Calculates the exponential moving average over a given axis.
    :param data: Input data, must be 1D or 2D array.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param axis: The axis to apply the moving average on.
        If axis==None, the data is flattened.
    :param offset: optional
        The offset for the moving average. Must be scalar or a
        vector with one element for each row of data. If set to None,
        defaults to the first value of each row.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Ignored if axis is not None.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    """
    data = np.array(data, copy=False)

    assert data.ndim <= 2

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float64
    else:
        dtype = np.dtype(dtype)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    if data.size < 1:
        # empty input, return empty array
        return out

    if axis is None or data.ndim < 2:
        # use 1D version
        if isinstance(offset, np.ndarray):
            offset = offset[0]
        return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                               out=out)

    assert -data.ndim <= axis < data.ndim

    # create reshaped data views
    out_view = out
    if axis < 0:
        axis = data.ndim - int(axis)

    if axis == 0:
        # transpose data views so columns are treated as rows
        data = data.T
        out_view = out_view.T

    if offset is None:
        # use the first element of each row as the offset
        offset = np.copy(data[:, 0])
    elif np.size(offset) == 1:
        offset = np.reshape(offset, (1,))

    alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

    # calculate the moving average
    row_size = data.shape[1]
    row_n = data.shape[0]
    scaling_factors = np.power(1. - alpha, np.arange(row_size + 1, dtype=dtype),
                               dtype=dtype)
    # create a scaled cumulative sum array
    np.multiply(
        data,
        np.multiply(alpha * scaling_factors[-2], np.ones((row_n, 1), dtype=dtype),
                    dtype=dtype)
        / scaling_factors[np.newaxis, :-1],
        dtype=dtype, out=out_view
    )
    np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
    out_view /= scaling_factors[np.newaxis, -2::-1]

    if not (np.size(offset) == 1 and offset == 0):
        offset = offset.astype(dtype, copy=False)
        # add the offsets to the scaled cumulative sums
        out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

    return out

def get_max_row_size(alpha, dtype=float):
    assert 0. <= alpha < 1.
    # This will return the maximum row size possible on
    # your platform for the given dtype. I can find no impact on accuracy
    # at this value on my machine.
    # Might not be the optimal value for speed, which is hard to predict
    # due to numpy's optimizations
    # Use np.finfo(dtype).eps if you  are worried about accuracy
    # and want to be extra safe.
    epsilon = np.finfo(dtype).tiny
    # If this produces an OverflowError, make epsilon larger
    return int(np.log(epsilon)/np.log(1-alpha)) + 1


def window_size(alpha, sum_proportion):
    # solve (1-alpha)**window_size = (1-sum_proportion) for window_size
    return int(np.log(1-sum_proportion) / np.log(1-alpha))


def ewma_vectorized_safe(data, alpha, row_size=None, dtype='float64', order='C', out=None):
    """
    Reshapes data before calculating EWMA, then iterates once over the rows
    to calculate the offset without precision issues
    :param data: Input data, will be flattened.
    :param alpha: scalar float in range (0,1)
        The alpha parameter for the moving average.
    :param row_size: int, optional
        The row size to use in the computation. High row sizes need higher precision,
        low values will impact performance. The optimal value depends on the
        platform and the alpha being used. Higher alpha values require lower
        row size. Default depends on dtype.
    :param dtype: optional
        Data type used for calculations. Defaults to float64 unless
        data.dtype is float32, then it will use float32.
    :param order: {'C', 'F', 'A'}, optional
        Order to use when flattening the data. Defaults to 'C'.
    :param out: ndarray, or None, optional
        A location into which the result is stored. If provided, it must have
        the same shape as the desired output. If not provided or `None`,
        a freshly-allocated array is returned.
    :return: The flattened result.
    """
    data = np.array(data, copy=False)

    if dtype is None:
        if data.dtype == np.float32:
            dtype = np.float32
        else:
            dtype = np.float
    else:
        dtype = np.dtype(dtype)

    row_size = int(row_size) if row_size is not None else get_max_row_size(alpha, dtype)

    if data.size <= row_size:
        # The normal function can handle this input, use that
        return ewma_vectorized(data, alpha, dtype=dtype, order=order, out=out)

    if data.ndim > 1:
        # flatten input
        data = np.reshape(data, -1, order=order)

    if out is None:
        out = np.empty_like(data, dtype=dtype)
    else:
        assert out.shape == data.shape
        assert out.dtype == dtype

    row_n = int(data.size // row_size)  # the number of rows to use
    trailing_n = int(data.size % row_size)  # the amount of data leftover
    first_offset = data[0]

    if trailing_n > 0:
        # set temporary results to slice view of out parameter
        out_main_view = np.reshape(out[:-trailing_n], (row_n, row_size))
        data_main_view = np.reshape(data[:-trailing_n], (row_n, row_size))
    else:
        out_main_view = out
        data_main_view = data

    # get all the scaled cumulative sums with 0 offset
    ewma_vectorized_2d(data_main_view, alpha, axis=1, offset=0, dtype=dtype,
                       order='C', out=out_main_view)

    scaling_factors = (1 - alpha) ** np.arange(1, row_size + 1)
    last_scaling_factor = scaling_factors[-1]

    # create offset array
    offsets = np.empty(out_main_view.shape[0], dtype=dtype)
    offsets[0] = first_offset
    # iteratively calculate offset for each row
    for i in range(1, out_main_view.shape[0]):
        offsets[i] = offsets[i - 1] * last_scaling_factor + out_main_view[i - 1, -1]

    # add the offsets to the result
    out_main_view += offsets[:, np.newaxis] * scaling_factors[np.newaxis, :]

    if trailing_n > 0:
        # process trailing data in the 2nd slice of the out parameter
        ewma_vectorized(data[-trailing_n:], alpha, offset=out_main_view[-1, -1],
                        dtype=dtype, order='C', out=out[-trailing_n:])
    return out

def p_power_error(y_true, y_pred):
    return np.mean(np.power(y_pred - y_true, 6), axis=-1)

# 保证进入deep_processing之前数据concatenate为2维（point_id,dim）
# 顺序是先进行取范数再进行指数滑动平均
def deep_processing(y_true, y_pred,_train_,predict_win,power):
    print("正字进行p次方误差深度处理，准备计算异常数值")
    window = predict_win
    sum_proportion =.5
    # 取自TEP论文中的half-life标准，因此我们取为
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    powered_error=np.mean(np.power(y_pred - y_true, power), axis=-1)
    # print("此时alpha数值为"+alpha)
    if _train_==1:
        # threshold=np.percentile(powered_error,99)
        threshold=np.percentile(ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype=None, order='C', out=None),99)
        return threshold
    else:
        return ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None)

def deep_processing_abs(y_true, y_pred,_train_,predict_win):
    print("正在进行L1范数误差深度处理，准备绘制异常曲线")
    window = predict_win
    sum_proportion =.5
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    powered_error=np.mean(abs(y_pred - y_true), axis=-1)
    if _train_==1:
        threshold=np.percentile(ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None),99)
        return threshold
    else:
        return ewma_vectorized_safe(powered_error, alpha, row_size=None, dtype='float64', order='C', out=None)

def get_anomaly(train_true, train_predict,test_true,test_predict,predict_win):
    print("正在尝试捕捉异常")
    scores2=deep_processing(test_true, test_predict, 0,predict_win,6)
    threshold=deep_processing(train_true,train_predict, 1,predict_win,6)
    return zip(scores2 >= (40000000*threshold), scores2)


def report_evaluation_metrics(y_true, y_pred):
    average_precision = average_precision_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    recall = recall_score(y_true, y_pred, labels=[0, 1], pos_label=1)
    f1 = f1_score(y_true, y_pred, labels=[0, 1], pos_label=1)

    print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    print('Precision: {0:0.2f}'.format(precision))
    print('Recall: {0:0.2f}'.format(recall))
    print('F1: {0:0.2f}'.format(f1))

def metrics_and_diagnosis(train_true, train_predict,test_true,test_predict, ground_truth,input_win):
    prediction_error=[]
    pred_label=[]
    anomaly_information= get_anomaly(train_true, train_predict,test_true,test_predict,input_win)
    for idx, (is_anomaly, dist) in enumerate(anomaly_information):
        predicted_label = 1 if is_anomaly else 0
        pred_label.append(predicted_label)
        prediction_error.append(dist)
    prediction_error=np.array((prediction_error),np.float64)
    # pred_label=np.array(pred_label)
    # print(ground_truth[input_win:].shape)
    # print(len(pred_label))
    # print(list(ground_truth[input_win:]))
    # print(pred_label)
    report_evaluation_metrics(list(ground_truth[input_win:]),pred_label)
    return prediction_error,pred_label

def experiment_result_save(train_true, train_predict,test_true,test_predict,ground_truth,input_win):

    threshold=deep_processing_abs(train_true, train_predict, 1, input_win)
    error_conse_ploted=deep_processing_abs(test_true,test_predict, 0, input_win)
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')  # 现在
    time_result =nowTime + r"experiment_result.npz"
    np.savez(time_result,train_predict_true=train_true, train_predict_by_mach=train_predict,test_predict_true=test_true,test_predict_by_mach=test_predict,threshold=np.array([threshold]),error_conse_plotted=error_conse_ploted,ground_truth=ground_truth[input_win:])
    print("4种row data、经过计算的连续误差以及经过计算的阈值已存储")
# def hybrid_evaluation(ground_truth,pred_label):
def error_consecutive_plot(train_true, train_predict,test_true,test_predict,ground_truth_plotted,input_win):
    threshold_plotted=deep_processing_abs(train_true, train_predict, 1, input_win)
    error_conse_plotted=deep_processing_abs(test_true,test_predict, 0, input_win)
    datetime1 = pd.date_range('20151228 10:00:00', periods=ground_truth_plotted[input_win:].shape[0], freq='S')
    # datetime1 = pd.date_range('20151228 10:00:00', periods=, freq='S')
    # datetime2=np.datetime64(datetime1)
    # df_mae.set_index(datetime1,inplace=True)
    nowTime = datetime.datetime.now().strftime('%Y_%m_%d_%H-%M-%S')  # 现在
    output_file(nowTime + r"Consecutive_error_plot.html")
    p = figure(plot_width=1000, plot_height=800, x_axis_type="datetime")
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Average_Error over 51 variables'
    # p.ygrid.band_fill_color = "olive"
    # p.ygrid.band_fill_alpha = 0.1
    p.circle(datetime1, ground_truth_plotted, size=10, color='darkgrey', alpha=0.5, legend='Real_value')
    p.line(datetime1, error_conse_plotted, line_width=2, color='forestgreen', legend='Trend')
    p.line(datetime1, threshold_plotted[0], legend="Threshold", line_width=3, color='firebrick')
    p.legend.location = "top_left"
    export_png(p, filename=nowTime+r"plot.png")



if __name__=='__main__':
    mae_of_predictions=np.load("mae_of_predictions.npy")
    output_file("kankan.html")
    p = figure(plot_width=1000, plot_height=800, x_axis_type="datetime")
    p.xaxis.axis_label = 'Time'
    p.yaxis.axis_label = 'Average_Error over 51 variables'
    datetime1 = pd.date_range('20190211 12:50:00', periods=975, freq='S')
    df_mae = pd.DataFrame(mae_of_predictions, index=datetime1)
    # p.ygrid.band_fill_color = "olive"
    # p.ygrid.band_fill_alpha = 0.1
    '''window,alpha定义'''
    window = 2
    sum_proportion =.5
    alpha = 1 - np.exp(np.log(1 - sum_proportion) / window)
    # 让sum_proportation==0.5是看到的论文里用到的，然后让windowd==predic_window
    ''''''
    _avg=ewma_vectorized_safe(np.mean(mae_of_predictions,axis=1), alpha, row_size=None, dtype='float64', order='C', out=None)
    # 不采用2D
    p.circle(datetime1,np.mean(mae_of_predictions,axis=1), size=10, color='darkgrey', alpha=0.5, legend='Real_value')
    p.line(datetime1, _avg,line_width=2,color='forestgreen', legend='Trend')
    p.line(datetime1,1.40752929449,legend="Threshold",line_width=3,color='firebrick')
    p.legend.location = "top_left"
    # show(p)
    a=np.array([[1,2,3],[4,5,6],[7,8,9],[1,2,3],[4,5,6],[7,8,9]])
    b=np.array([[2,3,4],[5,6,7],[8,9,10],[1,2,3],[4,5,6],[7,8,9]])
    c=a
    d=b
    e=np.array([1,0,1,0,0,1,0,1])
    metrics_and_diagnosis(a,b,c,d,e,2)