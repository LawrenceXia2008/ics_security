## MCCED

This repo shows how the whole anomaly detection system works including data preprocessing, model training, model test (anomaly detection) and data visualization.

### Installation
```
conda install keras=2.2.0
conda install tensorflow-gpu=1.8.0
pip install Bokeh
pip install numpy=1.14.3
pip install scipy=1.1.0
pip install scikit-learn=0.19.1
```
### Data Preprocessing


We recommend that the folders look like :
```
<base_dir/MCCED>
                |</dataset/>
                  |--swat_train
                  |--swat_test
                  |--wadi_train
                  |--wadi_test
                |</seconddata/>
                |</resultdata/>
                |</code/>
                  |--PreprocessingSWAT.py
                  ...

```
We used `SWaT` dataset and `WADI` dataset. `normtogether.txt` is the setting file. Begin preprocessing data :
<br>`SWaT:`
```
python PreprocessingSWAT.py --settings_file normtogether
```
`WADI:`
```
python PreprocessingWADI.py --settings_file normtogether
```
The results of preprocessing will be saved in `<seconddata>` folder.


### Training
<b>If you would like to train the MCCED model by yourself (`train.txt` is the setting file) :</b>
```
python Memory_enhanced_Composite_Encoder_Decoder.py --settings_file train
```
The trained model weights will be saved in `<resultdata>` folder.

<b>If you would like to use weights which we have got by training the model:</b>
 <br>you can download weights directly in test stage from the `<resultdata/>` fold where we provide a whole training result.



 ### Anomaly detection
 Firstly, you should change the weights path in `universal_find_best_in_trained_models.py`:
 ```
 model_path="../resultdata/11_17_21_24conditional_results/models"
 ```
 The path should be changed baesed on your own cases.

 <br>Secondly, begin test:
 ```
 python universal_find_best_in_trained_models.py --settings_file train
 ```
 It will search the best model from ten models based on the best F1 score. Meanwhile, it will select the threshold by grid search.

 <br>Lastly, the F1 score, recall, precision of the best model with the right threshold will be printed in the screen.


 ### Data visualization
Firstly, you should change the prediction results file path in `cum.py` file.
```
result=np.load("the_best_results_of_test_10_25_00_45.npz")['result'][()]
```
In the anomaly detection stage, the model will save the prediction results of test dataset in npz file in the `code` folder for data visualization.

<br>Secondly, Set the time period you want to visualize:
```
start_t1 = '2015-12-28 10:02:00 AM'
end_t1 = '2016-1-2 2:59:59 PM'
```
<br>Lastly:

<b>Draw the prediction curves of a specific sensor:</b>
<br> Run `single_sensor_global_analysis` function in `cum.py, conditional_testSWAT.py` or `conditional_testWADI.py`'

<b>Draw error curves:</b>
<br> Run `运行local_error_for_outlier` function in `cum.py, conditional_testSWAT.py` or `conditional_testWADI.py`'

```
python conditional_testSWAT.py
```








