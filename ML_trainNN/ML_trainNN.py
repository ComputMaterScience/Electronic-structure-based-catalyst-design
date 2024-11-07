# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# for multiprocessing jobs
import time
from timeit import default_timer as timer
from multiprocessing import Pool, cpu_count

# for tensorflow
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from keras.layers import Dense
from keras.layers import Activation
from keras.models import Sequential
from keras.utils.vis_utils import pydot
from keras.utils.vis_utils import plot_model

# for scikit-learn
import shap
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Program verion
app_ver = 'v1.7 - 01/11/2021'

# Default Variables
num_core = cpu_count()
num_nn = list()
ratio_train = 0.8
num_run = 1
num_epochs = 500
learn_rate = 0.001
decay_rate = 0.95
decay_step = 200
num_hidden_layer = 1
act_func = list()
loss_func = 'mae'

job_makeinput = True
job_select_feature = False
job_train = True
job_predict = True
restart = False

# Feature Selection
plot_pair = False
plot_comat = False
use_back_elim = False
use_recu_elim = False
use_lasso = False
use_randomforest = False
use_permutation = False
use_shap = False

# system
use_gpu = False

input_names = list()
output_names = list()


######################## IO functions #################################

def get_filenames(folder):
    # get filenames in database folder
    list_files = list()
    for path in os.listdir(folder):
        full_path = os.path.join(folder, path)
        if os.path.isfile(full_path):
            list_files.append(full_path)
    return list_files
        
def combineData(input,output):
    # combine csv files into one csv file
    list_files = get_filenames(input)
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in list_files ])
    #export to csv
    combined_csv.to_csv(output, index=False, encoding='utf-8-sig')

def sort_min_data(input, output, col_name1, col_name2):
    # sort and return only minimum of col_name1 based on col_name2
    data = pd.read_csv(input)
    # get unique data for each col_name2
    unique_value = pd.unique(data[col_name2])
    unique_index = list()
    for i in range(len(unique_value)):
        unique_index.append(data.index[data[col_name2] == unique_value[i]].tolist())
    # find data minimum in col_name1
    list_min = list()
    for i in range(len(unique_value)):
        ldata = data.loc[unique_index[i],col_name1]
        lmin = ldata == np.min(ldata)
        list_min.append(unique_index[i][np.where(lmin)[0][0]])
    # filter data
    data = data.iloc[list_min,:]
    #data = data.drop(columns=col_name1)
    #data = data.drop(columns=col_name2)
    data.to_csv(output, index=False, encoding='utf-8-sig')

def remove_unique_col(input,output,col_remove=[]):
    # make scaled database
    data = pd.read_csv(input)
    col_names = list(data.columns)
    num_col = len(col_names)
    # get unique data for each columm
    unique_data = list()
    for i in range(num_col):
        unique_data.append(pd.unique(data.iloc[:,i]))
    for i in range(num_col):
        if len(unique_data[i]) == 1:
            data = data.drop(columns=col_names[i])
    for i in range(len(col_remove)):
        data = data.drop(columns=col_remove[i])
    data.to_csv(output, index=False, encoding='utf-8-sig')

def prepare_data(path, ratio_train):
    data = pd.read_csv(path)
    data = data[input_names+output_names]
    col_names = list(data.columns)
    num_col = len(col_names)
    # Drop rows containing unknown values
    data = data.dropna()
    # get unique data for each columm
    unique_cnt = -1
    obj2num_dict = {}
    dataTypeSeries = data.dtypes
    for i in range(num_col):
        if dataTypeSeries[i] == 'object':
            unique_data = pd.unique(data.iloc[:,i])
            dict_value = {}
            for j in range(len(unique_data)):
                dict_value[unique_data[j]] = j
            obj2num_dict[col_names[i]] = dict_value
    with open('obj2num.json','w') as obj2num: # save unique data to json file
        json.dump(obj2num_dict, obj2num)
    # convert object to number for database
    unique_cnt = -1
    for i in range(num_col):
        if dataTypeSeries[i] == 'object':
            unique_cnt += 1
            data[col_names[i]] = data[col_names[i]].map(obj2num_dict[col_names[i]])
    # split the data into train and test
    train_dataset = data.sample(frac=ratio_train, random_state=0)
    test_dataset = data.drop(train_dataset.index)
    # print overall statistics of dataset
    train_dataset.describe().transpose().to_csv('overall_statistics.csv')
    # save train and test datasets
    train_dataset.to_csv('train_dataset.csv', index=False, encoding='utf-8-sig')
    test_dataset.to_csv('test_dataset.csv', index=False, encoding='utf-8-sig')

def convert_data_from_obj2num(path, reverse=False):
    
    ## function for swap dictionary elements
    #def swap_dict(old_dict):
    #    new_dict = {}
    #    for key, value in old_dict.items():
    #        if value in new_dict:
    #            new_dict[value].append(key)
    #        else:
    #            new_dict[value]=[key]
    #    return new_dict

    # read data for prediction
    data = pd.read_csv(path)
    data = data[input_names]
    col_names = list(data.columns)
    num_col = len(col_names)
    # Drop rows containing unknown values
    data = data.dropna()
    # Opening obj2num JSON file
    f = open('obj2num.json','r')
    obj2num_dict = json.load(f)
    for i in obj2num_dict.keys():
        data[i] = data[i].map(obj2num_dict[i])
    data.to_csv('indexed_predict_dataset.csv', index=False, encoding='utf-8-sig')

######################## NN functions #################################

def plot_loss(history, index, output_names): # plot loss figure
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('Error [{0}]'.format(output_names[0]))
    plt.legend()
    plt.grid(True)
    plt.savefig('{0}/loss.png'.format(index),dpi=150, bbox_inches='tight')

def plot_true_vs_test(test_labels, test_predictions, index, output_names): # plot loss figure
    plt.figure()
    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [{0}]'.format(output_names[0]))
    plt.ylabel('Predictions [{0}]'.format(output_names[0]))
    lims = [np.min(test_labels), np.max(test_labels)]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.savefig('{0}/true_vs_test.png'.format(index),dpi=150, bbox_inches='tight')

# make training model
def build_and_compile_model(norm, num_nn, learn_rate):
    #model = keras.Sequential([norm,
    #        layers.Dense(40, activation='relu'),
    #        layers.Dense(10, activation='relu'),
    #        layers.Dense(1, activation='linear')])
    #opt = tf.keras.optimizers.Adam(learn_rate)
    #model.compile(loss=loss_func, 
    #              optimizer=opt,
    #              metrics=['mae'])

    model = keras.Sequential()
    model.add(norm)
    for i in range(num_hidden_layer):
        model.add(layers.Dense(num_nn[i], activation=act_func[i]))
    model.add(layers.Dense(1))
    opt = tf.keras.optimizers.Adam(learn_rate)
    #opt = tf.keras.optimizers.SGD(learn_rate)
    model.compile(loss=loss_func, 
                  optimizer=opt,
                  metrics=['mae'])
    #plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    return model

def train(index):

    #------------------------------
    #make new folder
    os.makedirs(str(index), exist_ok=True)
    #------------------------------
    #this block enables gpu enabled multiprocessing
    if (use_gpu == True):
        core_config = tf.compat.v1.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=core_config)
        keras.backend.set_session(session)
    #-------------------------------
    #to limit cpu cores in sub program
    """
    keras.backend.set_session(
        keras.backend.tf.Session(
            config=keras.backend.tf.ConfigProto(
            intra_op_parallelism_threads=3
            , inter_op_parallelism_threads=3)
            )
    )
    """
    #-------------------------------
    ##prepare input and output values

    train_dataset = pd.read_csv('train_dataset.csv')
    test_dataset = pd.read_csv('test_dataset.csv')

    train_features = train_dataset.copy()
    test_features = test_dataset.copy()

    train_labels = train_features.pop(output_names[0])
    test_labels = test_features.pop(output_names[0])

    # Data normalization

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    #------------------------------
       
    dnn_model = build_and_compile_model(normalizer, num_nn, learn_rate)
    print(dnn_model.summary())

    #------------------------------
    if (restart):
        if (os.path.isdir('{0}'.format(index))):
            dnn_model.load_weights('{0}/weights.ckpt'.format(index))
            print('Restart NN Training !!')

    # save checkpoint
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath='{0}/weights.ckpt'.format(index),
                                                    save_weights_only=True,
                                                    verbose=1)

    # early stop when there is no improvement
    cp_earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.01, patience=50,
                                                    verbose=0, mode='auto', baseline=0.2, 
                                                    restore_best_weights=False)

    # automatically reduce the learning rate after steps

    # This function reduce the learning rate by x% after each 200 steps
    def scheduler(epoch, lr):
        decay_rate = 0.95
        decay_step = 200
        if epoch % decay_step == 0 and epoch:
            return lr * decay_rate
        return lr

    cp_reduce_learnrate = tf.keras.callbacks.LearningRateScheduler(scheduler)

    history = dnn_model.fit(train_features, train_labels,
                            validation_split=0.2,
                            verbose=1, epochs=num_epochs, 
                            callbacks=[cp_callback,cp_reduce_learnrate],
                            batch_size=int(0.1*len(train_features.index)),shuffle=True)

    plot_loss(history, index, output_names)
    #------------------------------

    test_results = dnn_model.evaluate(test_features, test_labels, verbose=1)

    with open('{0}/loss_sets.txt'.format(index), 'w') as fd:
        fd.write("Loss: {0}, Val_loss: {1}, Test_loss: {2}".format(history.history['loss'][-1],
                                                                    history.history['val_loss'][-1],
                                                                    test_results))

    test_predictions = dnn_model.predict(test_features).flatten()
    plot_true_vs_test(test_labels, test_predictions, index, output_names)

    #finally, close sessions
    if (use_gpu == True):
        session.close()
        keras.backend.clear_session()
    return 0

def predict(index):

    #------------------------------
    #this block enables gpu enabled multiprocessing
    if (use_gpu == True):
        core_config = tf.compat.v1.ConfigProto()
        core_config.gpu_options.allow_growth = True
        session = tf.compat.v1.Session(config=core_config)
        keras.backend.set_session(session)
    #-------------------------------
    #to limit cpu cores in sub program
    """
    keras.backend.set_session(
        keras.backend.tf.Session(
            config=keras.backend.tf.ConfigProto(
            intra_op_parallelism_threads=3
            , inter_op_parallelism_threads=3)
            )
    )
    """
    #-------------------------------
    ##prepare input and output values

    train_dataset = pd.read_csv('train_dataset.csv')

    train_features = train_dataset.copy()

    train_labels = train_features.pop(output_names[0])

    predict_dataset = pd.read_csv('indexed_predict_dataset.csv')

    predict_features = predict_dataset[input_names]

    # Data normalization

    normalizer = preprocessing.Normalization(axis=-1)
    normalizer.adapt(np.array(train_features))

    #------------------------------
       
    dnn_model = build_and_compile_model(normalizer, num_nn, learn_rate)
    print(dnn_model.summary())

    #------------------------------

    dnn_model.load_weights('{0}/weights.ckpt'.format(index)).expect_partial()

    predict_dataset = pd.read_csv('predict_dataset.csv')

    predict_value = dnn_model.predict(predict_features).flatten()
    predict_dataset[output_names] = predict_value

    predict_dataset.to_csv('{0}/predict_results.csv'.format(0), index=False, encoding='utf-8-sig')

    #finally, close sessions
    if (use_gpu == True):
        session.close()
        keras.backend.clear_session()
    return 0

###################### Feature Selection Functions ##############################
def select_feature():
    train_dataset = pd.read_csv('train_dataset.csv')
    feature_names = input_names
    col_names = list(train_dataset.columns)
    # pairplot
    if (plot_pair):
        print(f'- Generate pairplot - joint distribution')
        #sns_plot = sns.pairplot(train_dataset[col_names],diag_kind='kde')
        sns_plot = sns.pairplot(train_dataset[col_names],height=2.0)
        sns_plot.savefig("pairplot.png",dpi=150, bbox_inches='tight')
    # covariance matrix
    if (plot_comat):
        print(f'- Calculate and plot covariance matrix')
        stdsc = StandardScaler() 
        X_std = stdsc.fit_transform(train_dataset[col_names].iloc[:,range(0,len(col_names))].values)
        cov_mat =np.cov(X_std.T)
        plt.figure()
        sns.set(font_scale=1.5)
        hm = sns.heatmap(cov_mat,
                         cbar=True,
                         annot=True,
                         square=True,
                         fmt='.2f',
                         annot_kws={'size': 12},
                         cmap='coolwarm',                 
                         yticklabels=col_names,
                         xticklabels=col_names,
                         vmin = -1, vmax = 1)
        plt.title('Covariance matrix showing correlation coefficients', size = 18)
        plt.tight_layout()
        plt.savefig('comatplot.png',dpi=150, bbox_inches='tight')
        plt.rcParams.update(plt.rcParamsDefault)
    # recursive feature elimination
    if (use_recu_elim):
        print(f'- Use recursive feature elimination method')
        #no of features
        nof_list=np.arange(1,len(feature_names))            
        high_score=0
        #Variable to store the optimum features
        nof=0           
        score_list =[]
        for n in range(len(nof_list)):
            X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names],
                                                                train_dataset[output_names], 
                                                                test_size = 0.3, random_state = 0)
            model = LinearRegression()
            rfe = RFE(model,n_features_to_select=nof_list[n])
            X_train_rfe = rfe.fit_transform(X_train,y_train)
            X_test_rfe = rfe.transform(X_test)
            model.fit(X_train_rfe,y_train)
            score = model.score(X_test_rfe,y_test)
            score_list.append(score)
            if(score>high_score):
                high_score = score
                nof = nof_list[n]
        print("Optimum number of features: %d" %nof)
        print("Score with %d features: %f" % (nof, high_score))
        cols = feature_names
        model = LinearRegression()
        #Initializing RFE model
        rfe = RFE(model, n_features_to_select=nof)             
        #Transforming data using RFE
        X_rfe = rfe.fit_transform(train_dataset[feature_names],train_dataset[output_names])  
        #Fitting the data to model
        model.fit(X_rfe,train_dataset[output_names])              
        temp = pd.Series(rfe.support_,index = cols)
        selected_features_rfe = temp[temp==True].index
        print(f'Optimum number of features:',selected_features_rfe)
    #embedded method - LassoCV
    if (use_lasso):
        print(f'- Use embedded method - LassoCV')
        reg = LassoCV()
        reg.fit(train_dataset[feature_names], np.ravel(train_dataset[output_names]))
        print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
        print("Best score using built-in LassoCV: %f" %reg.score(train_dataset[feature_names],train_dataset[output_names]))
        coef = pd.Series(reg.coef_, index = feature_names)
        print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")
        imp_coef = coef.sort_values()
        plt.figure()
        imp_coef.plot(kind = "barh")
        plt.title("Feature importance using Lasso Model")
        plt.savefig('LassoCV.png',dpi=150, bbox_inches='tight')
    #Random Forest method
    if (use_randomforest):
        print(f'- Use Random Forest method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.25, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        sorted_idx = rf.feature_importances_.argsort()
        plt.figure()
        plt.barh(np.array(feature_names)[sorted_idx], rf.feature_importances_[sorted_idx])
        plt.savefig('randomforest.png',dpi=150, bbox_inches='tight')
    #Permutation Importance method
    if (use_permutation):
        print(f'- Use Permutation Importance method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.25, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        perm_importance = permutation_importance(rf, X_test, y_test)
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.figure()
        plt.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.savefig('permutation.png',dpi=150, bbox_inches='tight')
    #SHAP Values method
    if (use_shap):
        print(f'- Use SHAP Values method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.25, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, max_display=len(feature_names), plot_type="bar",show=False)
        plt.savefig('shap_values.png',dpi=150, bbox_inches='tight')
        plt.figure()
        shap.summary_plot(shap_values, X_test,show=False)
        plt.savefig('shap_values_detail.png',dpi=150, bbox_inches='tight')
    # backward elimination
    if (use_back_elim):
        print(f'- Use backward elimination method')
        cols = feature_names
        pmax = 1
        while (len(cols)>0):
            p= []
            X_1 = train_dataset[cols]
            X_1 = sm.add_constant(X_1)
            model = sm.OLS(train_dataset[output_names],X_1).fit()
            p = pd.Series(model.pvalues.values[1:],index = cols)      
            pmax = max(p)
            feature_with_p_max = p.idxmax()
            if(pmax>0.05):
                cols.remove(feature_with_p_max)
            else:
                break
        selected_features_BE = cols
        print(selected_features_BE)


######################## Main Program #################################

def read_input():
    global num_core, num_nn, ratio_train, num_run, num_epochs
    global num_hidden_layer, act_func, loss_func, learn_rate
    global job_makeinput, job_select_feature, job_train, job_predict
    global input_names, output_names, use_gpu, restart
    global plot_pair, plot_comat, use_back_elim, use_recu_elim, use_lasso, use_randomforest
    global use_permutation, use_shap
    with open('INPUT','r') as fin:
        for line in fin:
            if (line.find('num_core') != -1): # get number of cpu core
                s = line.split('='); num_core = int(s[1])
            elif (line.find('num_nn') != -1): # get number of neural network in hidden layer
                s = line.split('='); s = s[1].split(',')
                for i in s:
                    num_nn.append(int(i))
            elif (line.find('ratio_train') != -1): # get ratio of train data
                s = line.split('='); ratio_train = float(s[1])
            elif (line.find('num_run') != -1): # get number of training loop
                s = line.split('='); num_run = int(s[1])
            elif (line.find('num_epochs') != -1): # get maximum number of epochs for each NN train 
                s = line.split('='); num_epochs = int(s[1])
            elif (line.find('num_hidden_layer') != -1): # get number of hidden layers 
                s = line.split('='); num_hidden_layer = int(s[1])
            elif (line.find('learn_rate') != -1): # get learning rate
                s = line.split('='); learn_rate = float(s[1])
            elif (line.find('decay_rate ') != -1): # get decay rate
                s = line.split('='); decay_rate  = float(s[1])
            elif (line.find('decay_step') != -1): # get decay step
                s = line.split('='); decay_step = float(s[1])
            elif (line.find('act_func') != -1): # get activation function
                s = line.split('='); s = s[1].split(',')
                for i in s:
                    act_func.append(i.strip())
            elif (line.find('loss_func') != -1): # get loss function
                s = line.split('='); loss_func = s[1].strip()
            elif (line.find('job_makeinput') != -1): # run makeinput or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_makeinput = True
                else:
                    job_makeinput = False
            elif (line.find('job_select_feature') != -1): # run job_select_feature or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_select_feature = True
                else:
                    job_select_feature = False
            elif (line.find('job_train') != -1): # run train_NN or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_train = True
                else:
                    job_train = False
            elif (line.find('job_predict') != -1): # run predict_NN or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_predict = True
                else:
                    job_predict = False
            elif (line.find('restart') != -1): # restart or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    restart = True
                else:
                    restart = False
            elif (line.find('plot_pair') != -1): # plot joint distribution of training data
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    plot_pair = True
                else:
                    plot_pair = False
            elif (line.find('plot_comat') != -1): # plot covariance matrix of training data
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    plot_comat = True
                else:
                    plot_comat = False
            elif (line.find('use_back_elim') != -1): # using Backward Elimination for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_back_elim = True
                else:
                    use_back_elim = False
            elif (line.find('use_recu_elim') != -1): # using Recursive Feature Elimination for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_recu_elim = True
                else:
                    use_recu_elim = False
            elif (line.find('use_lasso') != -1): # using Embedded Method - LassoCV for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_lasso = True
                else:
                    use_lasso = False
            elif (line.find('use_randomforest') != -1): # using Random Forest method for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_randomforest = True
                else:
                    use_randomforest = False
            elif (line.find('use_permutation') != -1): # using Permutation Importance method for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_permutation = True
                else:
                    use_permutation = False
            elif (line.find('use_shap') != -1): # using SHAP method for feature selection
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_shap = True
                else:
                    use_shap = False
            elif (line.find('input_names') != -1): # get input names 
                s = line.split('='); s = s[1].split()
                for i in s:
                    input_names.append(i)
            elif (line.find('output_names') != -1): # get output names 
                s = line.split('='); s = s[1].split()
                for i in s:
                    output_names.append(i)
            elif (line.find('use_gpu') != -1): # use use_gpu or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    use_gpu = True
                else:
                    use_gpu = False
    if (len(input_names) == 0 | len(output_names) == 0 | len(num_nn) == 0 | len(num_nn) != len(act_func)  | len(act_func) == 0):
        return 1
    else:
        return 0

def print_input():
    print(f'Input parameters:')
    print(f'num_core             = ', num_core)
    print(f'num_nn               = ', num_nn)
    print(f'num_hidden_layer     = ', num_hidden_layer)
    print(f'act_func             = ', act_func)
    print(f'loss_func            = ', loss_func)
    print(f'ratio_train          = ', ratio_train)
    print(f'num_run              = ', num_run)
    print(f'num_epochs           = ', num_epochs)
    print(f'learn_rate           = ', learn_rate)
    print(f'decay_rate           = ', decay_rate)
    print(f'decay_step           = ', decay_step)
    print(f'job_makeinput        = ', job_makeinput)
    print(f'job_select_feature   = ', job_select_feature)
    print(f'job_train            = ', job_train)
    print(f'job_predict          = ', job_predict)
    print(f'input_names          = ', input_names)
    print(f'output_names         = ', output_names)
    print(f'use_gpu              = ', use_gpu)
    print(f'restart              = ', restart)
    print(f'plot_pair            = ', plot_pair)
    print(f'plot_comat           = ', plot_comat)
    print(f'use_back_elim        = ', use_back_elim)
    print(f'use_recu_elim        = ', use_recu_elim)
    print(f'use_lasso            = ', use_lasso)
    print(f'use_randomforest     = ', use_randomforest)
    print(f'use_permutation      = ', use_permutation)
    print(f'use_shap             = ', use_shap)
    return 0

#-----------------------------
#main program
if __name__ == '__main__':
    # Start timer and read input parameters
    start0 = timer()
    error = read_input()
    if (error == 0):
        print(f'*************************************************************')
        print(f'            Starting computations on {num_core} cores')
        print(f'*************************************************************')
        print(f'                Version: {app_ver}')
        print(f'*************************************************************')
        print_input()
        print(f'*************************************************************')
        if (job_makeinput):
            print(f'Reading and processing Database')
            # Start timer
            start = timer()
            # read and process input data
            # 1. combine data
            combineData('databases','combined_data.csv')
            # 2. prepare data
            sort_min_data('combined_data.csv', 'combined_data.csv', 'tot_en', 'formula')
            prepare_data('combined_data.csv', ratio_train)
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print(f'*************************************************************')
        if (job_select_feature):
            print(f'Check and recommend features for training dataset')
            # Start timer
            start = timer()
            select_feature()
            # if we run select_feature(), we will stop and choose more suitable input_names
            ####################
            job_train = False
            job_predict = False
            ####################
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print(f'*************************************************************')
            print(f'Let\'s stop and choose input_names for the next job')
            print(f'*************************************************************')
        if (job_train):
            # Start timer
            start = timer()
            # set up NN
            print(f'Setting up Neural Network')
            for i in range(num_run):
                train(i)
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print(f'*************************************************************')
        if (job_predict):
            # Start timer
            start = timer()
            # set up NN
            print(f'Predict data from Neural Network')
            convert_data_from_obj2num('predict_dataset.csv')
            for i in range(num_run):
                predict(i)
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print(f'*************************************************************')
        print('Total time: %10.2f s' % (end-start0))
    else:
        print(f'Error in INPUT file')