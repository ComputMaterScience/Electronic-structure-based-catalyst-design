# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# for timing
import time
from timeit import default_timer as timer

# pysubgroup
import pysubgroup as ps

# scikit-learn
from sklearn import linear_model, datasets
import shap
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)

# Program verion
app_ver = 'v1.0 - 06/11/2021'

# Job lists
job_select_feature = False
job_fit_normal_linear = False
job_subgroup = False

# Feature Selection
plot_pair = False
plot_comat = False
use_back_elim = False
use_recu_elim = False
use_lasso = False
use_randomforest = False
use_permutation = False
use_shap = False

# input and output lists
input_names = list()
output_names = list()
data_type = list()

######################      Fitting Functions      ##############################
def normal_linear_fit():
    colors = ['b', 'r', 'g', 'k', 'c', 'm', 'y']
    markers = ['o', 's', '^', 'd', '*', 'p', '>']
    dataset = pd.read_csv('data.csv')
    # Plot dataset
    plt.figure()
    if len(data_type) == 0:
        plt.scatter(dataset[input_names], dataset[output_names], c='r',s=40)
    else:
        # get unique data
        unique_value = pd.unique(dataset['data_type'])
        unique_index = list()
        for i in range(len(unique_value)):
            unique_index.append(dataset.index[dataset['data_type'] == unique_value[i]].tolist())
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(dataset[input_names], dataset[output_names])
    pre_val = lr.predict(dataset[input_names])
    plt.plot(dataset[input_names], pre_val, "k-", linewidth=2)
    for i in range(len(unique_value)):
        plt.scatter(dataset.loc[unique_index[i],input_names], dataset.loc[unique_index[i],output_names], c=colors[i],s=40,label=data_type[i],marker=markers[i])
    # The coefficients
    a = np.around(lr.coef_[0][0],3)
    b = np.around(lr.intercept_[0],2)
    print(f"Coefficients: a = {a}, b = {b}")
    xl, xr = plt.xlim()
    yd, yu = plt.ylim()
    print(xr,yu)
    x_val = xl + 0.1*(xr-xl)
    y_val = yd + 0.9*(yu - yd)
    # The mean squared error
    r2_val = r2_score(dataset[output_names],pre_val)
    print("Mean squared error: %.2f" % mean_squared_error(dataset[output_names],pre_val))
    # The coefficient of determination: 1 is perfect prediction
    if b < 0:
        sig = '-'
    else:
        sig = '+'
    print("Coefficient of determination: %.2f" % r2_val)
    plt.text(x_val, y_val, f"y = {a}x {sig} {np.abs(b)}", ha='left', wrap=True, fontsize=14)
    plt.text(x_val, 0.95*y_val, f"$R^2$ = {np.around(r2_val,2)}", ha='left', wrap=True, fontsize=14)
    # Figure options
    plt.legend(loc="lower right")
    #plt.title('Linear Fitting')
    plt.xlabel(input_names[0])
    plt.ylabel(output_names[0])
    plt.savefig('linear_fit.png',dpi=150)
    #plt.show()

def subgroup():
    dataset = pd.read_csv('data.csv')
    dataset = dataset[input_names+output_names]
    target = ps.NumericTarget(output_names)
    searchspace = ps.create_selectors(dataset, ignore=output_names)
    task = ps.SubgroupDiscoveryTask(
        dataset,
        target,
        searchspace,
        result_set_size=15,
        depth=5,
        qf=ps.StandardQFNumeric(a = 1.0))
    result = ps.BeamSearch().execute(task)
    data_re = result.to_dataframe()
    data_re.to_csv('subgroup_result.csv')

###################### Feature Selection Functions ##############################
def select_feature():
    train_dataset = pd.read_csv('data.csv')
    feature_names = input_names
    col_names = list(train_dataset.columns)
    # pairplot
    if (plot_pair):
        print(f'- Generate pairplot - joint distribution')
        #sns_plot = sns.pairplot(train_dataset[col_names],diag_kind='kde')
        sns_plot = sns.pairplot(train_dataset[input_names+output_names],height=2.0)
        sns_plot.savefig("pairplot.png",dpi=150, bbox_inches='tight')
    # covariance matrix
    if (plot_comat):
        print(f'- Calculate and plot covariance matrix')
        stdsc = StandardScaler() 
        X_std = stdsc.fit_transform(train_dataset[input_names+output_names].iloc[:,range(0,len(input_names+output_names))].values)
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
                         yticklabels=input_names+output_names,
                         xticklabels=input_names+output_names,
                         vmin = -1, vmax = 1)
        plt.title('Covariance matrix showing correlation coefficients', size = 18)
        plt.tight_layout()
        plt.savefig('comatplot.png',dpi=150)
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
                                                                test_size = 0.1, random_state = 0)
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
        print(imp_coef)
    #Random Forest method
    if (use_randomforest):
        print(f'- Use Random Forest method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.1, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        sorted_idx = rf.feature_importances_.argsort()
        plt.figure()
        plt.barh(np.array(feature_names)[sorted_idx], rf.feature_importances_[sorted_idx])
        plt.savefig('randomforest.png',dpi=150, bbox_inches='tight')
        for i in sorted_idx:
            print('{0} {1}'.format(np.array(feature_names)[i],rf.feature_importances_[i]))
    #Permutation Importance method
    if (use_permutation):
        print(f'- Use Permutation Importance method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.1, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        perm_importance = permutation_importance(rf, X_test, y_test)
        sorted_idx = perm_importance.importances_mean.argsort()
        plt.figure()
        plt.barh(np.array(feature_names)[sorted_idx], perm_importance.importances_mean[sorted_idx])
        plt.savefig('permutation.png',dpi=150, bbox_inches='tight')
        for i in sorted_idx:
            print('{0} {1}'.format(np.array(feature_names)[i],rf.feature_importances_[i]))
    #SHAP Values method
    if (use_shap):
        print(f'- Use SHAP Values method')
        X_train, X_test, y_train, y_test = train_test_split(train_dataset[feature_names], train_dataset[output_names], test_size=0.1, random_state=12)
        rf = RandomForestRegressor(n_estimators=100)
        rf.fit(X_train, np.ravel(y_train))
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        plt.figure()
        shap.summary_plot(shap_values, X_test, max_display=len(feature_names), plot_type="bar",show=False)
        plt.savefig('shap_values.png',dpi=150, bbox_inches='tight')
        
        vals= np.abs(shap_values).mean(0)
        #sorted_idx = vals.argsort()
        #print(sorted_idx)
        for i in sorted_idx:
            print('{0} {1}'.format(feature_names[i],vals[i]))

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
    global job_select_feature
    global input_names, output_names
    global plot_pair, plot_comat, use_back_elim, use_recu_elim, use_lasso, use_randomforest
    global use_permutation, use_shap
    global job_fit_normal_linear, job_subgroup
    global data_type
    with open('INPUT','r') as fin:
        for line in fin:
            if (line.find('job_fit_normal_linear') != -1): # run job_fit_normal_linear or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_fit_normal_linear = True
                else:
                    job_fit_normal_linear = False
            elif (line.find('job_select_feature') != -1): # run job_select_feature or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_select_feature = True
                else:
                    job_select_feature = False
            elif (line.find('job_subgroup') != -1): # run job_subgroup or not
                line = line.lower(); s = line.split('='); s = s[1].strip()
                if (s.find('t') != -1):
                    job_subgroup = True
                else:
                    job_subgroup = False  
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
            elif (line.find('data_type') != -1): # get data_type 
                s = line.split('='); s = s[1].split()
                for i in s:
                    data_type.append(i)
    if (len(input_names) == 0 | len(output_names) == 0):
        return 1
    else:
        return 0

def print_input():
    print(f'Input parameters:')
    print(f'job_fit_normal_linear        = ', job_fit_normal_linear)
    print(f'job_subgroup                 = ', job_subgroup)
    print(f'job_select_feature           = ', job_select_feature)
    print(f'input_names                  = ', input_names)
    print(f'output_names                 = ', output_names)
    print(f'plot_pair                    = ', plot_pair)
    print(f'plot_comat                   = ', plot_comat)
    print(f'use_back_elim                = ', use_back_elim)
    print(f'use_recu_elim                = ', use_recu_elim)
    print(f'use_lasso                    = ', use_lasso)
    print(f'use_randomforest             = ', use_randomforest)
    print(f'use_permutation              = ', use_permutation)
    print(f'use_shap                     = ', use_shap)
    return 0


#-----------------------------
#main program
if __name__ == '__main__':
    # Start timer and read input parameters
    start0 = timer()
    error = read_input()
    if (error == 0):
        print(f'*************************************************************')
        print(f'                Version: {app_ver}')
        print(f'*************************************************************')
        print_input()
        print(f'*************************************************************')
        if (job_select_feature):
            print(f'Check and recommend features for dataset')
            # Start timer
            start = timer()
            select_feature()
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print('*************************************************************')
        if (job_fit_normal_linear):
            print(f'Fit linear function')
            # Start timer
            start = timer()
            normal_linear_fit()
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print('*************************************************************')
        if (job_subgroup):
            print(f'Subgroup discovery')
            # Start timer
            start = timer()
            subgroup()
            end = timer()
            print('Done! Elapsed time: %10.2f s' % (end-start))
            print('*************************************************************')
        end = timer()
        print('Total time: %10.2f s' % (end-start0))
    else:
        print(f'Error in INPUT file')