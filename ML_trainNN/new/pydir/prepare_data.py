# Join the data files together
import pandas as pd
import json
import sys

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

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        folname = sys.argv[1]
        finame = sys.argv[2]
    else:
        folname = 'data'
        finame = 'combined_data.csv'
    num_data_file = combineData(folname,finame)