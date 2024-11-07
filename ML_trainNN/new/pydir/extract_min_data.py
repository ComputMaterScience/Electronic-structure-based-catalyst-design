# Join the data files together
import pandas as pd
import numpy as np
import sys

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
    data.to_csv(output, index=False, encoding='utf-8-sig')

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        inname = sys.argv[1]
        outname = sys.argv[2]
        colname1 = sys.argv[3]
        colname2 = sys.argv[4]
    else:
        inname = 'combined_data.csv'
        outname = 'combined_data.csv'
        colname1 = 'tot_en'
        colname2 = 'formula'
    sort_min_data(inname, outname, colname1, colname2)