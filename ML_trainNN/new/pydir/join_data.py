# Join the data files together
import pandas as pd
import os
import sys

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
    return len(list_files)

if __name__ == '__main__':
    if (len(sys.argv) > 1):
        folname = sys.argv[1]
        finame = sys.argv[2]
    else:
        folname = 'data'
        finame = 'combined_data.csv'
    num_data_file = combineData(folname,finame)