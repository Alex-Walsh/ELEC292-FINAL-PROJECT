import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
from gensim.parsing import preprocessing
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import h5py
from sklearn import preprocessing
import h5py
from sklearn.model_selection import train_test_split

#STEP 2 - HET

csv_member1 = "het_data.csv"
csv_member2 = "Chengxi_data.csv"
csv_member3 = "Alex_data.csv"

data_member1 = pd.read_csv(csv_member1)
data_member2 = pd.read_csv(csv_member2)
data_member3 = pd.read_csv(csv_member3)
print("DM1", data_member1)

def segment_data(data, window_size):
    num_segments = data.shape[0] // window_size
    segments = [data.iloc[i * window_size:(i + 1) * window_size].reset_index(drop=True) for i in range(num_segments)]
    return segments
    # print("Spd.concat(segments, axis=1))
    # return pd.concat(segments, ignore_index=True)


sampling_rate = 100
window_size = 5 * sampling_rate

segmented_data_member1 = segment_data(data_member1, window_size)
print("SEGMENTED_DM1: ", segmented_data_member1)
segmented_data_member2 = segment_data(data_member2, window_size)
segmented_data_member3 = segment_data(data_member3, window_size)

#TODO: UNCOMMENT THIS
# combined_segmented_data = pd.concat([segmented_data_member1, segmented_data_member2, segmented_data_member3],
#                                     ignore_index=True)
# shuffled_segmented_data = combined_segmented_data.sample(frac=1).reset_index(drop=True)
# train_data_segmented, test_data_segmented = train_test_split(shuffled_segmented_data, test_size=0.1)
#
# print("Train", train_data_segmented)
# print("Test", test_data_segmented)

#TODO: UNCOMMENT THIS
# hdf5_file_path = 'hdf5_data.h5'
#
# with h5py.File(hdf5_file_path, 'w') as hdf_file:
#     dataset_group = hdf_file.create_group('dataset')
#     train_group = dataset_group.create_group('Train')
#     train_group.create_dataset('data', data=train_data_segmented.to_numpy())
#     test_group = dataset_group.create_group('Test')
#     test_group.create_dataset('data', data=test_data_segmented.to_numpy())
#
#     member1_group = hdf_file.create_group('Member1 name')
#     member1_group.create_dataset('data', data=data_member1.to_numpy())
#     member2_group = hdf_file.create_group('Member2 name')
#     member2_group.create_dataset('data', data=data_member2.to_numpy())
#     member3_group = hdf_file.create_group('Member3 name')
#     member3_group.create_dataset('data', data=data_member3.to_numpy())
#


# PREPROCESSING - ALEX WALSH

#There are missing values, to deal with the missing values I will use sample and hold imputation


#https://stackoverflow.com/questions/7696924/how-do-i-create-multiline-comments-in-python
'''
def pre_processing(dataset):
    #https://www.w3schools.com/python/pandas/pandas_dataframes.asp -> documentation for pandas
    #https://www.w3schools.com/python/python_functions.asp documentation of python functions

    #using sample and hold imputation
    #Doing Dimension Reduction, I am getting rid of the individual directional vectors values, as the absolute acceleration is a product of those values combined
    dataset = pd.DataFrame(dataset)
    dataset = dataset.drop(columns=["Linear Acceleration x (m/s^2)","Linear Acceleration y (m/s^2)", "Linear Acceleration z (m/s^2)" , "Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"])
    #acceleration does not change instantly, it changes gradually so I will use sample and hold imputation so that we can use that

    #noise reduction
#TODO: MODIFY SO IT DOES PREPROCESSING ON EVERY DATASET
    sma_window_size = 5
    # print(dataset.iloc[:, 1])
    y_data = dataset.iloc[:,1]
    print(y_data)
    y_sma5 = y_data.rolling(sma_window_size).mean()
    sma_window_size = 10
    y_sma10 = y_data.rolling(sma_window_size).mean()
    x_length = len(dataset)
    x_input = np.arange(x_length)
    # x_input = dataset.iloc[:, 0]
    print("X-AXIS", x_input)

    fig, ax = plt.subplots(figsize=(10,10))
    ax.plot(x_input, y_data.to_numpy(), linewidth=2)
    ax.plot(x_input, y_sma5.to_numpy(), linewidth = 2)
    ax.plot(x_input, y_sma10.to_numpy(), linewidth=2)
    ax.legend(['NOISY', 'SMA5', 'SMA40'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Amplitude')
    plt.show()
    # print(dataset)

    #SMA DONE
    #Now Going to fill missing values
    # for i in range(x_length):
    #     if dataset.iloc[i, 1].isna():

#TODO: DETECT AND REMOVE OUTLIERS?


#TODO: NORMALIZE DATA



pre_processing(segmented_data_member1)

'''



dataframe_features = []

def feature_extraction(dataset):
    # TODO: FINISH EXTRACTING 10 DIFFERENT FEATURES, CURRENTLY ONLY AT 8
    i = 0
    for dataframe in dataset:
        # print("WINDOW: ", i, " :", dataframe.iloc[:,4])
        #FEATURE EXTRACTION
        abs_accel = dataframe.iloc[:,4]
        features = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'skewness', 'kurtosis', 'median','range'])
        features['mean'] = abs_accel.rolling(window=window_size).mean()
        features['std'] = abs_accel.rolling(window=window_size).std()
        features['max'] = abs_accel.rolling(window=window_size).max()
        features['min'] = abs_accel.rolling(window=window_size).min()
        features['skewness'] = abs_accel.skew()
        features['kurtosis'] = abs_accel.kurt()
        features['median'] = abs_accel.median()
        features['range'] = abs_accel.rank()
        dataframe_features.append(features)

        i = i + 1


def normalize(dataframe):
    for frame in dataframe:
        sc = preprocessing.StandardScaler()
        data = frame.iloc[:, 1:5]
        print(f'mean before normalizing: {data.mean(axis=0)}')
        print(f'std before normalizing: {data.std(axis=0)}')
        dataset = sc.fit_transform(data)
        print(f'mean after normalizing: {dataset.mean(axis=0).round()}')
        print(f'std after normalizing: {dataset.std(axis=0).round()}')




feature_extraction(segmented_data_member1)
normalize(segmented_data_member1)

