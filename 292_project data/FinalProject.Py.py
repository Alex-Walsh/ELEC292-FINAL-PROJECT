import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import h5py

import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

csv_member1 = "het_data.csv"
csv_member2 = "Chengxi_data.csv"
csv_member3 = "Alex_data.csv"

data_member1 = pd.read_csv(csv_member1)
data_member2 = pd.read_csv(csv_member2)
data_member3 = pd.read_csv(csv_member3)


def segment_data(data, window_size):
    num_segments = data.shape[0] // window_size
    segments = [data.iloc[i * window_size:(i + 1) * window_size].reset_index(drop=True) for i in range(num_segments)]
    return pd.concat(segments, ignore_index=True)


sampling_rate = 100
window_size = 5 * sampling_rate

segmented_data_member1 = segment_data(data_member1, window_size)
segmented_data_member2 = segment_data(data_member2, window_size)
segmented_data_member3 = segment_data(data_member3, window_size)

combined_segmented_data = pd.concat([segmented_data_member1, segmented_data_member2, segmented_data_member3],
                                    ignore_index=True)
shuffled_segmented_data = combined_segmented_data.sample(frac=1).reset_index(drop=True)
train_data_segmented, test_data_segmented = train_test_split(shuffled_segmented_data, test_size=0.1)

hdf5_file_path = 'D:/USERS PROFILES/Desktop/hdf5_data.h5'

with h5py.File(hdf5_file_path, 'w') as hdf_file:
    dataset_group = hdf_file.create_group('dataset')
    train_group = dataset_group.create_group('Train')
    train_group.create_dataset('data', data=train_data_segmented.to_numpy())
    test_group = dataset_group.create_group('Test')
    test_group.create_dataset('data', data=test_data_segmented.to_numpy())

    member1_group = hdf_file.create_group('Member1 name')
    member1_group.create_dataset('data', data=data_member1.to_numpy())
    member2_group = hdf_file.create_group('Member2 name')
    member2_group.create_dataset('data', data=data_member2.to_numpy())
    member3_group = hdf_file.create_group('Member3 name')
    member3_group.create_dataset('data', data=data_member3.to_numpy())
