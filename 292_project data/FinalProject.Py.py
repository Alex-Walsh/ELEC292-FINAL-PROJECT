import matplotlib.pyplot as plt
import numpy
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import h5py
from sklearn.model_selection import train_test_split

# Define file paths for the CSV datasets
csv_member1 = r"D:\USERS PROFILES\Desktop\het_data.csv"
csv_member2 = r"D:\USERS PROFILES\Desktop\Chengxi_data.csv"
csv_member3 = r"D:\USERS PROFILES\Desktop\Alex_data.csv"

# Read the CSV files into individual pandas DataFrames
data_member1 = pd.read_csv(csv_member1)
data_member2 = pd.read_csv(csv_member2)
data_member3 = pd.read_csv(csv_member3)

# Concatenate the individual DataFrames into a single combined DataFrame
combined_data = pd.concat([data_member1, data_member2, data_member3])

# Partition the combined DataFrame into training (90%) and testing (10%) sets
train_data, test_data = train_test_split(combined_data, test_size=0.1)

# Specify the output path for the HDF5 file
hdf5_file_path = 'D:/USERS PROFILES/Desktop/hdf5_data.h5'

# Open an HDF5 file and create hierarchical groups and datasets
with h5py.File(hdf5_file_path, 'w') as hdf_file:
    # Create a primary group called 'dataset' in the HDF5 file
    dataset_group = hdf_file.create_group('dataset')

    # Within the 'dataset' group, create two subgroups to store training and testing data
    train_group = dataset_group.create_group('Train')
    test_group = dataset_group.create_group('Test')

    # Store the training and testing data as datasets within their respective subgroups
    train_group.create_dataset('data', data=train_data.to_numpy())
    test_group.create_dataset('data', data=test_data.to_numpy())

    # Create separate groups for each member's data at the root level of the HDF5 file
    member1_group = hdf_file.create_group('Member1 name')
    member2_group = hdf_file.create_group('Member2 name')
    member3_group = hdf_file.create_group('Member3 name')

    # Store each member's data in a dataset within their respective group
    member1_group.create_dataset('data', data=data_member1.to_numpy())
    member2_group.create_dataset('data', data=data_member2.to_numpy())
    member3_group.create_dataset('data', data=data_member3.to_numpy())


csv_member1 = "het_data.csv"
csv_member2 = "Alex_data.csv"
csv_member3 = "Chengxi_data.csv"

data_member1 = pd.read_csv(csv_member1)
data_member2 = pd.read_csv(csv_member2)
data_member3 = pd.read_csv(csv_member3)

time_column = 'Time (s)'
axis_of_interest = "Acceleration_y"

plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})

fig, axs = plt.subplots(3, 1, figsize=(15, 18), sharex=True)


def plot_member_data(ax, data, member_label, color):

    data = data[data[time_column] <= 300]

    ax.plot(data[time_column], data[axis_of_interest], label=f'{member_label}', linewidth=2, color=color)
    ax.set_title(f'{member_label} - {axis_of_interest} over Time')
    ax.set_xlabel('')
    ax.set_ylabel(f'{axis_of_interest} (m/s^2)')

    ax.legend(loc='upper right', frameon=True, shadow=True)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_facecolor('#f5f5f5')  # light grey background color


plot_member_data(axs[0], data_member1, 'Het', 'blue')
plot_member_data(axs[1], data_member2, 'Alexander', 'green')
plot_member_data(axs[2], data_member3, 'Chengxi', 'red')


plt.tight_layout()
plt.show()




# dataset = pd.read_csv("Raw Data.csv")

#PRE-PROCESSING : ALEX WALSH

# X = dataset.iloc[:,4]
# print(X)
# X = pd.DataFrame(X)
# n_sample = 30565
# x_input = np.arange(n_sample)
# windowSize = 5
# y_sma40 = X.rolling(windowSize).mean()
# windowSize = 50
# y_sma60 = X.rolling(windowSize).mean()
# windowSize = 80
# y_sma80 = X.rolling(windowSize).mean()
# #PLOTTING
# fig, ax = plt.subplots(figsize=(10,10))
# ax.plot(x_input, X, linewidth = 2)
# ax.plot(x_input, y_sma40, linewidth = 2)
# ax.plot(x_input, y_sma60, linewidth = 2)
# ax.plot(x_input, y_sma80, linewidth = 2)
#
# plt.show()
#dealing with possible missing data

#dimensionality reduction?


#noise reduction


#source separation





# sc = preprocessing.StandardScaler()
# data = sc.fit_transform(X)




#Normalization and feature extraction