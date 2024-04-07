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
import requests
from scipy import stats
from tkinter import filedialog
from tkinter import *
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
from sklearn import preprocessing
import tkinter as tk
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA

# FILES FOR TRAINING

csv_member1 = "het_data.csv"
csv_member2 = "Chengxi_data.csv"
csv_member3 = "Alex_data.csv"

data_member1 = pd.read_csv(csv_member1)
data_member2 = pd.read_csv(csv_member2)
data_member3 = pd.read_csv(csv_member3)
data_member4_walking = pd.read_csv("Walking-Raw Data.csv")
data_member4_jumping = pd.read_csv("Jumping-Raw Data.csv")
data_member5_jumping = pd.read_csv("jumping.csv")
data_member5_walking = pd.read_csv("walking.csv")

# preprocessed_values_walking = []
# preprocessed_values_jumping = []






# segment_data returns an array of segmented data
def segment_data(data, window_size):
    num_segments = data.shape[0] // window_size
    segments = [data.iloc[i * window_size:(i + 1) * window_size].reset_index(drop=True) for i in range(num_segments)]
    return segments

#Global Variables

sampling_rate = 100
window_size = 5 * sampling_rate

segmented_data_member1 = segment_data(data_member1, window_size)
# print(segmented_data_member1)
segmented_data_member2 = segment_data(data_member2, window_size)
segmented_data_member3 = segment_data(data_member3, window_size)
segmented_data_member4_walking = segment_data(data_member4_walking, window_size)
segmented_data_member4_jumping = segment_data(data_member4_walking, window_size)
segmented_data_member5_jumping = segment_data(data_member5_jumping, window_size)
segmented_data_member5_walking = segment_data(data_member5_walking, window_size)





def full_set_labeling(dataset, movement_type):
    dataset = pd.DataFrame(dataset)
    if movement_type == 'walking':
        dataset.insert(0, 'label', 0)
    if movement_type == 'jumping':
        dataset.insert(0, 'label', 1)
    return dataset


def determine_array_average(input_array):
    return_array = []
    for array in input_array:
        if sum(array)/len(array) >= 0.5:
            return_array.append(1)
        else:
            return_array.append(0)
    return return_array


def create_and_combine_dataframes(dataset1, dataset2):
    combined_segmented_data = []
    for member in dataset1:
        print("MEMBER: ", member)
        member = pd.DataFrame(member)
        combined_segmented_data.append(member)
    for member in dataset2:
        print("MEMBER: ", member)
        member = pd.DataFrame(member)
        combined_segmented_data.append(member)
    return combined_segmented_data




#https://www.w3schools.com/python/python_for_loops.asp





# TODO: UNCOMMENT THIS
# print("COMBINED: ", combined_segmented_data)
# shuffled_segmented_data = combined_segmented_data.sample(frac=1).reset_index(drop=True)
# train_data_segmented, test_data_segmented = train_test_split(shuffled_segmented_data, test_size=0.1)

# print("Train", train_data_segmented)
# print("Test", test_data_segmented)

# TODO: UNCOMMENT THIS
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

#If There are missing values, to deal with the missing values I will use sample and hold imputation
#https://stackoverflow.com/questions/7696924/how-do-i-create-multiline-comments-in-python


time_column = 'Time (s)'
axis_of_interest = 'Linear Acceleration y (m/s^2)'

plt.style.use('seaborn-darkgrid')
plt.rcParams.update({'font.size': 12})

fig, axs = plt.subplots(2, 2, figsize=(15, 18), sharex=True)

def plot_walking_vs_jumping(walking_data, jumping_data):

    walking_data = pd.DataFrame(walking_data)
    jumping_data = pd.DataFrame(jumping_data)
    walking_data_x = walking_data.iloc[:, 1]
    walking_data_y = walking_data.iloc[:, 2]
    walking_data_z = walking_data.iloc[:, 3]
    walking_data_abs = walking_data.iloc[:, 4]
    jumping_data_x = jumping_data.iloc[:, 1]
    jumping_data_y = jumping_data.iloc[:, 2]
    jumping_data_z = jumping_data.iloc[:, 3]
    jumping_data_abs = jumping_data.iloc[:, 4]
    walking_data_abs = walking_data.iloc[:, 4]


    axs[0,0].plot(jumping_data_x, "b--" , walking_data_x, "r--")
    axs[0, 0].set_title("X Acceleration")

    axs[0, 1].plot(jumping_data_y, "b--", walking_data_y, "r--")
    axs[0, 1].set_title("Y Acceleration")

    axs[1, 0].plot(jumping_data_z, "b--", walking_data_z, "r--")
    axs[1, 0].set_title("Z Acceleration")

    axs[1, 1].plot(jumping_data_abs, "b--", walking_data_abs, "r--")
    axs[1, 1].set_title("Absolute Acceleration")

    plt.show()

def remove_outliers_frame(dataset):

    dataset = pd.DataFrame(dataset)
    z = np.abs(stats.zscore(dataset["Absolute acceleration (m/s^2)"]))
    Q1 = dataset["Absolute acceleration (m/s^2)"].quantile(0.25)
    Q3 = dataset["Absolute acceleration (m/s^2)"].quantile(0.75)
    IQR = Q3 - Q1

        # identify outliers
    threshold = 1.5
    outliers = dataset[(dataset["Absolute acceleration (m/s^2)"] < Q1 - threshold * IQR) | (dataset["Absolute acceleration (m/s^2)"] > Q3 + threshold * IQR)]
    dataframe = dataset.drop(outliers.index)
    dataframe.loc[z > threshold, "Absolute acceleration (m/s^2)"] = dataframe[
            "Absolute acceleration (m/s^2)"].median()
    dataframe = dataframe.interpolate()
    print("DATAFRAME: ", dataframe)
    return dataframe


def remove_outliers(dataset):
    return_array = []
    for dataframe in dataset:
        z = np.abs(stats.zscore(dataframe["Absolute acceleration (m/s^2)"]))
        Q1 = dataframe["Absolute acceleration (m/s^2)"].quantile(0.25)
        Q3 = dataframe["Absolute acceleration (m/s^2)"].quantile(0.75)
        IQR = Q3 - Q1

        # identify outliers
        threshold = 1.5
        outliers = dataframe[(dataframe["Absolute acceleration (m/s^2)"] < Q1 - threshold * IQR) | (dataframe["Absolute acceleration (m/s^2)"] > Q3 + threshold * IQR)]
        dataframe = dataframe.drop(outliers.index)
        dataframe.loc[z > threshold, "Absolute acceleration (m/s^2)"] = dataframe["Absolute acceleration (m/s^2)"].median()
        return_array.append(dataframe)
    return return_array


def graph_all_axes_frame(dataframe):
    full_dataframe = dataframe

    fig, axs = plt.subplots(2, 2, figsize=(15, 18), sharex=True)

    data_x = full_dataframe.iloc[:, 1]
    data_y = full_dataframe.iloc[:, 2]
    data_z = full_dataframe.iloc[:, 3]
    data_abs = full_dataframe.iloc[:, 4]

    axs[0, 0].plot(data_x, "b--", data_x, "r--")
    axs[0, 0].set_title("X Acceleration NOISY")

    axs[0, 1].plot(data_y, "b--", data_y, "r--")
    axs[0, 1].set_title("Y Acceleration NOISY")

    axs[1, 0].plot(data_z, "b--", data_z, "r--")
    axs[1, 0].set_title("Z Acceleration NOISY")

    axs[1, 1].plot(data_abs, "b--", data_abs, "r--")
    axs[1, 1].set_title("Absolute Acceleration NOISY")

    plt.show()
def graph_all_axes(dataset):
    return_array = []
    full_dataframe = pd.DataFrame()
    for dataframe in dataset:
        y_data = dataframe.iloc[:, 0:5]
        return_array.append(y_data)
        mergers = [full_dataframe, y_data]
        full_dataframe = pd.concat(mergers)

    fig, axs = plt.subplots(2, 2, figsize=(15, 18), sharex=True)

    data_x = full_dataframe.iloc[:, 1]
    data_y = full_dataframe.iloc[:, 2]
    data_z = full_dataframe.iloc[:, 3]
    data_abs = full_dataframe.iloc[:, 4]

    axs[0, 0].plot(data_x, "b--", data_x, "r--")
    axs[0, 0].set_title("X Acceleration NOISY")

    axs[0, 1].plot(data_y, "b--", data_y, "r--")
    axs[0, 1].set_title("Y Acceleration NOISY")

    axs[1, 0].plot(data_z, "b--", data_z, "r--")
    axs[1, 0].set_title("Z Acceleration NOISY")

    axs[1, 1].plot(data_abs, "b--", data_abs, "r--")
    axs[1, 1].set_title("Absolute Acceleration NOISY")

    plt.show()


def pre_processing_no_segmentation(dataset):

    graph_all_axes_frame(dataset)

    dataset = remove_outliers_frame(dataset)
    dataset = dataset.iloc[:, 0:5]
    sma_window_size = 10
    dataset = dataset.rolling(sma_window_size).mean()

    fig, axs = plt.subplots(2, 2, figsize=(15, 18), sharex=True)

    data_x = dataset.iloc[:, 1]
    data_y = dataset.iloc[:, 2]
    data_z = dataset.iloc[:, 3]
    data_abs = dataset.iloc[:, 4]

    axs[0, 0].plot(data_x, "b--", data_x, "r--")
    axs[0, 0].set_title("X Acceleration SMA 10")

    axs[0, 1].plot(data_y, "b--", data_y, "r--")
    axs[0, 1].set_title("Y Acceleration SMA 10")

    axs[1, 0].plot(data_z, "b--", data_z, "r--")
    axs[1, 0].set_title("Z Acceleration SMA 10")

    axs[1, 1].plot(data_abs, "b--", data_abs, "r--")
    axs[1, 1].set_title("Absolute Acceleration SMA 10")

    plt.show()
    #Normalizing skewed the data too much and resulted in worse results
    # dataset = normalize_frame(dataset)
    return dataset


def pre_processing(dataset):
    graph_all_axes(dataset)
    #https://www.w3schools.com/python/pandas/pandas_dataframes.asp -> documentation for pandas
    #https://www.w3schools.com/python/python_functions.asp documentation of python functions

    # using sample and hold imputation
    # Doing Dimension Reduction, I am getting rid of the individual directional vectors values, as the absolute acceleration is a product of those values combined
    # dataset = pd.DataFrame(dataset)
    # outlier removal
    dataset = remove_outliers(dataset)
    return_array = []
    full_dataframe = pd.DataFrame()
    for dataframe in dataset:
    # noise reduction
    # TODO: MODIFY SO IT DOES PREPROCESSING ON EVERY DATASET
        y_data = dataframe.iloc[:, 0:5]
        sma_window_size = 10
        y_sma10 = y_data.rolling(sma_window_size).mean()
        return_array.append(y_sma10)
        mergers = [full_dataframe, y_sma10]
        full_dataframe = pd.concat(mergers)

    fig, axs = plt.subplots(2, 2, figsize=(15, 18), sharex=True)


    data_x = full_dataframe.iloc[:, 1]
    data_y = full_dataframe.iloc[:, 2]
    data_z = full_dataframe.iloc[:, 3]
    data_abs = full_dataframe.iloc[:, 4]


    axs[0, 0].plot(data_x, "b--", data_x, "r--")
    axs[0, 0].set_title("X Acceleration SMA 10")

    axs[0, 1].plot(data_y, "b--", data_y, "r--")
    axs[0, 1].set_title("Y Acceleration SMA 10")

    axs[1, 0].plot(data_z, "b--", data_z, "r--")
    axs[1, 0].set_title("Z Acceleration SMA 10")

    axs[1, 1].plot(data_abs, "b--", data_abs, "r--")
    axs[1, 1].set_title("Absolute Acceleration SMA 10")

    plt.show()

    return return_array



def feature_extraction(dataset):
    dataframe_features = []

    i = 0
    for dataframe in dataset:
        # print("WINDOW: ", i, " :", dataframe.iloc[:,4])
        #FEATURE EXTRACTION
        # print("DATAFRAME: ", dataframe)
        abs_accel = dataframe
        # print("ABS_ACCEL: ",abs_accel)
        features = pd.DataFrame(columns=['mean', 'std', 'max', 'min', 'skewness', 'kurtosis', 'median', 'sum', 'variance', 'range'])
        features['mean'] = abs_accel.mean()
        features['std'] = abs_accel.std()
        features['max'] = abs_accel.max()
        features['min'] = abs_accel.min()
        features['skewness'] = abs_accel.skew()
        features['kurtosis'] = abs_accel.kurt()
        features['median'] = abs_accel.median()
        features['sum'] = abs_accel.sum()
        features['variance'] = abs_accel.var()
        features['variance'] = abs_accel.iloc[:,4].max()-abs_accel.iloc[:,4].min()
        # features['range'] = abs_accel.rank()
        # print("FEATURES: ", features)
        dataframe_features.append(features)

        i = i + 1
    return dataframe_features


def normalize_frame(dataset):
    sc = preprocessing.StandardScaler()
    data = dataset.iloc[10:, :]
    dataset = sc.fit_transform(data)
    return dataset

def normalize(dataframe):
    normalized_data = []
    for frame in dataframe:
        sc = preprocessing.StandardScaler()
        data = pd.DataFrame(frame)
        data = data.iloc[10:-1, :]
        dataset = sc.fit_transform(data)
        normalized_data.append(dataset)
    return normalized_data



def classifier(segmented_data):

    # CLASSIFY INTO WALKING AND JUMPING CLASSES
    walking_dataset = full_set_labeling(pre_processing_no_segmentation(data_member4_walking), "walking").iloc[10:]
    print("WAlking: ", walking_dataset)
    jumping_dataset = full_set_labeling(pre_processing_no_segmentation(data_member4_jumping), "jumping").iloc[10:]


    scaler = StandardScaler()
    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)


    # Need to combine datasets and shuffle them
    frames = [walking_dataset, jumping_dataset]
    full_frame = pd.concat(frames)


    labels = full_frame.iloc[:,0]
    features = full_frame.iloc[:,1:]
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.1, random_state=0, shuffle=True)
    # print(X_train)
    # print(Y_train)
    clf.fit(X_train, Y_train)

    Y_pred = clf.predict(X_test)
    y_clf_prob = clf.predict_proba(X_test)

    acc = accuracy_score(Y_test, Y_pred)
    print("accuracy is: ", acc)

    recall = recall_score(Y_test, Y_pred)
    print("recall is: ", recall)

    cm = confusion_matrix(Y_test, Y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.show()

    fpr, tpr, _ = roc_curve(Y_test, y_clf_prob[:, 1], pos_label=clf.classes_[1])
    roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
    plt.show()

    f1 = f1_score(Y_test, Y_pred)
    print("The F1 Score is: ", f1)

    auc = roc_auc_score(Y_test, y_clf_prob[:, 1])
    print("The AUC is: ", auc)


    # testFile = segmented_data_member3
    testFile = segmented_data
    returnArray = []
    for frame in testFile:
        # print("FRAME: ", frame)
        # print("PREDICTION: ", clf.predict(frame))
        returnArray.append(clf.predict(frame))

    print(returnArray)
    return returnArray

def generate_csv(values):

    time_intervals = []
    running_or_jumping = []
    front_counter = 0
    back_counter = 5

    for value in values:
        time_intervals.append("Time {}-{}".format(front_counter, back_counter))
        front_counter = front_counter + 5
        back_counter = back_counter + 5
        if value == 1:
            running_or_jumping.append("jumping")
        else:
            running_or_jumping.append("running")

    return_dataframe = {
        "time_intervals": time_intervals,
        "running_or_jumping": running_or_jumping
    }

    return_dataframe = pd.DataFrame(return_dataframe)
    return_dataframe.to_csv("outputCSV", sep=',', index=True, encoding='utf-8')


def upload_file(event=None):
    window = Tk()
    filename = filedialog.askopenfilename()
    file_to_segment = pd.read_csv(filename)
    file_to_segment = segment_data(file_to_segment, window_size)



    # print(filename)
    print('Selected:', filename)
    fig = Figure(figsize=(5, 5), dpi=100)
    y = determine_array_average(classifier(file_to_segment))
    generate_csv(y)
    plot1 = fig.add_subplot(111)
    plot1.plot(y)

    canvas = FigureCanvasTkAgg(fig,
                               master=window)
    canvas.draw()

    canvas.get_tk_widget().pack()



def graphical_user_interface():
    # How i learned tkinter as it was not taught in the course
    # https: // realpython.com / python - gui - tkinter /  # building-your-first-python-gui-application-with-tkinter
    # Learning how to import files in tkinter
    # https://dev.to/jairajsahgal/creating-a-file-uploader-in-python-18e0

    window = Tk()
    window.title('292 Final Project')
    window.geometry("500x500")
    opening_statement = tk.Label(text="Welcome To Our Final Project for 292")
    opening_statement.pack()
    button = tk.Button(window, text='Upload File', command=upload_file)
    button.pack()

    # run the gui
    window.mainloop()


def main_function():
    plot_walking_vs_jumping(data_member4_walking, data_member4_jumping)

    preprocessed_values = pre_processing(create_and_combine_dataframes(segment_data(data_member1, window_size), segment_data(data_member2, window_size)))


    features = feature_extraction(preprocessed_values)
    # features = feature_extraction(preprocessed_values)
    print(features)

    graphical_user_interface()

main_function()
