
# coding: utf-8

# # Siamese CNN & OCSVM
# 
# *Created by Holger Buech, Q1/2019*
# 
# **Description**   
# 
# Reimplemenation of an approach to Continuous Authentication described by [1]. It leverages a Siamese CNN to generate Deep Features, which are then used as input for an OCSVM authentication classifier.  
# 
# **Purpose**
# 
# - Verify results of [1]
# - Test the approach with upfront global subject wise normalization (NAIVE_APPROACH)
# - Change the normalization setting to be more realistic: Training data is normalized upfront again, but the Testing data is normalized using a single scaler fitted on training data only. (VALID_APPROACH)
# - Identify parameters performing better in a valid setup than the parameters proposed by [1]. (ALTERNATIVE_APPROACH) 
# 
# **Data Sources**   
# 
# - [H-MOG Dataset](http://www.cs.wm.edu/~qyang/hmog.html)  
#   (Downloaded beforehand using  [./src/data/make_dataset.py](./src/data/make_dataset.py), stored in [./data/external/hmog_dataset/](./data/external/hmog_dataset/) and converted to [./data/processed/hmog_dataset.hdf5](./data/processed/hmog_dataset.hdf5))
# 
# **References**   
# 
# - [1] Centeno, M. P. et al. (2018): Mobile Based Continuous Authentication Using Deep Features. Proceedings of the 2^nd International Workshop on Embedded and Mobile Deep Learning (EMDL), 2018, 19-24.
# 
# **Table of Contents**
# 
# **1 - [Preparations](#1)**   
# 1.1 - [Imports](#1.1)   
# 1.2 - [Configuration](#1.2)   
# 1.3 - [Experiment Parameters](#1.3)   
# 1.4 - [Select Approach](#1.4)   
# 
# **2 - [Initial Data Prepratation](#2)**   
# 2.1 - [Load Dataset](#2.1)   
# 2.2 - [Normalize Features (if global)](#2.2)   
# 2.3 - [Split Dataset for Valid/Test](#2.3)   
# 2.4 - [Normalize Features (if not global)](#2.4)   
# 2.5 - [Check Splits](#2.5)   
# 2.6 - [Reshape Features](#2.6)     
# 
# **3 - [Generate Scenario Pairs](#3)**    
# 3.1 - [Load cached Data](#3.1)  
# 3.2 - [Build positive/negative Pairs](#3.2)  
# 3.3 - [Inspect Pairs](#3.3)  
# 3.4 - [Cache Pairs](#3.4)  
# 
# **4 - [Siamese Network](#4)**  
# 4.1 - [Load cached Pairs](#4.1)   
# 4.2 - [Build Model](#4.2)   
# 4.3 - [Prepare Features](#4.3)   
# 4.4 - [Search optimal Epoch](#4.4)   
# 4.5 - [Check Distances](#4.5)   
# 4.6 - [Rebuild and train to optimal Epoch](#4.6)   
# 4.7 - [Cache Model](#4.7)   
# 
# **5 - [Visualize Deep Features](#5)**   
# 5.1 - [Load cached Data](#5.1)  
# 5.2 - [Extract CNN from Siamese Model](#5.2)  
# 5.3 - [Test Generation of Deep Features](#5.3)  
# 5.4 - [Visualize in 2D using PCA](#5.4)  
# 
# **6 - [OCSVM](#6)**  
# 6.1 - [Load cached Data](#6.1)  
# 6.2 - [Load trained Siamese Model](#6.2)  
# 6.3 - [Search for Parameters](#6.3)  
# 6.4 - [Inspect Search Results](#6.4) 
# 
# **7 - [Testing](#7)**  
# 7.1 - [Load cached Data](#7.1)  
# 7.2 - [Evaluate Auth Performance](#7.2)  
# 7.3 - [Evaluate increasing Training Set Size (Training Delay)](#7.3)  
# 7.4 - [Evaluate increasing Test Set Sizes (Detection Delay)](#7.4)  
# 
# **8 - [Report Results](#8)**  

# ## 1. Preparations <a id='1'>&nbsp;</a> 

# ### 1.1 Imports <a id='1.1'>&nbsp;</a> 
# **Note:** The custom `DatasetLoader` is a helper class for easier loading and subsetting data from the datasets.

# In[1]:


# Standard
from pathlib import Path
import os
import sys
import warnings
import random
import dataclasses
import math
import multiprocessing as mp

# Extra
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.svm import OneClassSVM
from sklearn.model_selection import cross_validate, RandomizedSearchCV
from sklearn.decomposition import PCA
import statsmodels.stats.api as sms
#import tensorflow as tf
#from keras import backend as K
#from keras.models import Model
from utils_dataset import *
#from triplet.custom_layers import subtract, norm, subtract1
'''
from keras.layers import (
    Dense,
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Lambda,
    Conv2D,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    GlobalAveragePooling1D,
    Activation,
    add,
    concatenate,
    BatchNormalization,
    GlobalMaxPooling2D
    
)
from keras.utils import plot_model
from keras.optimizers import Adam, SGD,RMSprop
from keras.models import load_model
from keras.callbacks import Callback
'''
from tqdm.auto import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display
from scipy.fftpack import fft,fft2, fftshift
from statsmodels import robust
from scipy.stats import skew, kurtosis, entropy
from features import Features
from scipy import stats
#from keras.layers.advanced_activations import LeakyReLU
# Custom

print("the path is:", sys.path)
from utility.dataset_loader_hdf5 import DatasetLoader
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from numpy import array 
from sklearn.preprocessing import LabelEncoder

from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
import torch
#from ../../src/utility/dataset_loader_hdf5 import DatasetLoader
# Global utitlity functions are loaded from separate notebook:
#get_ipython().magic('run utils.ipynb')


# ### 1.2 Configuration <a id='1.2'>&nbsp;</a>

# In[2]:

#print("---------------------history_dict.keys()------------------:", history_dict.keys())
# Configure Data Loading & Seed
SEED = 712  # Used for every random function
#HMOG_HDF5 = Path.cwd().parent / "data" / "processed" / "hmog_dataset.hdf5"
HMOG_HDF5 = "/home/tremb1e/work/mobilephone/VQGAN-pytorch-main-con-nor/hmog_dataset.hdf5"
#HMOG_HDF5 = None
#HMOG_HDF5 = "/home/tmac/data/processed/hmog_dataset.hdf5"
EXCLUDE_COLS = ["sys_time"]
CORES = mp.cpu_count()

# For plots and CSVs
OUTPUT_PATH = Path.cwd() / "output" / "chapter-6-1-4-siamese-cnn"  # Cached data & csvs
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path.cwd().parent / "reports" / "figures" # Figures for thesis
REPORT_PATH.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
'''
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
'''
'''
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
'''
# Improve performance of Tensorflow (this improved speed _a_lot_ on my machine!!!)

# Plotting
#get_ipython().magic('matplotlib inline')
utils_set_output_style()

# Silence various deprecation warnings...
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
#tf.logging.set_verbosity(tf.logging.ERROR)
#np.warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore")


# In[3]:


# Workaround to remove ugly spacing between tqdm progress bars:
HTML("<style>.p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty{padding: 0;border: 0;} div.output_subarea{padding:0;}</style>")


# ### 1.3 Experiment Parameters <a id='1.3'>&nbsp;</a> 
# Selection of parameters set that had been tested in this notebook. Select one of them to reproduce results.

# In[66]:
'''
胡明明:数据转换，把传感器数据经过特征值计算后再输入数据
'''
def FeatureTransfer(dataset, samples, overlap, feautureObject, a_computeFeauture = 0):

    for w in range(0, dataset.shape[0] - samples, overlap):
        end = w + samples 
        
        
        #DFT
        discreteFourier = fft(dataset.iloc[w:end, a_computeFeauture])
        # Frequencies
        freq = np.fft.fftfreq(samples)

        # Amplitudes
        idx = (np.absolute(discreteFourier)).argsort()[-2:][::-1]
        amplitude1 = np.absolute(discreteFourier[idx[0]])#振幅
        amplitude2 = np.absolute(discreteFourier[idx[1]])
        frequency2 = freq[idx[1]]

        # Frequency features
        mean_frequency = np.mean(freq)
        Median_frequency = np.median(freq)
        feautureObject.setAmplitude1(amplitude1)
        feautureObject.setAmplitude2(amplitude2)
        feautureObject.setFrequency2(frequency2)
        feautureObject.setMean_frequency(mean_frequency)
        feautureObject.setMedian_frequency(Median_frequency)

        # Time Based Feautures
        
        feautureObject.setΜean(np.mean(dataset.iloc[w:end, a_computeFeauture]))#(mean)
        feautureObject.setSTD(np.std(dataset.iloc[w:end, a_computeFeauture]))#标准差(stanard deviation)
        feautureObject.setVariance(np.var(dataset.iloc[w:end, a_computeFeauture]))#方差(variance)
        CoefficientOfVariation = np.mean(dataset.iloc[w:end, a_computeFeauture])/np.std(dataset.iloc[w:end, a_computeFeauture])
        feautureObject.setCoefficientOfVariation(CoefficientOfVariation)
        feautureObject.setMax(np.max(dataset.iloc[w:end, a_computeFeauture]))
        feautureObject.setMin(np.min(dataset.iloc[w:end, a_computeFeauture]))
        feautureObject.setRange(np.ptp(dataset.iloc[w:end, a_computeFeauture]))#最大值与最小值的差
        CoefficientOfRange = np.ptp(dataset.iloc[w:end, a_computeFeauture])/np.mean(dataset.iloc[w:end, a_computeFeauture])
        #print("the CofficientOfRange is:", CoefficientOfRange)
        feautureObject.setCoefficientOfRange(CoefficientOfRange)

        percentile = np.percentile(dataset.iloc[w:end, a_computeFeauture], [25, 50, 75, 95])
        feautureObject.setPercentile25(percentile[0])
        feautureObject.setPercentile50(percentile[1])
        feautureObject.setPercentile75(percentile[2])
        feautureObject.setPercentile95(percentile[3])
        InterQuartileRange = (percentile[2] - percentile[0])
        feautureObject.setInterQuartileRange(InterQuartileRange)
        feautureObject.setMeanAbsoluteDeviation(robust.mad(dataset.iloc[w:end, a_computeFeauture]))
        feautureObject.setMedianAbsoluteDeviation(stats.median_absolute_deviation(dataset.iloc[w:end, a_computeFeauture]))
        feautureObject.setEntropy(entropy(dataset.iloc[w:end, a_computeFeauture], base = 2))

        feautureObject.setKurtosis(kurtosis(dataset.iloc[w:end, a_computeFeauture]))
        feautureObject.setSkewness(skew(dataset.iloc[w:end, a_computeFeauture]))
        
        feautureObject.setSubject(dataset.iloc[0, 1])
        feautureObject.setSession(dataset.iloc[0, 2])
        feautureObject.setTaskType(dataset.iloc[0, 3])

        # Output Label
        #feautureObject.setY(output)

    return  feautureObject



def DataframeGenerate(dataset, feautureObject=None, samples=100, overlap=50, computeFeauture = 0):
    
    dataFrame = pd.DataFrame(columns=['a_magnitude','subject', 'session', 'task_type'])
    x = dataset['acc_x']
    y = dataset['acc_y']    
    z = dataset['acc_z']
    m = x**2 + y**2 + z**2
    m = np.sqrt(m)
    dataFrame['a_magnitude'] = m
    dataFrame['subject'] = dataset['subject']
    dataFrame['session'] = dataset['session']
    dataFrame['task_type'] = dataset['task_type']
    subjects = dataset["subject"].unique().tolist()
    tempList = []
    print("the head of dataFrame is:\n", dataFrame.head())
    
    for i in range(len(subjects)):
        tempList.clear()
        tempList.append(subjects[i])
        df = dataFrame.query("subject in @tempList").copy()
        print("the subject is: {0},the num is:{1}".format(subjects[i], df.shape[0]))
        FeatureTransfer(df, samples, overlap, feautureObject, computeFeauture)


def DataframeGenerateRaw(dataset):
    
    df_data = pd.DataFrame()
    for session, df_group in tqdm(dataset.groupby("session"), desc="Session", leave=False):
        dataFrame = pd.DataFrame(columns=['a_magnitude','subject', 'session', 'task_type'])
        #print("the len of df_group is:", len(df_group), session)
        num = min(20000, len(df_group))
        x = df_group['acc_x'][:num]
        y = df_group['acc_y'][:num]    
        z = df_group['acc_z'][:num]
        a_m = x**2 + y**2 + z**2
        a_m = np.sqrt(a_m)
        
        dataFrame['acc_x'] = df_group['acc_x'][:num]
        dataFrame['acc_y'] = df_group['acc_y'][:num]
        dataFrame['acc_z'] = df_group['acc_z'][:num]
        dataFrame['a_magnitude'] = a_m
        
        x = df_group['gyr_x'][:num]
        y = df_group['gyr_y'][:num]   
        z = df_group['gyr_z'][:num]
        g_m = x**2 + y**2 + z**2
        g_m = np.sqrt(g_m)
        
        dataFrame['gyr_x'] = df_group['gyr_x'][:num]
        dataFrame['gyr_y'] = df_group['gyr_y'][:num]
        dataFrame['gyr_z'] = df_group['gyr_z'][:num]
        dataFrame['g_magnitude'] = g_m
        
        x = df_group['mag_x'][:num]
        y = df_group['mag_y'][:num]    
        z = df_group['mag_z'][:num]
        m_m = x**2 + y**2 + z**2
        m_m = np.sqrt(m_m)
    
        dataFrame['mag_x'] = df_group['mag_x'][:num]
        dataFrame['mag_y'] = df_group['mag_y'][:num]
        dataFrame['mag_z'] = df_group['mag_z'][:num]
        dataFrame['m_magnitude'] = m_m
        
        dataFrame['subject'] = df_group['subject'][:num]
        dataFrame['session'] = df_group['session'][:num]
        dataFrame['task_type'] = df_group['task_type'][:num]
        df_data = df_data.append(dataFrame, ignore_index=True)
    subjects = dataset["subject"].unique().tolist()
    print("the num of subject is:", len(subjects))
    del dataset
    print("the head of dataFrame is:\n", df_data.head(2))
    
    return df_data
    

@dataclasses.dataclass
class ExperimentParameters:
    """Contains all relevant parameters to run an experiment."""

    name: str  # Name of Experiments Parameter set. Used as identifier for charts etc.

    # Data / Splitting:
    frequency: int
    feature_cols: list  # Columns used as features
    max_subjects: int
    exclude_subjects: list  # Don't load data from those users
    n_valid_train_subjects: int
    n_valid_test_subjects: int
    n_test_train_subjects: int
    n_test_test_subjects: int
    seconds_per_subject_train: float
    seconds_per_subject_test: float
    task_types: list  # Limit scenarios to [1, 3, 5] for sitting or [2, 4, 6] for walking, or don't limit (None)

    # Reshaping
    window_size: int  # After resampling
    step_width: int  # After resampling

    # Normalization
    scaler: str  # {"std", "robust", "minmax"}
    scaler_scope: str  # {"subject", "session"}
    scaler_global: bool  # scale training and testing sets at once (True), or fit scaler on training only (False)

    # Siamese Network
    max_pairs_per_session: int  # Max. number of pairs per session
    margin: float  # Contrastive Loss Margin
    model_variant: str  # {"1d", "2d"} Type of architecture
    filters: list  # List of length 4, containing number of filters for conv layers
    epochs_best: int  # Train epochs to for final model
    epochs_max: int
    batch_size: int
    optimizer: str  # Optimizer to use for Siamese Network
    optimizer_lr: float  # Learning Rate
    optimizer_decay: float

    # OCSVM
    ocsvm_nu: float  # Best value found in random search, used for final model
    ocsvm_gamma: float  # Best value found in random search, used for final model

    # Calculated values
    def __post_init__(self):
        # HDF key of table:
        self.table_name = f"sensors_{self.frequency}hz"

        # Number of samples per _session_ used for training:
        self.samples_per_subject_train = math.ceil(
            (self.seconds_per_subject_train * 100)
            / (100 / self.frequency)
            / self.window_size
        )

        # Number of samples per _session_ used for testing:
        self.samples_per_subject_test = math.ceil(
            (self.seconds_per_subject_test * 100)
            / (100 / self.frequency)
            / self.window_size
        )


# INSTANCES
# ===========================================================

custom_feature_cols=[
        #acc
        "acc_x",
        "acc_y",
        "acc_z",
        "a_magnitude",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "g_magnitude",
        "mag_x",
        "mag_y",
        "mag_z",
        "m_magnitude",
    ]


'''
custom_feature_cols=[
        #acc
        "acc_x",
        "acc_y",
        "acc_z",
        "acc_m",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "gyr_m",
        "mag_x",
        "mag_y",
        "mag_z",
        "mag_m",
    ]
'''
features_num = (int)(len(custom_feature_cols)/3)
# NAIVE_MINMAX (2D Filters)
# -----------------------------------------------------------
NAIVE_MINMAX_2D = ExperimentParameters(
    name="NAIVE-MINMAX-2D",
    # Data / Splitting
    frequency=100,
    feature_cols=[
        "acc_x",
        "acc_y",
        "acc_z",
        "gyr_x",
        "gyr_y",
        "gyr_z",
        "mag_x",
        "mag_y",
        "mag_z",
    ],
    max_subjects=90,
    exclude_subjects=[],
    n_valid_train_subjects=20,
    n_valid_test_subjects=20,
    n_test_train_subjects=20,
    n_test_test_subjects=30,
    seconds_per_subject_train=100,
    seconds_per_subject_test=100,
    task_types=None,
    # Reshaping
    window_size=50,  # 1 sec
    step_width=50,
    # Normalization
    scaler="robust_no_center",
    scaler_scope="subject",
    scaler_global=False,
    # Siamese Network
    model_variant="2d",
    filters=[32, 64, 128, 32],
    epochs_best=12,
    epochs_max=12,
    batch_size=128,
    optimizer="adam",
    optimizer_lr=0.002,
    optimizer_decay=0,
    max_pairs_per_session=60,  # => 4min
    margin=0.2,
    # OCSVM
    ocsvm_nu=0.092,
    ocsvm_gamma=1.151,
)  # <END NAIVE_APPROACH>


# ### 1.4 Select Approach <a id='1.4'>&nbsp;</a> 
# Select the parameters to use for current notebook execution here!

# In[67]:

#noted by humingming run different scene
#P = VALID_FCN_ROBUST
P = NAIVE_MINMAX_2D

# **Overview of current Experiment Parameters:**

# In[68]:


#utils_ppp(P)


# ## 2. Initial Data Preparation <a id='2'>&nbsp;</a> 

# ### 2.1 Load Dataset <a id='2.1'>&nbsp;</a> 
import time
from sklearn.manifold import TSNE

def plot_tsne_valid(df,index):
    # t-sne
    #tsne = TSNE(n_components=2, perplexity = 30, early_exaggeration = 12, init='random')
    tsne = TSNE(n_components=2, perplexity = 10, early_exaggeration = 4, learning_rate = 200, init='random')
    #tsne = TSNE(n_components=2, init='pca')
    test = np.stack(list(df["X"].values))
    print("the type of test is:", type(test))
    print("the shape of test is:", test.shape)
    #deep_transformed = pca.fit_transform(df.drop(columns=["subject", "session", "task_type"]).values)
    deep_transformed = tsne.fit_transform(test)
    # Create df with data needed for chart only
    df_viz = df.copy()
    df_viz["TSNE0"] = deep_transformed[:, 0]
    df_viz["TSNE1"] = deep_transformed[:, 1]
    df_viz.drop(
        columns=[c for c in df_viz.columns if c not in ["TSNE0", "TSNE1", "subject"]]
    )

    # Generate color index for every subject
    df_viz["Subject"] = pd.Categorical(df_viz["subject"])
    df_viz["colors"] = df_viz["Subject"].cat.codes
    print("the len is:", len(df_viz["Subject"].unique()))
    if len(df_viz["Subject"].unique()) <= 10:
        #pal = sns.color_palette("tab10")
        pal = sns.color_palette(n_colors=len(df_viz["Subject"].unique()))
    else:
        pal = sns.color_palette("tab20")   
    # Actual plot
    fig = plt.figure(figsize=(5.473 / 1.5, 5.473 / 2), dpi=180)
    sns.scatterplot(
        x="TSNE0",
        y="TSNE1",
        data=df_viz,
        hue="Subject",
        legend="full",
        palette=pal,
        s=3,
        linewidth=0,
        alpha=0.6,
    )
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=5)
    plt.title('t-SNE Visualization with GlobalMaxPooling2D layer features')
    plt.xlabel('')
    plt.ylabel('')
    
    fig.tight_layout()
    plt.show()
    localtime = time.asctime(time.localtime(time.time()))
    fig.savefig(f"./image/{index}-{localtime}tnse-train.png")
    #localtime = time.asctime(time.localtime(time.time()))
    #fig.savefig(REPORT_PATH / f"{localtime}-tnse-train.png")
    return plt


def plot_pca(df, index):
    # PCA
    pca = PCA(n_components=2, svd_solver='full')
    test = np.stack(list(df["X"].values))
    print("the type of test is:", type(test))
    print("the shape of test is:", test.shape)
    #deep_transformed = pca.fit_transform(df.drop(columns=["subject", "session", "task_type"]).values)
    deep_transformed = pca.fit_transform(test)
    # Create df with data needed for chart only
    df_viz = df.copy()
    df_viz["PCA0"] = deep_transformed[:, 0]
    df_viz["PCA1"] = deep_transformed[:, 1]
    df_viz.drop(
        columns=[c for c in df_viz.columns if c not in ["PCA0", "PCA1", "subject"]]
    )

    # Generate color index for every subject
    df_viz["Subject"] = pd.Categorical(df_viz["subject"])
    df_viz["colors"] = df_viz["Subject"].cat.codes
    print("the len is:", len(df_viz["Subject"].unique()))
    if len(df_viz["Subject"].unique()) <= 10:
        #pal = sns.color_palette("tab10")
        pal = sns.color_palette(n_colors=len(df_viz["Subject"].unique()))
    else:
        pal = sns.color_palette("tab20")   
    # Actual plot
    fig = plt.figure(figsize=(5.473 / 1.5, 5.473 / 2), dpi=180)
    sns.scatterplot(
        x="PCA0",
        y="PCA1",
        data=df_viz,
        hue="Subject",
        legend="full",
        palette=pal,
        s=3,
        linewidth=0,
        alpha=0.6,
    )
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=5)
    
    fig.tight_layout()
    #plt.show()
    localtime = time.asctime(time.localtime(time.time()))
    fig.savefig(f"./image/{index}-pca-train.png")
    return plt
'''
def plot_pca_test(x_test, y_test, index):
    # PCA
    pca = PCA(n_components=2, svd_solver='full')
    #pca = PCA(n_components=2)
    test = x_test
    #print("the type of test is:", type(test))
    #print("the shape of test is:", test.shape)
    #deep_transformed = pca.fit_transform(df.drop(columns=["subject", "session", "task_type"]).values)
    deep_transformed = pca.fit_transform(test)
    # Create df with data needed for chart only
    df_viz = pd.DataFrame()
    df_viz["PCA0"] = deep_transformed[:, 0]
    df_viz["PCA1"] = deep_transformed[:, 1]
    
    # Generate color index for every subject
    df_viz["subject"] = y_test
    df_viz["Subject"] = pd.Categorical(df_viz["subject"])
    df_viz["colors"] = df_viz["Subject"].cat.codes
    #print("the len is:", len(df_viz["Subject"].unique()))
    if len(df_viz["Subject"].unique()) <= 10:
        #pal = sns.color_palette("tab10")
        pal = sns.color_palette(n_colors=len(df_viz["Subject"].unique()))
    else:
        pal = sns.color_palette("tab20")   
    # Actual plot
    fig = plt.figure(figsize=(5.473 / 1.5, 5.473 / 2), dpi=180)
    sns.scatterplot(
        x="PCA0",
        y="PCA1",
        data=df_viz,
        hue="Subject",
        legend="full",
        palette=pal,
        s=3,
        linewidth=0,
        alpha=0.6,
    )
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=5)
    
    fig.tight_layout()
    #plt.show()
    #localtime = time.asctime(time.localtime(time.time()))
    fig.savefig(f"./image/{index}-pca-train.png")
    return plt
'''

def plot_pca_test(x_test, y_test, index):
    # PCA
    pca = PCA(n_components=2, svd_solver='full')
    #pca = PCA(n_components=2)
    test = x_test
    #print("the type of test is:", type(test))
    #print("the shape of test is:", test.shape)
    #deep_transformed = pca.fit_transform(df.drop(columns=["subject", "session", "task_type"]).values)
    deep_transformed = pca.fit_transform(test)
    # Create df with data needed for chart only
    df_viz = pd.DataFrame()
    df_viz["PCA0"] = deep_transformed[:, 0]
    df_viz["PCA1"] = deep_transformed[:, 1]
    '''
    df_viz.drop(
        columns=[c for c in df_viz.columns if c not in ["PCA0", "PCA1", "subject"]]
    )
    '''
    # Generate color index for every subject
    y_label = []
    for i in range(len(y_test)):
        if y_test[i] == 1:
            y_label.append("Normal")
        else:
            y_label.append("Abnormal")
    #df_viz["subject"] = y_test
    df_viz["subject"] = y_label
    df_viz["Subject"] = pd.Categorical(df_viz["subject"])
    df_viz["colors"] = df_viz["Subject"].cat.codes
    #print("the len is:", len(df_viz["Subject"].unique()))
    if len(df_viz["Subject"].unique()) <= 10:
        #pal = sns.color_palette("tab10")
        pal = sns.color_palette(n_colors=len(df_viz["Subject"].unique()))
    else:
        pal = sns.color_palette("tab20")   
    # Actual plot
    fig = plt.figure(figsize=(5.473 / 1.5, 5.473 / 2), dpi=180)
    sns.scatterplot(
        x="PCA0",
        y="PCA1",
        data=df_viz,
        hue="Subject",
        legend="full",
        palette=pal,
        s=3,
        linewidth=0,
        alpha=0.6,
    )
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=5)
    
    fig.tight_layout()
    #plt.show()
    #localtime = time.asctime(time.localtime(time.time()))
    fig.savefig(f"./image/{index}-pca-train.png")
    return plt


def plot_tsne(features, label, index):
    # t-sne
    tsne = TSNE(n_components=2, perplexity = 17, early_exaggeration = 1)
    #test = np.stack(list(df["X"].values))
    print("the type of test is:", type(features))
    print("the shape of test is:", features.shape)
    #deep_transformed = pca.fit_transform(df.drop(columns=["subject", "session", "task_type"]).values)
    deep_transformed = tsne.fit_transform(features)
    # Create df with data needed for chart only
    df_viz = pd.DataFrame() 
    df_viz["TSNE0"] = deep_transformed[:, 0]
    df_viz["TSNE1"] = deep_transformed[:, 1]
    df_viz["subject"] = label
    '''
    df_viz.drop(
        columns=[c for c in df_viz.columns if c not in ["TSNE0", "TSNE1", "subject"]]
    )
    '''
    # Generate color index for every subject
    df_viz["Subject"] = pd.Categorical(df_viz["subject"])
    df_viz["colors"] = df_viz["Subject"].cat.codes
    print("the len is:", len(df_viz["Subject"].unique()))
    if len(df_viz["Subject"].unique()) <= 10:
        #pal = sns.color_palette("tab10")
        pal = sns.color_palette(n_colors=len(df_viz["Subject"].unique()))
    else:
        pal = sns.color_palette("tab20")   
    # Actual plot
    fig = plt.figure(figsize=(5.473 / 1.5, 5.473 / 2), dpi=180)
    sns.scatterplot(
        x="TSNE0",
        y="TSNE1",
        data=df_viz,
        hue="Subject",
        legend="full",
        palette=pal,
        s=3,
        linewidth=0,
        alpha=0.6,
    )
 
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0, fontsize=5)
    plt.title('t-SNE Visualization with GlobalMaxPooling2D layer features')
    plt.xlabel('')
    plt.ylabel('')
    
    fig.tight_layout()
    #plt.show()
    #localtime = time.asctime(time.localtime(time.time()))
    fig.savefig(f"./image/{index}-tnse-train.png")
    return plt

# In[7]:
def construct_dataset():
    #50-10-50HZmul_feature_data_normalize
    
    #df_data = pd.read_csv('./nagiva_dataset.csv', sep=',')
    df_data = pd.read_csv('./hmog_raw_feature_data_new.csv', sep=',')
    #df_data = pd.read_csv('./20-10-100HZhmog_mul_feature_data_normalize.csv', sep=',')
    #df_data = pd.read_csv('./10-5-100HZhmog_mul_feature_data_normalize.csv', sep=',')
    #df_data = pd.read_csv('./6-3-100HZnagiva_mul_feature_data_normalize.csv', sep=',')
    #df_data = pd.read_csv('./50-10-50HZmul_feature_data_normalize.csv', sep=',')
    #df_data = pd.read_csv('./25-20-100HZmul_feature_data_normalize.csv', sep=',')
    #df_data = pd.read_csv('./100-50-mul_feature_data_normalize.csv', sep=',')
    #print("the subject num is:", len(list(df_data["subject"].unique())))
    
    hmog = DatasetLoader(
        hdf5_file=HMOG_HDF5,
        table_name=P.table_name,
        max_subjects=P.max_subjects,
        task_types=P.task_types,
        exclude_subjects=P.exclude_subjects,
        exclude_cols=EXCLUDE_COLS,
        seed=SEED,
    )
    hmog.data_summary()
    '''
    if not P.scaler_global:
        print("Normalize all data before splitting into train and test sets...")
        hmog.all, scalers = utils_custom_scale(
            hmog.all,
            scale_cols=P.feature_cols,        
            feature_cols=P.feature_cols,
            scaler_name="minmax",
            scope=P.scaler_scope,
            plot=False,
        )
    else:
        print("Skipped, normalize after splitting.")
    
    df = DataframeGenerateRaw(hmog.all)
    del hmog.all
    #acc

    print("the head of df is:", df.head(2))
    subject_name = list(df["subject"].unique())
    print("the num of subject is:", len(subject_name))
    print(subject_name)
    df.to_csv('./hmog_raw_feature_data_new.csv', sep=',', header=True, index=False)
    print("data write finished")
    exit(-1)
    '''
    hmog.all = df_data
    
    del df_data
    #print("the hmog.all shape is:", hmog.all.shape)
    
    # ### 2.2 Normalize Features (if global) <a id='2.2'>&nbsp;</a> 
    # Used here for naive approach (before splitting into test and training sets). Otherwise it's used during generate_pairs() and respects train vs. test borders.
    
    
    if not P.scaler_global:
        print("Normalize all data before splitting into train and test sets...")
        hmog.all, scalers = utils_custom_scale(
            hmog.all,
            scale_cols=custom_feature_cols,        
            feature_cols=custom_feature_cols,
            scaler_name=P.scaler,
            scope=P.scaler_scope,
            plot=False,
        )
    else:
        print("Skipped, normalize after splitting.")
        
    
    hmog.all = utils_reshape_features(
        hmog.all,
        feature_cols=custom_feature_cols,
        window_size=P.window_size,
        step_width=P.step_width,
    )
    
    
    hmog.generate_train_valid_train_test_data(
        hmog.all,
        n_valid_train=P.n_valid_train_subjects,
        n_valid_test=P.n_valid_test_subjects,
        n_test_train=P.n_test_train_subjects,
        n_test_test=P.n_test_test_subjects,
    )
    '''
    hmog.split_train_valid_train_test_transfer(
        hmog.all,
        n_valid_train=P.n_valid_train_subjects,
        n_valid_test=P.n_valid_test_subjects,
        n_test_train=P.n_test_train_subjects,
        n_test_test=P.n_test_test_subjects,
    )
    '''
    hmog.data_summary()
    del hmog.all
    
    # ### 2.4 Normalize features (if not global) <a id='2.4'>&nbsp;</a> 
    
    # In[10]:
    
    
    if P.scaler_global:
        print("Scaling Data for Siamese Network only...")
        print("Training Data:")
        hmog.valid_train, _ = utils_custom_scale(
            hmog.valid_train,
            scale_cols=custom_feature_cols,
            feature_cols=custom_feature_cols,
            scaler_name=P.scaler,
            scope=P.scaler_scope,
            plot=False,        
        )
        print("Validation Data:")
        hmog.valid_test, _ = utils_custom_scale(
            hmog.valid_test,
            scale_cols=custom_feature_cols,        
            feature_cols=custom_feature_cols,
            scaler_name=P.scaler,
            scope=P.scaler_scope,
            plot=False,        
        )
        
        print("test train Data:")
        hmog.test_train, _ = utils_custom_scale(
            hmog.test_train,
            scale_cols=custom_feature_cols,        
            feature_cols=custom_feature_cols,
            scaler_name=P.scaler,
            scope=P.scaler_scope,
            plot=False,        
        )
        
        print("test test Data:")
        hmog.test_test, _ = utils_custom_scale(
            hmog.test_test,
            scale_cols=custom_feature_cols,        
            feature_cols=custom_feature_cols,
            scaler_name=P.scaler,
            scope=P.scaler_scope,
            plot=False,        
        )
    else:
        print("Skipped, already normalized.")    
    
    
    # ### 2.5 Check Splits <a id='2.5'>&nbsp;</a> 
    
    # In[11]:
    
    #print("the head of hmog.valid_train is:\n", hmog.valid_train.head(1))
    
    #utils_split_report(hmog.valid_train)
    
    
    # In[12]:
    
    
    #utils_split_report(hmog.valid_test)
    
    
    # In[13]:
    
    
    #utils_split_report(hmog.test_train)
    
    
    # In[14]:
    
    
    #utils_split_report(hmog.test_test)
    
    
    # ### 2.6 Reshape Features  <a id='2.6'>&nbsp;</a> 
    
    # **Reshape & cache Set for Training Siamese Network:**
    
    # In[15]:
   
    df_siamese_train = hmog.valid_train
    '''
    df_siamese_train = utils_reshape_features(
        hmog.valid_train,
        feature_cols=custom_feature_cols,
        window_size=P.window_size,
        step_width=P.step_width,
    )
    '''

    # Clean memory
    del hmog.valid_train
    #get_ipython().magic('reset_selective -f hmog.train')
    '''
    print("Validation data after reshaping:")
    display(df_siamese_train.head())
    print("the shape of validation data X is:", type(df_siamese_train["X"]))
    print("the head of validation data X is:", np.array(df_siamese_train["X"][0]).shape)
    '''
    # Store iterim data
    #HMM0529
    #df_siamese_train.to_msgpack(OUTPUT_PATH / "df_siamese_train.msg")
    df_siamese_train.to_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_train.p")
    del df_siamese_train
    # Clean memory
    #get_ipython().magic('reset_selective -f df_siamese_train')
    
    
    # **Reshape & cache Set for Validating Siamese Network:** (also used to optimize OCSVM)
    
    # In[16]:
    
    df_siamese_valid = hmog.valid_test
    '''
    df_siamese_valid = utils_reshape_features(
        hmog.valid_test,
        feature_cols=custom_feature_cols,
        window_size=P.window_size,
        step_width=P.step_width,
    )
    '''
    
    del hmog.valid_test
    #get_ipython().magic('reset_selective -f hmog.valid')
    
    #print("Testing data after reshaping:")
    display(df_siamese_valid.head())
    
    # Store iterim data
    #HMM0529
    #df_siamese_valid.to_msgpack(OUTPUT_PATH / "df_siamese_valid.msg")
    df_siamese_valid.to_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_valid.p")
    del df_siamese_valid
    # Clean memory
    #get_ipython().magic('reset_selective -f df_siamese_valid')
    
    
    # **Reshape & cache Set for Training/Validation OCSVM:**
    
    # In[17]:
    
    df_ocsvm_train_valid = hmog.test_train
    '''
    df_ocsvm_train_valid = utils_reshape_features(
        hmog.test_train,
        feature_cols=custom_feature_cols,
        window_size=P.window_size,
        step_width=P.step_width,
    )
    '''
    del hmog.test_train
    #get_ipython().magic('reset_selective -f hmog.test_train')
    
    #print("Testing data after reshaping:")
    display(df_ocsvm_train_valid.head())
    
    # Store iterim data
    #HMM0529
    #df_ocsvm_train_valid.to_msgpack(OUTPUT_PATH / "df_ocsvm_train_valid.msg")
    df_ocsvm_train_valid.to_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_ocsvm_train_valid.p")
    #del df_ocsvm_train_valid
    # Clean memory
    #get_ipython().magic('reset_selective -f df_ocsvm_train_valid')
    
    
    # **Reshape & cache Set for Training/Testing OCSVM:**
    
    # In[18]:
    
    df_ocsvm_train_test = hmog.test_test
    '''
    df_ocsvm_train_test = utils_reshape_features(
        hmog.test_test,
        feature_cols=custom_feature_cols,
        window_size=P.window_size,
        step_width=P.step_width,
    )
    '''
    del hmog.test_test
   # print("the len of df_ocsvm_train_test is:", len(df_ocsvm_train_test))
    #get_ipython().magic('reset_selective -f hmog.test_test')
    
    #print("Testing data after reshaping:")
    display(df_ocsvm_train_test.head())
    
    # Store iterim data
    #HMM
    #df_ocsvm_train_test.to_msgpack(OUTPUT_PATH / "df_ocsvm_train_test.msg")
    df_ocsvm_train_test.to_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_ocsvm_train_test.p")
    #del df_ocsvm_train_test
    # Clean memory
    #get_ipython().magic('reset_selective -f df_ocsvm_train_test')
    #get_ipython().magic('reset_selective -f df_')
    
    
    # ## 3. Generate Scenario Pairs <a id='3'>&nbsp;</a> 
    
    # ### 3.1 Load cached Data <a id='3.1'>&nbsp;</a> 
    
    # In[19]:
    #HMM0529
    '''
    df_siamese_train = pd.read_msgpack(OUTPUT_PATH / "df_siamese_train.msg")
    df_siamese_valid = pd.read_msgpack(OUTPUT_PATH / "df_siamese_valid.msg")
    
    '''
    df_siamese_train = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_train.p")
    df_siamese_valid = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_valid.p")
    
    
    # ### 3.2 Build positive/negative Pairs  <a id='3.2'>&nbsp;</a> 
    
    # In[20]:
    
    
    def prep_X_y_pair_dataset(df):
        
        X_left = np.stack(list(df["left_X"].values))
        X_right = np.stack(list(df["right_X"].values))
        
        X = [X_left, X_right]
        y = df["label"].values
        
        return X, y
    
    
    def prep_X_y_triplet_dataset(df):
        
        X = np.stack(list(df["X"].values))
    
        y = df["subject"].values
        return X, y
    
  
    
    #print("the head of train is:", df_siamese_train.head(5))
    df_ocsvm_train_valid = pd.concat([df_ocsvm_train_test, df_ocsvm_train_valid],axis=0)
    df_siamese_train = pd.concat([df_siamese_train, df_siamese_valid],axis=0)
    
    subjects = df_siamese_train["subject"].unique().tolist()
    print("the num of df_siamese_train subject is:\n", len(subjects))
    subjects = df_ocsvm_train_valid["subject"].unique().tolist()
    print("the num of df_ocsvm_train_valid subject is:\n", len(subjects))
    
    X_train, y_train = prep_X_y_triplet_dataset(df_siamese_train)
    #X_train = X_train[:,:8,:]
    X_valid, y_valid = prep_X_y_triplet_dataset(df_ocsvm_train_valid)
    #X_valid = X_valid[:,:8,:]
    
    # 2D Filter Model needs flat 4th dimension
    
    #if P.model_variant == "2d":
    if P.model_variant == "3d":
        train_acc_a = X_train[:,:,np.arange(0,features_num)]
        train_gry_a = X_train[:,:,np.arange(features_num,2*features_num)]
        train_mag_a = X_train[:,:,np.arange(2*features_num,3*features_num)]
        
        
        
        train_acc_a = train_acc_a.reshape((*train_acc_a.shape,1))
        train_gry_a = train_gry_a.reshape((*train_gry_a.shape,1))
        train_mag_a = train_mag_a.reshape((*train_mag_a.shape,1))
        
        
        X_train = np.concatenate([train_acc_a, train_gry_a, train_mag_a], axis=-1)
        
    
        valid_acc_a = X_valid[:,:,np.arange(0,features_num)]
        valid_gry_a = X_valid[:,:,np.arange(features_num,2*features_num)]
        valid_mag_a = X_valid[:,:,np.arange(2*features_num,3*features_num)]
      
        
        valid_acc_a = valid_acc_a.reshape((*valid_acc_a.shape,1))
        valid_gry_a = valid_gry_a.reshape((*valid_gry_a.shape,1))
        valid_mag_a = valid_mag_a.reshape((*valid_mag_a.shape,1))
       
        X_valid = np.concatenate([valid_acc_a, valid_gry_a, valid_mag_a], axis=-1)
    
    
    #print("the X_train shape is:", X_train.shape)
    #print("the X_valid shape is:", X_valid.shape)
    #print("the X_train is:", X_train[0])
    #print("the y_train shape is:", y_train.shape)
    #paddingData = X_train.shape[1]-X_train.shape[2]
    
    #X_train = np.pad(X_train, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    #X_valid = np.pad(X_valid, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    '''
    X_train = np.pad(X_train, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    X_valid = np.pad(X_valid, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    '''
    #X_train = X_train.transpose((0,3,1,2))
    #X_valid = X_valid.transpose((0,3,1,2))
    X_train = X_train.reshape((*X_train.shape,1))
    X_valid = X_valid.reshape((*X_valid.shape,1))
    X_train = X_train.transpose((0,3,2,1))
    X_valid = X_valid.transpose((0,3,2,1))
    
    
    #print("the X_train is:", X_train[0])
    print("the X_train shape is:", X_train.shape)
    print("the X_valid shape is:", X_valid.shape)
    
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train.shape)
    '''
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_valid = label_encoder.fit_transform(y_valid)
    
    #print("the y_train is:", y_train[:1000])
    #print("the y_valid[0:10] is:", y_valid[:500][:])
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train[:500][:])
    '''
    return X_train, y_train, X_valid, y_valid

features_num_ = 23
def utils_generate_deep_features(df, model):
    # Predict deep features
    X = np.stack(list(df["X"].values))
    
    # 2D Filter Model needs flat 4th dimension
   
    acc_a = X[:,:,np.arange(0,features_num_)]
    gry_a = X[:,:,np.arange(features_num_,2*features_num_)]
    mag_a = X[:,:,np.arange(2*features_num_,3*features_num_)]
    acc_a = acc_a.reshape((*acc_a.shape,1))
    gry_a = gry_a.reshape((*gry_a.shape,1))
    mag_a = mag_a.reshape((*mag_a.shape,1))
    X = np.concatenate([acc_a, gry_a, mag_a], axis=-1)
    X = np.pad(X, [(0, 0), (5, 4), (21, 20), (0,0)], 'constant')
    
    

    # Overwrite original features
    #df["X"] = [list(vect[256:512]) for vect in X_pred]
    df["X"] = [list(vect) for vect in X_pred]

    return df.copy()


def getDataset(train=True):
    x_train, y_train, x_valid, y_valid = construct_dataset()
    if train:
        return TensorDataset(x_train,y_train)
    else:
        return TensorDataset(x_valid,y_valid)
        


def read_data():
    
    df_ocsvm_train_valid = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_ocsvm_train_valid.p")
    df_ocsvm_train_test = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_ocsvm_train_test.p")
    
    # ### 3.2 Build positive/negative Pairs  <a id='3.2'>&nbsp;</a> 
    
    # In[20]:
    
    
    def prep_X_y_pair_dataset(df):
        
        X_left = np.stack(list(df["left_X"].values))
        X_right = np.stack(list(df["right_X"].values))
        
        X = [X_left, X_right]
        y = df["label"].values
        
        return X, y
    
    
    def prep_X_y_triplet_dataset(df):
        
        X = np.stack(list(df["X"].values))
    
        y = df["subject"].values
        return X, y
    
  
    
    #print("the head of train is:", df_siamese_train.head(5))
    df_ocsvm_train_valid1 = pd.concat([df_ocsvm_train_test, df_ocsvm_train_valid],axis=0)
    
    del df_ocsvm_train_valid
    del df_ocsvm_train_test
    df_siamese_train = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_train.p")
    df_siamese_valid = pd.read_pickle("/home/tremb1e/work/mobilephone/RobustNoCenterDatasetNew/df_siamese_valid.p")
    df_siamese_train1 = pd.concat([df_siamese_train, df_siamese_valid],axis=0)
    del df_siamese_train
    del df_siamese_valid
    subjects = df_siamese_train1["subject"].unique().tolist()
    print("the num of df_siamese_train subject is:\n", len(subjects))
    subjects = df_ocsvm_train_valid1["subject"].unique().tolist()
    print("the num of df_ocsvm_train_valid subject is:\n", len(subjects))
    
    X_train, y_train = prep_X_y_triplet_dataset(df_siamese_train1)
    #X_train = X_train[:,:,:8]
    del df_siamese_train1
    X_valid, y_valid = prep_X_y_triplet_dataset(df_ocsvm_train_valid1)
    #X_valid = X_valid[:,:,:8]
    del df_ocsvm_train_valid1
    # 2D Filter Model needs flat 4th dimension
    
    #if P.model_variant == "2d":
    if P.model_variant == "3d":
        train_acc_a = X_train[:,:,np.arange(0,features_num)]
        train_gry_a = X_train[:,:,np.arange(features_num,2*features_num)]
        train_mag_a = X_train[:,:,np.arange(2*features_num,3*features_num)]
        
        
        
        train_acc_a = train_acc_a.reshape((*train_acc_a.shape,1))
        train_gry_a = train_gry_a.reshape((*train_gry_a.shape,1))
        train_mag_a = train_mag_a.reshape((*train_mag_a.shape,1))
        
        
        X_train = np.concatenate([train_acc_a, train_gry_a, train_mag_a], axis=-1)
        
    
        valid_acc_a = X_valid[:,:,np.arange(0,features_num)]
        valid_gry_a = X_valid[:,:,np.arange(features_num,2*features_num)]
        valid_mag_a = X_valid[:,:,np.arange(2*features_num,3*features_num)]
      
        
        valid_acc_a = valid_acc_a.reshape((*valid_acc_a.shape,1))
        valid_gry_a = valid_gry_a.reshape((*valid_gry_a.shape,1))
        valid_mag_a = valid_mag_a.reshape((*valid_mag_a.shape,1))
       
        X_valid = np.concatenate([valid_acc_a, valid_gry_a, valid_mag_a], axis=-1)
    
    
    #print("the X_train shape is:", X_train.shape)
    #print("the X_valid shape is:", X_valid.shape)
    #print("the X_train is:", X_train[0])
    #print("the y_train shape is:", y_train.shape)
    #paddingData = X_train.shape[1]-X_train.shape[2]
    
    #X_train = np.pad(X_train, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    #X_valid = np.pad(X_valid, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    '''
    X_train = np.pad(X_train, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    X_valid = np.pad(X_valid, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    '''
    #X_train = X_train.transpose((0,3,1,2))
    #X_valid = X_valid.transpose((0,3,1,2))
    X_train = X_train.reshape((*X_train.shape,1))
    X_valid = X_valid.reshape((*X_valid.shape,1))
    X_train = X_train.transpose((0,3,2,1))
    X_valid = X_valid.transpose((0,3,2,1))
    
    
    #print("the X_train is:", X_train[0])
    #print("the X_train shape is:", X_train.shape)
    #print("the X_valid shape is:", X_valid.shape)
    
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train.shape)
    '''
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_valid = label_encoder.fit_transform(y_valid)
    
    #print("the y_train is:", y_train[:1000])
    #print("the y_valid[0:10] is:", y_valid[:500][:])
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train[:500][:])
    '''
 
    
    return X_train, y_train, X_valid, y_valid, subjects


def read_data_attack():
    
    df_ocsvm_train_valid = pd.read_pickle("./WisdmDataset/df_ocsvm_train_valid.p")
    df_ocsvm_train_test = pd.read_pickle("./WisdmDataset/df_ocsvm_train_test.p")
    
    # ### 3.2 Build positive/negative Pairs  <a id='3.2'>&nbsp;</a> 
    
    # In[20]:
    
    
    def prep_X_y_pair_dataset(df):
        
        X_left = np.stack(list(df["left_X"].values))
        X_right = np.stack(list(df["right_X"].values))
        
        X = [X_left, X_right]
        y = df["label"].values
        
        return X, y
    
    
    def prep_X_y_triplet_dataset(df):
        
        X = np.stack(list(df["X"].values))
    
        y = df["subject"].values
        label_list = df["subject"].unique().tolist()
        label_list.sort()
        '''
        print("-------------------------the subjects is:\n", label_list)
        
        print("-------------------------the subjects id is:\n", label_list[1], label_list[3], label_list[7], label_list[11], label_list[12], label_list[23], label_list[28], label_list[40], label_list[42],        label_list[45])
        
        print("\n")
        '''
        return X, y
    
  
    
    #print("the head of train is:", df_siamese_train.head(5))
    df_ocsvm_train_valid1 = pd.concat([df_ocsvm_train_test, df_ocsvm_train_valid],axis=0)
    
    del df_ocsvm_train_valid
    del df_ocsvm_train_test
    df_siamese_train = pd.read_pickle("./WisdmDataset/df_siamese_train.p")
    df_siamese_valid = pd.read_pickle("./WisdmDataset/df_siamese_valid.p")
    df_siamese_train1 = pd.concat([df_siamese_train, df_siamese_valid],axis=0)
    del df_siamese_train
    del df_siamese_valid
    subjects = df_siamese_train1["subject"].unique().tolist()
    print("the num of df_siamese_train subject is:\n", len(subjects))
    subjects = df_ocsvm_train_valid1["subject"].unique().tolist()
    print("the num of df_ocsvm_train_valid subject is:\n", len(subjects))
    
    X_train, y_train = prep_X_y_triplet_dataset(df_siamese_train1)
    X_train = X_train[:,:,:8]
    del df_siamese_train1
    X_valid, y_valid = prep_X_y_triplet_dataset(df_ocsvm_train_valid1)
    X_valid = X_valid[:,:,:8]
    del df_ocsvm_train_valid1
    # 2D Filter Model needs flat 4th dimension
    
    #if P.model_variant == "2d":
    if P.model_variant == "3d":
        train_acc_a = X_train[:,:,np.arange(0,features_num)]
        train_gry_a = X_train[:,:,np.arange(features_num,2*features_num)]
        train_mag_a = X_train[:,:,np.arange(2*features_num,3*features_num)]
        
        
        
        train_acc_a = train_acc_a.reshape((*train_acc_a.shape,1))
        train_gry_a = train_gry_a.reshape((*train_gry_a.shape,1))
        train_mag_a = train_mag_a.reshape((*train_mag_a.shape,1))
        
        
        X_train = np.concatenate([train_acc_a, train_gry_a, train_mag_a], axis=-1)
        
    
        valid_acc_a = X_valid[:,:,np.arange(0,features_num)]
        valid_gry_a = X_valid[:,:,np.arange(features_num,2*features_num)]
        valid_mag_a = X_valid[:,:,np.arange(2*features_num,3*features_num)]
      
        
        valid_acc_a = valid_acc_a.reshape((*valid_acc_a.shape,1))
        valid_gry_a = valid_gry_a.reshape((*valid_gry_a.shape,1))
        valid_mag_a = valid_mag_a.reshape((*valid_mag_a.shape,1))
       
        X_valid = np.concatenate([valid_acc_a, valid_gry_a, valid_mag_a], axis=-1)
    
    
    #print("the X_train shape is:", X_train.shape)
    #print("the X_valid shape is:", X_valid.shape)
    #print("the X_train is:", X_train[0])
    #print("the y_train shape is:", y_train.shape)
    #paddingData = X_train.shape[1]-X_train.shape[2]
    
    #X_train = np.pad(X_train, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    #X_valid = np.pad(X_valid, [(0, 0), (0, 0), (5, 4), (0,0)], 'constant')
    '''
    X_train = np.pad(X_train, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    X_valid = np.pad(X_valid, [(0, 0), (0, 0), (0, paddingData), (0,0)], 'constant')
    '''
    #X_train = X_train.transpose((0,3,1,2))
    #X_valid = X_valid.transpose((0,3,1,2))
    X_train = X_train.reshape((*X_train.shape,1))
    X_valid = X_valid.reshape((*X_valid.shape,1))
    X_train = X_train.transpose((0,3,2,1))
    X_valid = X_valid.transpose((0,3,2,1))
    
    
    #print("the X_train is:", X_train[0])
    print("the X_train shape is:", X_train.shape)
    print("the X_valid shape is:", X_valid.shape)
    
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train.shape)
    '''
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_valid = label_encoder.fit_transform(y_valid)
    
    #print("the y_train is:", y_train[:1000])
    #print("the y_valid[0:10] is:", y_valid[:500][:])
    '''
    y_train = pd.get_dummies(y_train)
    print("the y_train[0:10] is:", y_train[:500][:])
    '''
 
    
    return X_train, y_train, X_valid, y_valid


def get_mnist_con(cls=0):
    x_train, y_train, x_valid, y_valid = read_data()
    #x_train, y_train, x_valid, y_valid = construct_dataset()
    #x_train = x_train.transpose((0,2,3,1))
    #x_valid = x_valid.transpose((0,2,3,1))
    #print("the shape of x_train is:", x_train.shape)
    #print("the shape of y_train is:", set(y_train))
    #d_train, d_test = keras.datasets.mnist.load_data()
    batch_size = 16
    
    
    valid_len = len(x_valid)//batch_size
    x_test, y_test = x_valid[:int(valid_len*batch_size)], y_valid[:int(valid_len*batch_size)]
    mask_test_true = y_valid==cls
    x_plot, y_plot = x_valid[mask_test_true], y_valid[mask_test_true]
    cls_false = cls+2
    mask_test_false = y_valid==cls_false
    x_plot = np.vstack((x_plot, x_valid[mask_test_false]))
    y_plot = np.concatenate((y_plot, y_valid[mask_test_false]),axis=0)
    plot_len = len(x_plot)//batch_size
    x_plot = x_plot[:int(plot_len*batch_size)]
    y_plot = y_plot[:int(plot_len*batch_size)]
    mask = y_train == cls

    x_train = x_train[mask]
    y_train = y_train[mask]
    train_len = len(x_train)//batch_size
    
    x_train, y_train = x_train[:int(train_len*batch_size)], y_train[:int(train_len*batch_size)]
    
    #print("the x_train is:", x_train[0][:,:,0])
    #x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    #x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)
    #print("the x_train shape is:", x_train.shape)
    #y_test_b = (y_test == cls)
    y_test = (y_test == cls).astype(np.float32)
    y_plot = (y_plot == cls).astype(np.float32)
    y_train = (y_train == cls).astype(np.float32)
    #print("the default data type is:", torch.get_default_dtype())
    #print("the type of get is:", torch.from_numpy(x_train).to(dtype=torch.Double)[0,0,0,0])
    return x_train, y_train, x_plot, y_plot, x_plot, y_plot  # y_test: normal -> 1 / abnormal -> 0


def get_mnist_con_another(cls=0):
    x_train, y_train, x_valid, y_valid, subjects = read_data()
    #x_train, y_train, x_valid, y_valid = construct_dataset()
    #exit(-1)
    batch_size = 16
    '''
    x_train = x_train.transpose((0,2,3,1))
    x_valid = x_valid.transpose((0,2,3,1))
    '''
    #print("the shape of x_train is:", x_train.shape)
    #print("the shape of y_train is:", set(y_train))
    #d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = x_train, y_train
    
    x_test, y_test = x_valid, y_valid
    mask_test_true = y_valid==cls
    x_plot, y_plot = x_valid[mask_test_true], y_valid[mask_test_true]
    for i in range(90):
        if i != cls:
            mask_test_false = y_valid==i
            #print("the type of mask_test_false is:", type(mask_test_false))
            #print("the mask_test_false[0] is:", mask_test_false[0])
            #mask_test_false = random.sample(mask_test_false, 50)
            x_plot = np.vstack((x_plot, x_valid[mask_test_false][0:60]))
            
            #print("the shape of y_test is:", y_test.shape)
            #print("the shape of y_valid is:", y_valid[mask_test_false].shape)
            #y_test = np.vstack((y_test, y_valid[mask_test_false]))
            y_plot = np.concatenate((y_plot, y_valid[mask_test_false][0:60]),axis=0)

    mask = y_train == cls

    
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    train_len = len(x_train)//batch_size
    
    x_train, y_train = x_train[:int(train_len*batch_size)], y_train[:int(train_len*batch_size)]
    
    #print("the shape of y_train_true is:", y_valid[mask_test].shape)
    #print("the shape of x_train is:", x_train.shape)
    #print("the x_train is:", x_train[0][:,:,0])
    #x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    #x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)
    #print("the x_train shape is:", x_train.shape)
    #y_test_b = (y_test == cls)
    plot_len = len(x_plot)//batch_size
    x_plot = x_plot[:int(plot_len*batch_size)]
    y_plot = y_plot[:int(plot_len*batch_size)]
    
    #y_train = (y_test == cls).astype(np.float32)
    y_train = (y_train == cls).astype(np.float32)
    y_test = (y_test == cls).astype(np.float32)
    y_plot = (y_plot == cls).astype(np.float32)
    
    return x_train, y_train, x_valid, y_valid, x_plot, y_plot, subjects  # y_test: normal -> 1 / abnormal -> 0


def get_mnist_con_augdata(cls, x_train, y_train, x_valid, y_valid, subjects):
    #x_train, y_train, x_valid, y_valid, subjects = read_data()
    #x_train, y_train, x_valid, y_valid = construct_dataset()
    #exit(-1)
    batch_size = 16
    '''
    x_train = x_train.transpose((0,2,3,1))
    x_valid = x_valid.transpose((0,2,3,1))
    '''
    #print("the shape of x_train is:", x_train.shape)
    #print("the shape of y_train is:", set(y_train))
    #d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = x_train, y_train
    
    x_test, y_test = x_valid, y_valid
    mask_test_true = y_valid==cls
    x_plot, y_plot = x_valid[mask_test_true], y_valid[mask_test_true]
    for i in range(90):
        if i != cls:
            mask_test_false = y_valid==i
            #print("the type of mask_test_false is:", type(mask_test_false))
            #print("the mask_test_false[0] is:", mask_test_false[0])
            #mask_test_false = random.sample(mask_test_false, 50)
            x_plot = np.vstack((x_plot, x_valid[mask_test_false][0:60]))
            
            #print("the shape of y_test is:", y_test.shape)
            #print("the shape of y_valid is:", y_valid[mask_test_false].shape)
            #y_test = np.vstack((y_test, y_valid[mask_test_false]))
            y_plot = np.concatenate((y_plot, y_valid[mask_test_false][0:60]),axis=0)

    mask = y_train == cls

    
    x_train = x_train[mask]
    y_train = y_train[mask]

    length = 20
    #print("the length of x_train is1:", len(x_train))
    #data_tensor = torch.load("./Data/HMOG_user_" + str(cls) + ".pt")
    data_tensor = torch.load("./Data/HMOG_user_0.pt")
    #print("the length of data_tensor is2:", len(data_tensor))
    
    x_train = np.concatenate((x_train, data_tensor))
    #print("the length of x_train is2:", len(x_train))
    y_train = list(y_train)
    for i in range(10000):
            y_train.append(cls)

    y_train = np.array(y_train)
    
    train_len = len(x_train)//batch_size
    
    x_train, y_train = x_train[:int(train_len*batch_size)], y_train[:int(train_len*batch_size)]
    
    #print("the shape of y_train_true is:", y_valid[mask_test].shape)
    #print("the shape of x_train is:", x_train.shape)
    #print("the x_train is:", x_train[0][:,:,0])
    #x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    #x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)
    #print("the x_train shape is:", x_train.shape)
    #y_test_b = (y_test == cls)
    #print("the length of x_train is2:", len(x_plot))
    
    x_plot = np.concatenate((x_plot, data_tensor[8000:10000]))
    #print("the length of x_train is2:", len(x_plot))
    #exit(-1)
    y_plot = list(y_plot)
    for i in range(2000):
            y_plot.append(cls)

    y_plot = np.array(y_plot)
    
    plot_len = len(x_plot)//batch_size
    x_plot = x_plot[:int(plot_len*batch_size)]
    y_plot = y_plot[:int(plot_len*batch_size)]
    
    y_train = (y_test == cls).astype(np.float32)
    y_test = (y_test == cls).astype(np.float32)
    y_plot = (y_plot == cls).astype(np.float32)
    
    return x_train, y_train, x_plot, y_plot, subjects  # y_test: normal -> 1 / abnormal -> 0


def get_mnist_con_noaugdata(cls, x_train, y_train, x_valid, y_valid, subjects):
    #x_train, y_train, x_valid, y_valid, subjects = read_data()
    #x_train, y_train, x_valid, y_valid = construct_dataset()
    #exit(-1)
    batch_size = 16
    '''
    x_train = x_train.transpose((0,2,3,1))
    x_valid = x_valid.transpose((0,2,3,1))
    '''
    #print("the shape of x_train is:", x_train.shape)
    #print("the shape of y_train is:", set(y_train))
    #d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = x_train, y_train
    
    x_test, y_test = x_valid, y_valid
    mask_test_true = y_valid==cls
    x_plot, y_plot = x_valid[mask_test_true], y_valid[mask_test_true]
    for i in range(90):
        if i != cls:
            mask_test_false = y_valid==i
            #print("the type of mask_test_false is:", type(mask_test_false))
            #print("the mask_test_false[0] is:", mask_test_false[0])
            #mask_test_false = random.sample(mask_test_false, 50)
            x_plot = np.vstack((x_plot, x_valid[mask_test_false][0:60]))
            
            #print("the shape of y_test is:", y_test.shape)
            #print("the shape of y_valid is:", y_valid[mask_test_false].shape)
            #y_test = np.vstack((y_test, y_valid[mask_test_false]))
            y_plot = np.concatenate((y_plot, y_valid[mask_test_false][0:60]),axis=0)

    mask = y_train == cls

    
    x_train = x_train[mask]
    y_train = y_train[mask]
    '''
    length = 20
    #print("the length of x_train is1:", len(x_train))
    data_tensor = torch.load("./Data/HMOG_user_0.pt")
    #print("the length of data_tensor is2:", len(data_tensor))
    
    x_train = np.concatenate((x_train, data_tensor))
    #print("the length of x_train is2:", len(x_train))
    y_train = list(y_train)
    for i in range(10000):
            y_train.append(cls)

    y_train = np.array(y_train)
    '''
    train_len = len(x_train)//batch_size
    
    x_train, y_train = x_train[:int(train_len*batch_size)], y_train[:int(train_len*batch_size)]
    '''
    x_plot = np.concatenate((x_plot, data_tensor[8000:10000]))
    #print("the length of x_train is2:", len(x_plot))
    #exit(-1)
    y_plot = list(y_plot)
    for i in range(2000):
            y_plot.append(cls)

    y_plot = np.array(y_plot)
    '''
    plot_len = len(x_plot)//batch_size
    x_plot = x_plot[:int(plot_len*batch_size)]
    y_plot = y_plot[:int(plot_len*batch_size)]
    
    y_train = (y_test == cls).astype(np.float32)
    y_test = (y_test == cls).astype(np.float32)
    y_plot = (y_plot == cls).astype(np.float32)
    
    return x_train, y_train, x_plot, y_plot, subjects  # y_test: normal -> 1 / abnormal -> 0


def get_mnist_con_random_attack(cls=0):
    x_train, y_train, x_valid, y_valid, subjects = read_data()
    x_attack, y_attack, x_attack1, y_attack1 = read_data_attack()
    #x_train, y_train, x_valid, y_valid = construct_dataset()
    batch_size = 16
    '''
    x_train = x_train.transpose((0,2,3,1))
    x_valid = x_valid.transpose((0,2,3,1))
    '''
    print("the shape of x_train is:", x_train.shape)
    #print("the shape of y_train is:", set(y_train))
    #d_train, d_test = keras.datasets.mnist.load_data()
    x_train, y_train = x_train, y_train
    
    x_test, y_test = x_valid, y_valid
    mask_test_true = y_valid==cls
    print("the mask_test_true is:", len(mask_test_true))
    #x_plot, y_plot = x_valid, y_valid
    
    x_plot, y_plot = x_valid[mask_test_true], y_valid[mask_test_true]
    
    for i in range(51):
        #if i != cls:
        mask_test_false = y_attack==i
        #print("the type of mask_test_false is:", type(mask_test_false))
        #print("the mask_test_false[0] is:", mask_test_false[0])
        #mask_test_false = random.sample(mask_test_false, 50)
        x_plot = np.vstack((x_plot, x_attack[mask_test_false][0:50]))
        
        #print("the shape of y_test is:", y_test.shape)
        #print("the shape of y_valid is:", y_valid[mask_test_false].shape)
        #y_test = np.vstack((y_test, y_valid[mask_test_false]))
        y_plot = np.concatenate((y_plot, y_attack[mask_test_false][0:50]),axis=0)
    
    mask = y_train == cls
    print("the mask is:", len(mask))
    x_train = x_train[mask]
    y_train = y_train[mask]
    
    train_len = len(x_train)//batch_size
    
    x_train, y_train = x_train[:int(train_len*batch_size)], y_train[:int(train_len*batch_size)]
    
    #print("the shape of y_train_true is:", y_valid[mask_test].shape)
    #print("the shape of x_train is:", x_train.shape)
    #print("the x_train is:", x_train[0][:,:,0])
    #x_train = np.expand_dims(x_train / 255., axis=-1).astype(np.float32)
    #x_test = np.expand_dims(x_test / 255., axis=-1).astype(np.float32)
    #print("the x_train shape is:", x_train.shape)
    #y_test_b = (y_test == cls)
    plot_len = len(x_plot)//batch_size
    x_plot = x_plot[:int(plot_len*batch_size)]
    y_plot = y_plot[:int(plot_len*batch_size)]
    
    y_train = (y_test == cls).astype(np.float32)
    y_test = (y_test == cls).astype(np.float32)
    y_plot = (y_plot == cls).astype(np.float32)
    
    return x_train, y_train, x_plot, y_plot, x_plot, y_plot, subjects  # y_test: normal -> 1 / abnormal -> 0

class SensorDataSet(Dataset):
    def __init__(self, op="train", transform=None, user_num=0):
        #x_train, y_train, x_valid, y_valid = read_data()
        #x_train, y_train, x_valid, y_valid = construct_dataset()
        x_train, y_train, x_valid, y_valid, x_plot, y_plot = get_mnist_con_another(cls = user_num)
        if op == "train":
            
            self.x_data = x_train
            self.y_data = y_train
            self.length = x_train.shape[0]
        elif op == "test":
            self.x_data = x_valid
            self.y_data = y_valid
            self.length = x_valid.shape[0]
        elif op == "plot":
            self.x_data = x_plot
            self.y_data = y_plot
            self.length = x_plot.shape[0]
            
            
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.length

class SensorDataSet_pro(Dataset):
    def __init__(self, x_data1, y_data1, op="train", transform=None, user_num=0):
        #x_train, y_train, x_valid, y_valid = read_data()
        #x_train, y_train, x_valid, y_valid = construct_dataset()
    
        #exit(-1)
        #x_train, y_train, x_valid, y_valid, x_plot, y_plot = get_mnist_con_another(cls = user_num)
        self.x_data = x_data1
        self.y_data = y_data1
        self.length = x_data1.shape[0]
        '''
        if op == "train":
            
            self.x_data = x_data
            self.y_data = y_data
            self.length = x_data.shape[0]
        elif op == "test":
            self.x_data = x_valid
            self.y_data = y_valid
            self.length = x_valid.shape[0]
        elif op == "plot":
            self.x_data = x_plot
            self.y_data = y_plot
            self.length = x_plot.shape[0]
         '''   
            
    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]
    def __len__(self):
        return self.length




if __name__ == '__main__':
    construct_dataset()
