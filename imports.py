import itertools
from itertools import product
import distutils
import os
import pandas as pd
from tqdm import tqdm, trange
import numpy as np
import random
import copy
import pickle
import numpy as np
from argparse import ArgumentParser
import sys
import collections
import statistics 
import ast
import yaml
from glob import glob

import scanpy as sc

import anndata

import wandb

from sklearn.cluster import KMeans
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

# roc curve and auc
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot

import pytorch_lightning as pl
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import seed_everything
from torch.utils.data import random_split, DataLoader
import torchvision.models as models
from torch import nn, optim

import torch

from mc_lightning.utilities.utilities import tile_sampler, subsample_tiles

from mc_lightning.models.resnet.resnet_transforms import RGBTrainTransform, RGBEvalTransform, HSVTrainTransform, HSVEvalTransform
from mc_lightning.models.resnet.resnet_dataset import *
from mc_lightning.models.resnet.resnet_trainer import * 
from mc_lightning.models.resnet.resnet_module import * 

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import json

from tabulate import tabulate

import torchmetrics
from torchmetrics import SpearmanCorrcoef

from temperature_scaling.temperature_scaling import ModelWithTemperature

import staintools

import warnings

from p_tqdm import p_umap

smfx_lyr = nn.Softmax(dim=1)

import textwrap


import matplotlib.patheffects as path_effects
from statannot import statannot
# from statannot.statannot.statannot import add_stat_annotation

from utils import *