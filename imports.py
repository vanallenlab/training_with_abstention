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

# from pytorch_lightning.callbacks import Callback
# from pl_bolts.callbacks import PrintTableMetricsCallback

sys.path.append('/home/jupyter/mc_lightning')

import scanpy as sc

import anndata

# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions

# import keras


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

# import umap

# from guppy import hpy; 

import staintools

import warnings

from p_tqdm import p_umap

smfx_lyr = nn.Softmax(dim=1)
temp1_path = "/home/jupyter/LUAD/Lung/be template 1.png"
temp2_path = "/home/jupyter/LUAD/Lung/be template 2.png"
temp1 = staintools.read_image(temp1_path)
temp2 = staintools.read_image(temp2_path)
temp1 = staintools.LuminosityStandardizer.standardize(temp1)
temp2 = staintools.LuminosityStandardizer.standardize(temp2)

normalizer1 = staintools.StainNormalizer(method='macenko')
normalizer2 = staintools.StainNormalizer(method='macenko')

normalizer1.fit(temp1)
normalizer2.fit(temp2)


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import textwrap

# sns.set_theme(context = 'paper')
sns.set_style("ticks")
sns.set(font_scale=1.5, style = 'ticks')

import matplotlib.patheffects as path_effects
from statannot import statannot
# from statannot.statannot.statannot import add_stat_annotation

def add_median_labels(ax):
    lines = ax.get_lines()
    # determine number of lines per box (this varies with/without fliers)
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    # iterate over median lines
    for median in lines[4:len(lines):lines_per_box]:
        # display median value at center of median line
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1]-median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:.2f}', ha='center', va='center',
                       fontweight='bold', color='white')
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def stain_transform_images(path): # p_umap(stain_transform_images, total['full_path'], num_cpus = 6)
    img = path.split('/')[-1].split('.')[0]
    try:
        pil_file1 = pil_loader(path[:-len(img + '.png')] + img + '_t1' + '.png')
        pil_file2 = pil_loader(path[:-len(img + '.png')] + img + '_t2' + '.png')
    except Exception as e:
        print(e)
        to_transform = staintools.read_image(path)
        to_transform = staintools.LuminosityStandardizer.standardize(to_transform)
        try:
            img1 = normalizer1.transform(to_transform)
            img2 = normalizer2.transform(to_transform)
            pil_file1 = Image.fromarray(img1)
            pil_file2 = Image.fromarray(img2)
        except Exception as e:
            print(e)
            pil_file1 = Image.fromarray(to_transform)
            pil_file2 = Image.fromarray(to_transform)
        
        pil_file1.save(path[:-len(img + '.png')] + img + '_t1' + '.png', format="png")
        pil_file2.save(path[:-len(img + '.png')] + img + '_t2' + '.png', format="png")

        

def sample_tiles_n_slides(x, num_slides, num_tiles):
        slides_sorted_by_size = x.groupby(['slide_id'])['full_path'].nunique().sort_values(ascending=False).index
        slides = slides_sorted_by_size[:num_slides]
        # slides = np.random.choice(x['slide_id'].unique(), size = min(num_slides, len(x['slide_id'].unique())), replace = False)
        x = x[x['slide_id'].isin(slides)]
        
        print(x.groupby(['slide_id']).agg('count'))

        return x.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=False)).reset_index(drop = True)

def get_correct(results):
    results['p_tumor'] = results['preds']
    results['p_healthy'] = 1 - results['preds']
    results['pred_healthy'] = pd.cut(results['p_healthy'], 2, labels = False)
    results['correct'] = results['pred_healthy'] == results['healthy']

    return results

def run_embedding(encoding_array, annotation_df, n_pcs=50, n_neighbors=25, use_rapids=True, run_louvain=False):
    print('creating AnnData object...')
    adata = anndata.AnnData(encoding_array)
    adata.obs = annotation_df # annotate data
    if use_rapids:
        print('using rapids GPU implementation')
        method='rapids'
        flavor='rapids'
    else:
        method='umap'
        flavor='vtraag'
    # run pca
    if n_pcs is not None:
        sc.tl.pca(adata, svd_solver='arpack', n_comps=n_pcs)
        # get neighbor graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, method=method)
    else:
        # get neighbor graph
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs, use_rep='X', method=method)
    # get umap embedding
    # sc.tl.umap(adata, method=method)
    sc.tl.umap(adata)
    if run_louvain:
        # run louvain clustering
        sc.tl.louvain(adata, flavor=flavor)
    return adata

def analyze_be_exp(EXP_VERSION, TASK, NUM_CLASSES):
    infile = open("/home/jupyter/LUAD/Lung/embeddings/embeddings" + EXP_VERSION + ".p",'rb')
    adata = pickle.load(infile)
    infile.close()
    val_paths = adata.obs
    
    val_paths = val_paths[val_paths.columns.drop(list(val_paths.filter(regex='Unnamed')))]
    val_paths.drop(columns = ['x', 'y'], inplace = True)

    if TASK == 'healthy':
        val_paths['p_healthy'] = 1 - val_paths['preds']

    val_paths['pred_class'] = pd.cut(1 - val_paths['preds'], NUM_CLASSES, labels = False)
    val_paths['correct'] = (val_paths['pred_class'] == val_paths[TASK]).astype(int)
    adata.obs = val_paths
    
    print("## Distribution of predicted probabilities")
    val_paths.hist(column='preds', by = TASK)
    
    print("## Seeing the distribution between of accuracy by hosp")
    df = val_paths.groupby(['source_id', TASK]).agg(['mean'])    
    print(tabulate(df, headers = 'keys', tablefmt = 'plain'))
    sizes = val_paths.groupby(['source_id', TASK])['correct'].agg(['size'])    
    print(tabulate(sizes, headers = 'keys', tablefmt = 'plain'))
    
    sns.set(rc={'figure.figsize':(15.7,12.27)})
    sns.set_style("white")
    
    print("## UMAP of GT, preds and source sites")
    sc.pl.umap(adata, edges = False, add_outline = False, size = 100, projection = '2d', color = ['pred_class', TASK], palette="tab20c", color_map=mpl.cm.tab20c)
    sc.pl.umap(adata, edges = False, add_outline = False, size = 100, projection = '2d', color = ['source_id', 'correct'], palette="tab20c", color_map=mpl.cm.tab20c)

    return val_paths

def sample_tiles_n_slides_from_source(x, num_slides = 5, num_tiles = 100):
    return x.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x = x, num_slides = num_slides, num_tiles = num_tiles)).reset_index(drop = True)

def create_tp_vp(lp, GROUP_COL, train_size, random_state, label = 'healthy', num_classes = 2, replace = False):
    
    gss = GroupShuffleSplit(n_splits=100, train_size=train_size, random_state=random_state)
    splits = list(gss.split(X = list(range(len(lp))), groups = lp[GROUP_COL]))

    td_ctr = collections.Counter([])
    vd_ctr = collections.Counter([])

    splits_iterator = iter(splits)

    ct = 0
    while any([(x not in y) for (x, y) in itertools.product(list(range(num_classes)), [td_ctr, vd_ctr])]):
        ct += 1
        if ct > 11:
            sys.exit()
            break
        
        train_idx, val_idx = next(splits_iterator)

        print(train_idx, val_idx)

        #Create train and val paths
        train_paths, val_paths = lp.iloc[train_idx] , lp.iloc[val_idx]

        val_paths, train_paths = balance_labels(val_paths, label, replace = replace), balance_labels(train_paths, label, replace = replace)    

        #Adjusting for the path lenghts
        if len(val_paths) > len(train_paths):
            train_paths, val_paths = val_paths, train_paths

        print('#Final Distributions')
        print('val_paths', val_paths.groupby(['source_id', label]).nunique())
        print('train_paths', train_paths.groupby(['source_id', label]).nunique())
        
        td_ctr = collections.Counter(train_paths.healthy.values)
        vd_ctr = collections.Counter(val_paths.healthy.values)
        print('train distribution', td_ctr,  'val distribution', vd_ctr)
    
    return train_paths, val_paths

def preprocess_df(df,
                    num_tiles = 100, 
                    num_hosps = 'all', 
                    min_num_slides_p_hosp = 10, 
                    min_num_tiles_p_slide = 100, 
                    replace = False
                    ):
    
    #Get the case IDs for the patients
    df['case_id'] = df['slide_id'].str.slice(0, len('TCGA-44-6144'))
    #Get the IDs for the Sources
    df['source_id'] = df['slide_id'].str.slice(len('TCGA-'), len('TCGA-44'))

    #Throw out hospitals with less than some number of slides
    n_slides = df.groupby('source_id')['slide_id'].nunique()
    print('before filter by min_num_slides_p_hosp', n_slides)
    at_least_n_slides = n_slides[n_slides > min_num_slides_p_hosp]
    print('after filter by min_num_slides_p_hosp', n_slides)
    df = df[df['source_id'].isin(at_least_n_slides.index)]

    all_hosps_sorted = df.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False).index

    if num_hosps == 'all':
        n_largest_hosps = all_hosps_sorted[:len(all_hosps_sorted)]
    else:
        n_largest_hosps = all_hosps_sorted[:num_hosps]

    print(n_largest_hosps)

    df = df[df['source_id'].isin(n_largest_hosps)]
    
    if replace == False:
        #Throw out slides with less than some number of tiles
        filtered = df.groupby('slide_id')['full_path'].filter(lambda x: len(x) >= min_num_tiles_p_slide)
        df = df[df['full_path'].isin(filtered)]

    #Change the file paths
    df.full_path = df.full_path.str.replace('/mnt/disks/data_disk/', '/home/jupyter/')

    #return a certain number of tiles for each slide
    df = df.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=replace)).reset_index(drop = True)

    print('after filter by min_num_tiles_p_slide', df.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False).index)

    return df

def balance_labels(df, label_col, random_state=1, replace = False):

    #Check the distributions
    ctr = collections.Counter(df[label_col].values)
    print('before balancing, label distribution', ctr)

    #Select the same number of samples from each class
    if replace == False:
        num_p_class = min(ctr.values())
    else:
        num_p_class = max(ctr.values())

    print('min_class_num', num_p_class)

    df = pd.concat([
        df[df[label_col] == i].sample(n=num_p_class, replace = replace, random_state=random_state) for i in ctr.keys()           
    ])

    #Shuffle the validation set
    df = df.sample(frac=1.0, random_state=random_state)
    df.reset_index(drop = True, inplace = True)

    #Check the distributions
    ctr = collections.Counter(df[label_col].values)
    print('after balancing, label distribution', ctr)

    return df

model_choices = {

    'Bekind_indhe' : Bekind_indhe,
    'Bekind_str' : Bekind_str,
    'Bekind_sl_best' : Bekind_sl_best,
    'Bekind_indhe_cos_sim' : Bekind_indhe_cos_sim,
    'Bekind_indhe_dro' : Bekind_indhe_dro,
    'Bekind_indhe_dro_log' : Bekind_indhe_dro_log,
    'Bekind_indhe_dro_BOBW' : Bekind_indhe_dro_BOBW,
    'PretrainedResnet50FT_Best_DRO_1_over_n' : PretrainedResnet50FT_Best_DRO_1_over_n,
    'PretrainedResnet50FT_Best_DRO_mean_and_worst' : PretrainedResnet50FT_Best_DRO_mean_and_worst,
    'PretrainedResnet50FT_Best_DRO_log' : PretrainedResnet50FT_Best_DRO_log,
    'PretrainedResnet50FT_random_matrix' : PretrainedResnet50FT_random_matrix,
    'PretrainedResnet18FT_random_matrix' : PretrainedResnet18FT_random_matrix,
    'PretrainedResnet50FT_Best_DRO_worst_of_batch' : PretrainedResnet50FT_Best_DRO_worst_of_batch,
    'PretrainedResnet50FT_Best_DRO_mean_loss' : PretrainedResnet50FT_Best_DRO_mean_loss,
    'PretrainedResnet50FT_Best_DRO_min' : PretrainedResnet50FT_Best_DRO_min,
    'Bekind_sl' : Bekind_sl,
    'PretrainedResnet50FT_Hosp_DRO_log' : PretrainedResnet50FT_Hosp_DRO_log,
    'PretrainedResnet50FT_Hosp_DRO_mean' : PretrainedResnet50FT_Hosp_DRO_mean,
    'PretrainedResnet50FT_Hosp_DRO_max' : PretrainedResnet50FT_Hosp_DRO_max,
    'PretrainedResnet50FT_Hosp_DRO_gap' : PretrainedResnet50FT_Hosp_DRO_gap,
    'PretrainedResnet50FT_Hosp_DRO_weighted' : PretrainedResnet50FT_Hosp_DRO_weighted,
    'PretrainedResnet50FT_Hosp_DRO_plus_1_over_n' : PretrainedResnet50FT_Hosp_DRO_plus_1_over_n,
    'PretrainedResnet50FT_Hosp_DRO_abstain' : PretrainedResnet50FT_Hosp_DRO_abstain,
    'PretrainedResnet50FT_Best_DRO_abstain' : PretrainedResnet50FT_Best_DRO_abstain,
    'PretrainedResnet50FT_Best_DRO_abstain_conservative' : PretrainedResnet50FT_Best_DRO_abstain_conservative,
    'PretrainedResnet50FT_Hosp_DRO_abstain_conservative' : PretrainedResnet50FT_Hosp_DRO_abstain_conservative,
    'PretrainedResnet50FT_Hosp_DRO_log_two_test' : PretrainedResnet50FT_Hosp_DRO_log_two_test
}