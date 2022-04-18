import sys
sys.path.append('/home/jupyter/LUAD/Lung')

from imports import *
infer = False
#define the random state

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparameter_defaults = dict(
    min_num_slides_p_hosp = 5,
    train_c = 0.00,
    test_c_times_train_c = -1,
    val_c_times_train_c = -1,
    train_s = 0,
    col_sch = 'RGB',
    crop_size = 224,
    batch_size = 128,
    sample = "without_replacement",
    split = 'source_id',
    model_type = 'normal',
    num_unhealthy_hosps = 50,
    num_healthy_hosps = 50,
    non_lin = 'relu',
    augment = 'normal',
    iter = 2,
    log_weight_decay = -5,
    p = 0.9,
    train_p = 0.75,
    val_p = 0.75,
    train_p_val_p_rln = 'false',
    patience = 5,
    becor_loss_choice = 'add_cos_sim_abs_agg',
    NUM_HOLD_OUT_HOSPS = 3,
    becor_model = 'Bekind_indhe_dro_log',
    normal_model = 'PretrainedResnet50FT_Hosp_DRO_abstain',
    becor_log_lr = -5,
    becor_min_num_tiles_p_slide = 100,
    dropout = 0.0,
    C = 1,
    log_m = 16,
    accumulate_grad_batches=1,
    val_check_interval=0.25,
    num_test_tiles = 2000,
    num_val_tiles = 2000,
    num_train_tiles = 10000,
    save_weights = False,
    confidence_threshold = 0.9,
    max_steps = 5000,
    early_stop = 'True',
    num_hosps = 3,
    num_test_hosps = 5,
    num_slides = 10,
    training_hosp = '44',
    val_hosp = '44',
    normalize = 'not_staintools',
    val_stain = 'same',
    include_all_val = 'False',
    include_num_confident = 'False',
    temp_scale = 'True'
  ) 

# Pass your defaults to wandb.init
run = wandb.init(config=hyperparameter_defaults)
config = wandb.config

if config.train_p_val_p_rln == 'peg':
    val_p = config.train_p
elif config.train_p_val_p_rln == 'invert':
    val_p = 1 - config.train_p
else:
    val_p = config.val_p

wandb.log({'val_p_to_train_p' : val_p / config.train_p})

NUM_HOLD_OUT_HOSPS = 2 * config.NUM_HOLD_OUT_HOSPS

random_state = config.iter #3
random.seed(a=random_state, version=2)
seed_everything(random_state)

non_lin = non_lin_choices[config.non_lin]

#How many slides per hospital
min_num_slides_p_hosp = config.min_num_slides_p_hosp
#How much train batch effect to inject
train_c = config.train_c
#What model to use
model_type = config.model_type
#How many tumor hospitals to take data from 
num_unhealthy_hosps = config.num_unhealthy_hosps
#How many healthy hospitals to take data from 
num_healthy_hosps = config.num_healthy_hosps
#Relative Val injection compared to train injection
val_c_times_train_c = config.val_c_times_train_c
val_c = val_c_times_train_c * train_c
#Relative Test injection compared to train injection
test_c_times_train_c = config.test_c_times_train_c
test_c = test_c_times_train_c * train_c
#RGB or HSV
col_sch = config.col_sch
if col_sch == 'HSV':
    train_transform = HSVTrainTransform
    eval_transform = HSVEvalTransform
elif col_sch == 'RGB':
    train_transform = RGBTrainTransform
    eval_transform = RGBEvalTransform
#Crop Size of the image
crop_size = config.crop_size
#Batch size used in the training process
batch_size = config.batch_size
#Relative Color Augmentation 
train_s = config.train_s

#How to sample the data
sample = config.sample
if sample == "with_replacement":
    replace = True
elif sample == "without_replacement":
    replace = False

#How to split the data between train and val
split = config.split

if (model_type == 'normal'):
  combine_embeddings = embeddings_combi_choices['just_task_embeddings']
  combine_losses = loss_choices['just_task_loss']
  min_num_tiles_p_slide = config.becor_min_num_tiles_p_slide
  loglr = config.becor_log_lr

elif model_type == 'becor':
  combine_embeddings = embeddings_combi_choices['sub_both_embeddings_avg']
  combine_losses = loss_choices[config.becor_loss_choice]
  min_num_tiles_p_slide = config.becor_min_num_tiles_p_slide
  loglr = config.becor_log_lr

num_tiles = min_num_tiles_p_slide 
lr = 10**(loglr)

weight_decay = 10**(config.log_weight_decay)

augment = config.augment



# if (config.training_hosp in ['49', '73', '78', '50', '99', '38']) and (config.val_hosp in ['49', '73', '78', '50', '99', '38']):
#     run.name = 'Delete Me!'
#     wandb.finish() #break from training process
#     sys.exit()

################    ################    ################    ################    ################    ################    

#Read the unhealthy dataset
unhealthy = pd.read_csv('/home/jupyter/LUAD/Lung/data/lung_paths.csv')
#Read the healthy dataset
healthy = pd.read_csv('/home/jupyter/LUAD/Lung/data/lung_healthy_paths.csv')

unhealthy.full_path = unhealthy.full_path.str.replace('/mnt/disks/data_disk/', '/home/jupyter/')
healthy.full_path = healthy.full_path.str.replace('/mnt/disks/data_disk/', '/home/jupyter/')

#If we want to work with images for which we have stain tools augmentations
if config.normalize == 'staintools':    
    imgs_we_have = [img[:-len('_t2.png')] + '.png' for x in tqdm(os.walk('/home/jupyter/LUAD/')) for img in glob(os.path.join(x[0], '*t2.png'))]
    
    healthy = healthy[healthy['full_path'].isin(imgs_we_have)]
    unhealthy = unhealthy[unhealthy['full_path'].isin(imgs_we_have)]
    print('num_unhealthy', len(unhealthy))
    print('num_healthy', len(healthy))

#Assign labels
unhealthy['healthy'] = 0
healthy['healthy'] = 1

#Start with a few hospitals from each 
healthy = preprocess_df(healthy, num_hosps = num_healthy_hosps, num_tiles = num_tiles, min_num_slides_p_hosp = 0, min_num_tiles_p_slide = num_tiles, replace = replace) 
unhealthy = preprocess_df(unhealthy, num_hosps = num_unhealthy_hosps, num_tiles = num_tiles, min_num_slides_p_hosp = min_num_slides_p_hosp, min_num_tiles_p_slide = min_num_tiles_p_slide, replace = replace)

print('len(healthy)', len(healthy), 'len(unhealthy)', len(unhealthy))

all_healthy_hosps = set(healthy['source_id'])
all_unhealthy_hosps = set(unhealthy['source_id'])

all_hosps = all_healthy_hosps.union(all_unhealthy_hosps)
hosps_in_both = all_healthy_hosps.intersection(all_unhealthy_hosps)

hosps = [config.training_hosp, config.val_hosp]

total = pd.concat([healthy, unhealthy])
total[total['source_id'].isin(hosps_in_both)]

sorted_hosps = total.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False).index

print(healthy.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False))
print(unhealthy.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False))

print(hosps_in_both)
print('num_unhealthy', len(unhealthy))
print('num_healthy', len(healthy))
# sys.exit()

training_hosp = config.training_hosp
val_hosp = config.val_hosp

def sample_tiles_n_slides_from_source(x, num_slides = config.num_slides):
    return x.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x = x, num_slides = num_slides, num_tiles = num_tiles)).reset_index(drop = True)

if training_hosp != val_hosp:
    healthy_train = healthy[healthy['source_id'].isin([config.training_hosp])]
    unhealthy_train = unhealthy[unhealthy['source_id'].isin([config.training_hosp])]

    healthy_val = healthy[healthy['source_id'].isin([config.val_hosp])]
    unhealthy_val = unhealthy[unhealthy['source_id'].isin([config.val_hosp])]
    
    train_paths = [healthy_train, unhealthy_train]
    val_paths = [healthy_val, unhealthy_val]

    train_paths = list(map(sample_tiles_n_slides_from_source, train_paths))
    val_paths = list(map(sample_tiles_n_slides_from_source, val_paths))

    train_paths = pd.concat(train_paths)
    val_paths = pd.concat(val_paths)
else:
    healthy_train = healthy[healthy['source_id'].isin([config.training_hosp])]
    unhealthy_train = unhealthy[unhealthy['source_id'].isin([config.training_hosp])]

    train_paths = [healthy_train, unhealthy_train]

    train_paths = [sample_tiles_n_slides_from_source(x, int(2 * config.num_slides)) for x in train_paths]

    train_paths, val_paths = create_tp_vp(lp = pd.concat(train_paths), GROUP_COL = 'slide_id', train_size = 0.7, random_state = random_state, num_classes = 2, replace = False)

###################Balance Labels
# train_paths, val_paths = balance_labels(train_paths, 'source_id', replace = replace), balance_labels(val_paths, 'source_id', replace = replace)
train_paths, val_paths = balance_labels(train_paths, 'healthy', replace = replace), balance_labels(val_paths, 'healthy', replace = replace)
###################


###################
test_healthy_hosp = random.sample(population = sorted(list(set(all_healthy_hosps).difference(set(hosps)))), k = config.num_test_hosps)
test_unhealthy_hosp = random.sample(population = sorted(list(set(all_unhealthy_hosps).difference(set(hosps)))), k = config.num_test_hosps)

assert set(hosps).intersection(set(test_healthy_hosp).union(set(test_unhealthy_hosp))) == set()

test_set = pd.concat([healthy[healthy['source_id'].isin(test_healthy_hosp)], unhealthy[unhealthy['source_id'].isin(test_unhealthy_hosp)]])

test_set = balance_labels(test_set, 'healthy', replace = replace)
###################


####################Adjusting for the path lenghts
# if len(val_paths) > len(train_paths):
#     train_paths, val_paths = val_paths, train_paths
###################

###################
train_paths = train_paths.sample(n = min(len(train_paths), config.num_train_tiles), random_state = random_state)
val_paths = val_paths.sample(n = min(len(val_paths), config.num_val_tiles), random_state = random_state)
test_set = test_set.sample(n = min(len(test_set), config.num_test_tiles), random_state = random_state)
###################

###################
if (config.normalize == 'staintools'): 
    train_paths['full_path'] = train_paths['full_path'].str[:-len('.png')] + '_t1.png'
      
    if (config.val_stain != 'same'):
        val_paths['full_path'] = val_paths['full_path'].str[:-len('.png')] + '_t2.png'
        
    else:
        val_paths['full_path'] = val_paths['full_path'].str[:-len('.png')] + '_t1.png'
###################

try:
    print('val_dups')
    print(val_paths[val_paths.duplicated(['full_path'], keep=False)])
except:
    pass

###################
print('#Final Distributions')
print('val_paths', val_paths.groupby(['source_id', 'healthy']).nunique())
print('train_paths', train_paths.groupby(['source_id', 'healthy']).nunique())
print('test_set', test_set.groupby(['source_id', 'healthy']).nunique())

td_ctr = collections.Counter(train_paths.healthy.values)
vd_ctr = collections.Counter(val_paths.healthy.values)
ted_ctr = collections.Counter(test_set.healthy.values)
print('train distribution', td_ctr,  'val distribution', vd_ctr, 'test_distribution', ted_ctr)
    
#Keeping track of the sources
train_sources = sorted(list(set(train_paths['source_id'])))
val_sources = sorted(list(set(val_paths['source_id'])))
test_sources = sorted(list(set(test_set['source_id'])))
###################


###################
wandb.log({'num_train' : len(train_paths), 
            'num_val' : len(val_paths),
            'num_test' : len(test_set)
            })

uniq_srcs = sorted(list(hosps))

#important - need this for the new model
srcs_map = {uniq_srcs[i] : i for i in range(len(uniq_srcs))}
###################



####################Create Datasets
train_dataset = SlideDataset(
        paths=train_paths.full_path.values,
        slide_ids=train_paths.slide_id.values,
        labels=train_paths.healthy.values,
        transform_compose=train_transform(full_size = 512, crop_size = config.crop_size, s = config.train_s)
    )

val_dataset = SlideDataset(
        paths=val_paths.full_path.values,
        slide_ids=val_paths.slide_id.values,
        labels=val_paths.healthy.values,
        transform_compose=eval_transform(full_size = 512, crop_size = config.crop_size)
    )

test_dataset = SlideDataset(
        paths=test_set.full_path.values,
        slide_ids=test_set.slide_id.values,
        labels=test_set.healthy.values,
        transform_compose=eval_transform(full_size = 512, crop_size = config.crop_size)
    )
###################


#Create Train Data Loader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)

#Create Val Data Loader
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#Create Val Data Loader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#sys.exit()
monitor_var = 'val_task_loss'

if config.model_type == 'normal':
    # Create model #mc_lightning_be_cor.models.resnet.resnet_module.
    model = model_choices[config.normal_model](hparams = {'lr' : lr, 
                                                'num_classes' : 2,
                                                'weight_decay' : weight_decay,
                                                'dropout' : config.dropout,
                                                'C' : config.C,
                                                'M' : 2 ** config.log_m,
                                                'confidence_threshold' : config.confidence_threshold,
                                                'srcs_map' : srcs_map,
                                                'include_all_val' : config.include_all_val,
                                                'include_num_confident' : config.include_num_confident
                                                })

elif config.model_type == 'becor':
    # Create model #mc_lightning_be_cor.models.resnet.resnet_module.
    model = model_choices[config.becor_model](hparams = {'lr' : lr, 
                                            'num_classes' : 2,
                                            'num_srcs' : len(train_sources) + len(val_sources),
                                            'num_slides' : len(pts_map),
                                            'srcs_map' : srcs_map,
                                            'slides_map' : pts_map,                                            
                                            'combine_loss' : combine_losses,
                                            'combine_embeddings': combine_embeddings,
                                            'model_type' : model_type,
                                            'num_stains' : 2,
                                            'non_lin' : non_lin,
                                            'weight_decay' : weight_decay,
                                            'dropout' : config.dropout
                                            })             


#Temperature Scale Model

if (config.temp_scale == 'True') and (config.normal_model == 'PretrainedResnet50FT_Hosp_DRO_abstain'):    
    model.set_temperature(val_dataloader)


#Create Checkpoint Callback
checkpoint_callback = ModelCheckpoint(monitor=monitor_var)

#Create Early Stopping Callback
early_stop_callback = EarlyStopping(
    monitor= monitor_var,
    min_delta=0.00,
    patience=config.patience,
    verbose=False,
    mode='min'
)

#Create a tb logger
# logger = TensorBoardLogger(save_dir = '/home/jupyter/LUAD/Lung/lightning_logs', name="healthy", )
logger = WandbLogger(name = 'becor', save_dir = '/home/jupyter/LUAD/Lung/lightning_logs/healthy', project = 'source_be')

# logger = TensorBoardLogger("tb_logs", name="my_model")

#Create a trainer obj
trainer = pl.Trainer(gpus=1, 
                    auto_lr_find = True,
                    auto_select_gpus=False,
                    accelerator='ddp',
                    gradient_clip_val=0,
                    track_grad_norm =0,
                    callbacks=[early_stop_callback, checkpoint_callback], 
                    logger=logger,  
                    # limit_train_batches=50,
                    val_check_interval=config.val_check_interval,
                    # limit_val_batches=10,
                    accumulate_grad_batches=config.accumulate_grad_batches,
                    log_every_n_steps=1,
                    # min_steps=100, 
                    max_steps=1000,
                    fast_dev_run=0,
                    deterministic=True,
                    stochastic_weight_avg=False)

# sys.exit()

#Fit the trainer and start training
trainer.fit(model, train_dataloader, val_dataloader)

# model = model_choices[config.normal_model].load_from_checkpoint(checkpoint_callback.best_model_path) 

#Test the model on some unseen test data
# trainer.test(model, test_dataloaders = val_dataloader)
trainer.test(model, test_dataloaders = test_dataloader)

#Save the model - good practice if you want to retrieve it
# EXP_VERSION = EXP = f'CE:tsk:src_be,srcs:{config.num_hosps},num_slides{config.min_num_slides_p_hosp},num_tiles{num_tiles},sp:{config.split},img:{config.col_sch},s:{config.train_s},fold:{config.iter}'

# torch.save(model.state_dict(), '/home/jupyter/LUAD/Lung/weights/weights_' + wandb.run.name)

try:    
    for item in wandb.run.summary._as_dict():
        if 'external' in item:
            qty = item[len('external_'):]
            try:
                wandb.run.summary['src_be_' + qty + 'bias'] = wandb.run.summary[item] - wandb.run.summary[qty]
            except Exception as e:
                print(e)
                print(qty)
                pass
    print(wandb.run.summary._as_dict())
except Exception as e:
    print(e)
    pass


if infer == False:
    wandb.finish()
    sys.exit()

#Use these lines to empty the GPU memory
# torch.cuda.empty_cache()
# del model

##############################################################################################################
print("# Inferring the predictions on a hold out test set slide")
#sample_val_slide data
#Change the paths since the paths were from an older dir struc
# total_cp = total.copy()

# total_cp.at[total_cp[GROUP_COL].isin(train_groups), 'train/val/test'] = 'train'
# total_cp.at[total_cp[GROUP_COL].isin(val_groups), 'train/val/test'] = 'val'

# total_cp.at[total_cp['full_path'].isin(set(train_paths['full_path'])), 'train'] = 1
# total_cp.at[~total_cp['full_path'].isin(set(train_paths['full_path'])), 'train'] = 0

# total_cp.at[total_cp['full_path'].isin(set(val_paths['full_path'])), 'val'] = 1
# total_cp.at[~total_cp['full_path'].isin(set(val_paths['full_path'])), 'val'] = 0
    
# Move the model to GPU
model = nn.DataParallel(model).to('cuda')
#Create a softmax layer to convert the preds to probs
smfx_lyr = nn.Softmax(dim=1)

# Create an empty tensor to store the outputs
embs = torch.Tensor([])

img_store = torch.Tensor([])
src_store = torch.Tensor([])
task_store = torch.Tensor([])

logits = torch.Tensor([])

with torch.no_grad():
    for (imgs, labels, slide_ids) in tqdm(test_dataloader):            
        # print(labels)
        model.eval()
        
        img_embs, src_embeddings, task_embeddings = model(imgs)   
        embs  = torch.cat([embs, task_embeddings.cpu()])                        

        img_store = torch.cat([img_store, img_embs.cpu()])                        
        src_store = torch.cat([src_store, src_embeddings.cpu()])                        
        task_store = torch.cat([task_store, task_embeddings.cpu()])                        
                
        # Convert the predictions to probabilities        
        pred_probs = smfx_lyr(model.module.classifier(task_embeddings))
        #Bring pred probs to cpu
        pred_probs = pred_probs.cpu()
        logits  = torch.cat([logits, pred_probs])

        # print('logits.shape', logits.shape) #should n x 2

# Convert the predictions to a list and save it as a column of the data Frame
test_set['preds'] = logits[:, 0].flatten().tolist()

###
task_adata = run_embedding(encoding_array = task_store.detach().cpu().numpy(),
                    annotation_df = test_set, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

task_adata.write('/home/jupyter/LUAD/Lung/embeddings/task_adata' + '_' + EXP_VERSION + '.h5ad')

sc.pl.umap(task_adata, size = 120, projection = '2d', color = ['healthy'], 
    palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + '_hlthy_' + model_type + '_' + EXP_VERSION + '.png')

###
src_adata = run_embedding(encoding_array = src_store.detach().cpu().numpy(),
                    annotation_df = test_set, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

src_adata.write('/home/jupyter/LUAD/Lung/embeddings/src_adata' + '_' + EXP_VERSION + '.h5ad')

sc.pl.umap(src_adata, size = 120, projection = '2d', color = ['source_id'], 
    palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + '_src_' + model_type + '_' + EXP_VERSION + '.png')

###
img_adata = run_embedding(encoding_array = img_store.detach().cpu().numpy(),
                    annotation_df = test_set, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

img_adata.write('/home/jupyter/LUAD/Lung/embeddings/img_adata' + '_' + EXP_VERSION + '.h5ad')

sc.pl.umap(img_adata, size = 120, projection = '2d', color = ['source_id'], 
    palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + '_img_src_' + model_type + '_' + EXP_VERSION + '.png')

sc.pl.umap(img_adata, size = 120, projection = '2d', color = ['healthy'], 
    palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + '_img_hlt_' + model_type + '_' + EXP_VERSION + '.png')

wandb.finish()

# Clear the memory
# torch.cuda.empty_cache()
# del model