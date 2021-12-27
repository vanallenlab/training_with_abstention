import sys
sys.path.append('/home/jupyter/LUAD/Lung')

from imports import *
#define the random state

# Set up your default hyperparameters before wandb.init
# so they get properly set in the sweep
hyperparameter_defaults = dict(
    min_num_slides_p_hosp = 0,
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
    NUM_HOLD_OUT_HOSPS = 5,
    becor_model = 'Bekind_indhe_dro_log',
    normal_model = 'PretrainedResnet50FT_Hosp_DRO_abstain',
    becor_log_lr = -5,
    becor_min_num_tiles_p_slide = 100,
    dropout = 0.0,
    C = 1,
    log_m = 16,
    accumulate_grad_batches=1,
    val_check_interval=0.25,
    num_test = 1000,
    num_val = 1000,
    save_weights = False,
    confidence_threshold = 0.5,
    max_steps = 1000,
    early_stop = 'True',
    include_all_val = 'False',
    include_num_confident = 'False',
    temp_scale = 'True',
    infer = 'False',
    infer_whole = 'False',
    train = 'True',
    filter_tumor = 'False'
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

# wandb.log({'val_p_to_train_p' : val_p / config.train_p})

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

# if config.non_lin not in ['celu', 'hardtanh', 'logsigmoid']:
#     run.name = 'Delete Me!'
#     wandb.finish() #break from training process
#     sys.exit()

################    ################    ################    ################    ################    ################    

#Read the unhealthy dataset
kirk_whole = pd.read_pickle(r'/mnt/disks/prad/20210712_all_tiles_with_Gleason_PTEN.pkl')
kirk_whole.full_path = kirk_whole.full_path.str.replace('/mnt/disks/prostate_data/', '/mnt/disks/prad/')
kirk_whole_g24 = kirk_whole[kirk_whole['Reviewed_Gleason_category'].isin(['3+3', '4+4', '4+5', '5+3', '5+4', '5+5'])]
kirk_whole_g24.loc[:, 'grade'] = kirk_whole['Reviewed_Gleason_category']

tumor_preds = pd.read_csv('/mnt/disks/prad/20210803_tissue_tcga_inference.csv')


if config.filter_tumor == 'True':
    kirk_whole_g24 = kirk_whole_g24[tumor_preds['2_prob'] > 0.33]
else:
    pass

#Assign labels
kirk_whole_g24.loc[kirk_whole_g24['grade'] != '3+3', 'grade'] = 1
kirk_whole_g24.loc[kirk_whole_g24['grade'] == '3+3', 'grade'] = 0

#Read the unhealthy dataset
unhealthy_whole = kirk_whole_g24[kirk_whole_g24['grade'] == 1]
#Read the healthy dataset
healthy_whole = kirk_whole_g24[kirk_whole_g24['grade'] == 0]

#Assign labels
unhealthy_whole['healthy'] = 0
healthy_whole['healthy'] = 1

healthy = preprocess_df(healthy_whole, num_hosps = num_healthy_hosps, num_tiles = num_tiles, min_num_slides_p_hosp = min_num_slides_p_hosp, min_num_tiles_p_slide = min_num_tiles_p_slide, replace = replace) 
unhealthy = preprocess_df(unhealthy_whole, num_hosps = num_unhealthy_hosps, num_tiles = num_tiles, min_num_slides_p_hosp = min_num_slides_p_hosp, min_num_tiles_p_slide = min_num_tiles_p_slide, replace = replace)

#Start with a few hospitals from each 

#Remove hospitals that have both tumor and surrounding tissue samples because we are going to augment these groups differently
hosps_in_both = set(healthy['source_id']).intersection(set(unhealthy['source_id']))

#Keep only this data (For now, we will remove this data from the unhealthy since the heathy tiles don't have that many samples)
# healthy = healthy[~healthy['source_id'].isin(hosps_in_both)]
# unhealthy = unhealthy[~unhealthy['source_id'].isin(hosps_in_both)]

print('healthy')
print(healthy.groupby(['source_id', 'healthy']).nunique())

print('unhealthy')
print(unhealthy.groupby(['source_id', 'healthy']).nunique())

total = pd.concat([unhealthy, healthy], axis = 0)
total.reset_index(inplace = True, drop = True)

#When doing a classification task, I need to have the targets be torch tensors
total['healthy'] = torch.from_numpy(total['healthy'].values.astype(float)).type(torch.LongTensor) 

#Create a hold out test set 
test_healthy_hosps = random.sample(population = sorted(list(set(healthy['source_id']))), k = NUM_HOLD_OUT_HOSPS//2)
test_unhealthy_hosps = random.sample(population = sorted(list(set(unhealthy['source_id']))), k = NUM_HOLD_OUT_HOSPS//2)

test_hosps = test_healthy_hosps + test_unhealthy_hosps

test_set = total[total['source_id'].isin(test_hosps)]
test_set = balance_labels(test_set, 'source_id', replace = replace)
test_set = balance_labels(test_set, 'healthy', replace = replace)
# test_set = test_set.sample(n = min(len(test_set), 1000), replace = True, random_state = random_state)
test_set = test_set.drop_duplicates(ignore_index = True)

total = total[~total['source_id'].isin(test_hosps)]

print('hold_out_sources', test_hosps, len(test_set))
print('test_set')
print(test_set.groupby(['source_id', 'healthy']).nunique())
print(test_set.groupby(['source_id', 'healthy']).size())


#Create group splitting object and split the dataset into groups
gss = GroupShuffleSplit(n_splits=100, train_size=0.7, random_state=random_state)
GROUP_COL = split
lp = total.copy()
lp.reset_index(drop = True, inplace = True)        
print('lp')
print(lp.groupby(['source_id', 'healthy']).nunique())

splits = list(gss.split(X = list(range(len(lp))), groups = lp[GROUP_COL]))

td_ctr = collections.Counter([])
vd_ctr = collections.Counter([])
ted_ctr = collections.Counter([])

splits_iterator = iter(splits)

ct = 0
while (0 not in td_ctr) or (1 not in td_ctr)  or (0 not in vd_ctr) or (1 not in vd_ctr) or (0 not in ted_ctr) or (1 not in ted_ctr):
    ct += 1
    if ct > 11:
        run.name = 'Delete Me!'
        wandb.finish() #break from training process
        sys.exit()
        break
    
    train_idx, val_idx = next(splits_iterator)

    print(train_idx, val_idx)

    #Create train and val paths
    train_paths, val_paths = lp.iloc[train_idx] , lp.iloc[val_idx]

    val_paths, train_paths = balance_labels(val_paths, 'healthy', replace = replace), balance_labels(train_paths, 'healthy', replace = replace)    

    #Adjusting for the path lenghts
    if len(val_paths) > len(train_paths):
        train_paths, val_paths = val_paths, train_paths

    print('#Final Distributions')
    print('val_paths', val_paths.groupby(['source_id', 'healthy']).nunique())
    print('train_paths', train_paths.groupby(['source_id', 'healthy']).nunique())
    # print('val_paths', val_paths.groupby(['source_id', 'healthy']).size())
    # print('train_paths', train_paths.groupby(['source_id', 'healthy']).size())

    td_ctr = collections.Counter(train_paths.healthy.values)
    vd_ctr = collections.Counter(val_paths.healthy.values)
    ted_ctr = collections.Counter(test_set.healthy.values)
    print('train distribution', td_ctr,  'val distribution', vd_ctr)

# try:
#     wandb.log(dict(td_ctr))
#     wandb.log(dict(vd_ctr))
#     wandb.log(dict(ted_ctr))
# except Exception as e:
#     print(e)
#     pass

#Keeping track of the sources
train_sources = sorted(list(set(train_paths['source_id'])))
val_sources = sorted(list(set(val_paths['source_id'])))
uniq_srcs = list(set(lp['source_id'])) + test_hosps

#important - need this for the new model
srcs_map = {uniq_srcs[i] : i for i in range(len(uniq_srcs))}

#Similarlt, for patients
uniq_pts = list(set(lp['slide_id'])) + list(set(test_set['slide_id']))
pts_map = {uniq_pts[i] : i for i in range(len(uniq_pts))}
                
#Just some checks to check that the patients don't overlap
assert set(val_paths[GROUP_COL]).intersection(set(train_paths[GROUP_COL])) == set()
assert set(test_hosps).intersection(train_sources) == set()
assert set(test_hosps).intersection(val_sources) == set()

# train_paths = train_paths.sample(n = 1000, random_state = random_state)
val_paths = val_paths.sample(n = min(len(val_paths), config.num_val), random_state = random_state)
total_whole = pd.concat([unhealthy_whole, healthy_whole], axis = 0)
test_set_whole = total_whole[total_whole['source_id'].isin(test_hosps)]
test_set = test_set.sample(n = min(len(test_set), config.num_test), random_state = random_state)



train_dataset = SlideDataset(
        paths=train_paths.full_path.values,
        slide_ids=train_paths.slide_id.values,
        labels=train_paths.healthy.values,
        transform_compose=train_transform(full_size=512, crop_size=crop_size, s = train_s)
    )

val_dataset = SlideDataset(
        paths=val_paths.full_path.values,
        slide_ids=val_paths.slide_id.values,
        labels=val_paths.healthy.values,
        transform_compose=eval_transform(full_size=512, crop_size=crop_size),
    ) 

test_dataset = SlideDataset(
        paths=test_set.full_path.values,
        slide_ids=test_set.slide_id.values,
        labels=test_set.healthy.values,
        transform_compose=eval_transform(full_size = 512, crop_size = crop_size)
    )    

test_inf_dataset = SlideDataset(
        paths=test_set_whole.full_path.values,
        slide_ids=test_set_whole.slide_id.values,
        labels=test_set_whole.healthy.values,
        transform_compose=eval_transform(full_size = 512, crop_size = crop_size)
    )    


#Create Train Data Loader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)

#Create Val Data Loader
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#Create Test Data Loader
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#Create Test Data Loader
test_inf_dataloader = DataLoader(test_inf_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

print('len(test_dataloader)', len(test_dataloader))
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
                                                'include_all_val' : config.include_all_val,
                                                'include_num_confident' : config.include_num_confident
                                                })

elif config.model_type == 'becor':
    # Create model #mc_lightning_be_cor.models.resnet.resnet_module.
    model = model_choices[config.becor_model](hparams = {'lr' : lr, 
                                            'num_classes' : 2,
                                            'num_srcs' : len(train_sources) + len(val_sources) + len(test_hosps),
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

if (config.temp_scale == 'True') and (config.normal_model == 'PretrainedResnet50FT_Hosp_DRO_abstain'):    
    model.set_temperature(val_dataloader)

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
logger = WandbLogger(name = 'pred_healthy', save_dir = '/home/jupyter/LUAD/Lung/lightning_logs/healthy', project = 'DRO')

# logger = TensorBoardLogger("tb_logs", name="my_model")

callbacks = []
if config.early_stop == 'True':
    callbacks.append(early_stop_callback)
elif config.early_stop == 'False':
    pass

#Create a trainer obj
trainer = pl.Trainer(gpus=1, 
                    auto_lr_find = True,
                    auto_select_gpus=False,
                    accelerator='ddp',
                    max_epochs = 100,
                    gradient_clip_val=0,
                    track_grad_norm =0,
                    callbacks=callbacks, 
                    logger=logger,  
                    # limit_train_batches=50,
                    val_check_interval=config.val_check_interval,
                    # limit_val_batches=10,
                    accumulate_grad_batches=config.accumulate_grad_batches,
                    log_every_n_steps=1,
                    # min_steps=100, 
                    max_steps=config.max_steps,
                    fast_dev_run=0,
                    deterministic=True,
                    stochastic_weight_avg=False)

#Fit the trainer and start training
trainer.fit(model, train_dataloader, val_dataloader)

# model.load_state_dict(torch.load('/home/jupyter/LUAD/Lung/weights/' + 'dark-firefly-7765'))

#Test the model on some unseen test data
trainer.test(model, test_dataloaders = test_dataloader)
# trainer.test(model, test_dataloaders = val_dataloader)

try:
    print(wandb.run.summary._as_dict())
    for item in wandb.run.summary._as_dict():
        if 'train' in item:
            qty = item[len('train'):]
            try:
                wandb.run.summary['tt_gap' + qty] = wandb.run.summary['test' + qty] - wandb.run.summary['train' + qty]
                wandb.run.summary['vt_gap' + qty] = wandb.run.summary['val' + qty] - wandb.run.summary['train' + qty]
            except Exception as e:
                print(e)
                print(qty)
                pass

except Exception as e:
    print(e)
    pass

#Save the model - good practice if you want to retrieve it
EXP_VERSION = wandb.run.name

if config.save_weights == True:
    torch.save(model.state_dict(), f'/home/jupyter/LUAD/Lung/weights/{EXP_VERSION}')

test_set.to_csv(f'/home/jupyter/LUAD/Lung/test_sets/{EXP_VERSION}.csv')

#Use these lines to empty the GPU memory
# torch.cuda.empty_cache()
# del model

##############################################################################################################
print("# Inferring the predictions on a hold out test set slide")

if config.infer_whole == 'True':
    # Move the model to GPU
    model = nn.DataParallel(model).to('cuda')
    #Create a softmax layer to convert the preds to probs
    smfx_lyr = nn.Softmax(dim=1)

    # Create an empty tensor to store the outputs
    logits = np.empty((1, config.num_mc_dropout), float)
    
    # Store the predicted probabilities 
    

    with torch.no_grad():
        for (imgs, labels, slide_ids) in tqdm(test_inf_dataloader):            
            model.eval()            
            task_embeddings = model(imgs)
            
            predicted_probabilities = np.empty((len(imgs), 1), float)
            
            for i in range(config.num_mc_dropout):
                #Create dropout module
                dropout = nn.Dropout(p = config.mc_dropout_p)
                
                #Apply dropout to the task embeddings
                task_embeddings = dropout(task_embeddings)   
                
                # Convert the predictions to probabilities        
                pred_probs_i = smfx_lyr(model.module.classifier(task_embeddings))

                pred_probs_i = pred_probs_i[:, 1]

                #Bring pred probs to cpui
                pred_probs_i = pred_probs_i.cpu()

                pred_probs_i = pred_probs_i.reshape((-1, 1))

                # print('pred_probs_i', pred_probs_i.shape)

                #
                predicted_probabilities = np.append(predicted_probabilities, pred_probs_i, axis = 1)

            # print('predicted_probabilities', predicted_probabilities.shape)
                       
            logits  = np.append(logits, predicted_probabilities[:, 1:], axis = 0)
    
    print('logits', logits.shape)

    mean_logits = np.mean(logits[1:, :], axis = 1)
    std_logits = np.std(logits[1:, :], axis = 1)
    
    print('mean_logits', mean_logits.shape)
    print('std_logits', std_logits.shape)
    
    print('mean_logits', mean_logits)
    print('std_logits', std_logits)
    # Convert the predictions to a list and save it as a column of the data Frame
    test_set_whole['prob_healthy'] = mean_logits
    test_set_whole['std_logits'] = std_logits

    ###
    # task_adata = run_embedding(encoding_array = task_store.detach().cpu().numpy(),
    #                     annotation_df = test_set_whole, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

    # task_adata.write('/home/jupyter/LUAD/Lung/embeddings/task_adata' + '_' + EXP_VERSION + '.h5ad')

    # sc.pl.umap(task_adata, size = 120, projection = '2d', color = ['healthy'], 
    #     palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + '_hlthy_' + model_type + '_' + EXP_VERSION + '.png')

    test_set_whole.to_csv(f'./preds/{EXP_VERSION}_whole.csv')


if config.infer == 'True':
    # Move the model to GPU
    model = nn.DataParallel(model).to('cuda')
    #Create a softmax layer to convert the preds to probs
    smfx_lyr = nn.Softmax(dim=1)

    # Create an empty tensor to store the outputs
    logits = torch.Tensor([])

    with torch.no_grad():
        for (imgs, labels, slide_ids) in tqdm(test_dataloader):            
            model.eval()            
            task_embeddings = model(imgs)                       
            # Convert the predictions to probabilities        
            pred_probs = smfx_lyr(model.module.classifier(task_embeddings))
            #Bring pred probs to cpu
            pred_probs = pred_probs.cpu()
            logits  = torch.cat([logits, pred_probs])

    # Convert the predictions to a list and save it as a column of the data Frame
    test_set['prob_healthy'] = logits[:, 1].flatten().tolist()

    test_set.to_csv(f'./preds/{EXP_VERSION}_whole.csv')


wandb.finish()

# Clear the memory
# torch.cuda.empty_cache()
# del model