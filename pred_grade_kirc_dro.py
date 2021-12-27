import sys
sys.path.append('/home/jupyter/LUAD/Lung')

from imports import *

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
    iter = 4,
    log_weight_decay = -5,
    p = 0.9,
    train_p = 0.75,
    val_p = 0.75,
    train_p_val_p_rln = 'false',
    patience = 5,
    becor_loss_choice = 'add_cos_sim_abs_agg',
    NUM_HOLD_OUT_HOSPS = 0,
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
    save_weights = True,
    confidence_threshold = 0.5,
    max_steps = 1000,
    early_stop = 'True',
    include_all_val = 'False',
    include_num_confident = 'False',
    temp_scale = 'True',
    infer = 'False',
    infer_whole = 'False',
    train = 'True',
    filter_tumor = 'True',
    label = 'mutation',
    gene = 'SETD2'
  ) 

# Pass your defaults to wandb.init
run = wandb.init(project="KIRC", config=hyperparameter_defaults)
config = wandb.config
EXP_VERSION = wandb.run.name

if config.train_p_val_p_rln == 'peg':
    val_p = config.train_p
elif config.train_p_val_p_rln == 'invert':
    val_p = 1 - config.train_p
else:
    val_p = config.val_p

# wandb.log({'val_p_to_train_p' : val_p / config.train_p})

NUM_HOLD_OUT_HOSPS = config.NUM_HOLD_OUT_HOSPS

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
kirk_whole = pd.read_csv('20210809_kirc_tile_inference_and_paths_for_surya_with_grade.csv')

if config.filter_tumor == 'True':
    kirk_whole = kirk_whole[kirk_whole['prob_tumor'] > 0.5]
else:
    pass

if config.label == 'grade':
    # kirk_whole.full_path = kirk_whole.full_path.str.replace('/mnt/disks/kirc/', '/mnt/disks/KIRC/')
    kirk_whole_g24 = kirk_whole[kirk_whole['grade'].isin(['G2', 'G4'])]

    #Assign labels
    kirk_whole_g24.loc[kirk_whole_g24['grade'] == 'G2', 'label'] = 0
    kirk_whole_g24.loc[kirk_whole_g24['grade'] == 'G4', 'label'] = 1
elif config.label == 'stage':
    # kirk_whole.full_path = kirk_whole.full_path.str.replace('/mnt/disks/kirc/', '/mnt/disks/KIRC/')
    kirk_whole_g24 = kirk_whole[kirk_whole['stage'].isin(['Stage I', 'Stage II', 'Stage III', 'Stage IV'])]

    #Assign labels
    kirk_whole_g24.loc[kirk_whole_g24['stage'] != 'Stage IV', 'label'] = 0
    kirk_whole_g24.loc[kirk_whole_g24['stage'] == 'Stage IV', 'label'] = 1

elif config.label == 'mutation':
    kirk_whole_g24 = kirk_whole.copy()
    #Assign labels
    kirk_whole_g24.loc[kirk_whole_g24[config.gene] == 'WT', 'label'] = 0
    kirk_whole_g24.loc[kirk_whole_g24[config.gene] == 'MUT', 'label'] = 1


#When doing a classification task, I need to have the targets be torch tensors
kirk_whole_g24['label'] = torch.from_numpy(kirk_whole_g24['label'].values.astype(float)).type(torch.LongTensor) 

kirk_whole_g24_sample = preprocess_df(kirk_whole_g24, num_hosps = num_unhealthy_hosps, num_tiles = num_tiles, min_num_slides_p_hosp = min_num_slides_p_hosp, min_num_tiles_p_slide = min_num_tiles_p_slide, replace = replace)

print(kirk_whole_g24.loc[kirk_whole_g24['label'] == 0]['source_id'].unique())
print(kirk_whole_g24.loc[kirk_whole_g24['label'] == 1]['source_id'].unique())

all_hosps = kirk_whole_g24['source_id'].unique()

#Remove hospitals that have both tumor and surrounding tissue samples because we are going to augment these groups differently

#Create group splitting object and split the dataset into groups
gss = GroupShuffleSplit(n_splits=100, train_size=0.7, random_state=random_state)
GROUP_COL = split
lp = kirk_whole_g24_sample.copy()
lp.reset_index(drop = True, inplace = True)        
print('lp')
print(lp.groupby(['source_id', 'label']).nunique())

splits = list(gss.split(X = list(range(len(lp))), groups = lp[GROUP_COL]))

td_ctr = collections.Counter([])
vd_ctr = collections.Counter([])


splits_iterator = iter(splits)

ct = 0
while (0 not in td_ctr) or (1 not in td_ctr)  or (0 not in vd_ctr) or (1 not in vd_ctr):
    ct += 1
    if ct > 110:
        run.name = 'Delete Me!'
        wandb.finish() #break from training process
        sys.exit()
        break
    
    train_idx, val_idx = next(splits_iterator)
    print(train_idx, val_idx)
    #Create train and val paths
    train_paths, val_paths = lp.iloc[train_idx] , lp.iloc[val_idx]
    val_paths, train_paths = balance_labels(val_paths, 'label', replace = replace), balance_labels(train_paths, 'label', replace = replace)    

    #Adjusting for the path lenghts
    if len(val_paths) > len(train_paths):
        train_paths, val_paths = val_paths, train_paths

    print('#Final Distributions')
    print('val_paths', val_paths.groupby(['source_id', 'label']).nunique())
    print('train_paths', train_paths.groupby(['source_id', 'label']).nunique())
    
    td_ctr = collections.Counter(train_paths['label'].values)
    vd_ctr = collections.Counter(val_paths['label'].values)
    print('train distribution', td_ctr,  'val distribution', vd_ctr)

#Keeping track of the sources
train_sources = sorted(list(set(train_paths['source_id'])))
val_sources = sorted(list(set(val_paths['source_id'])))
uniq_srcs = list(set(kirk_whole_g24['source_id']))

#important - need this for the new model
srcs_map = {uniq_srcs[i] : i for i in range(len(uniq_srcs))}

#Similarlt, for patients
uniq_pts = list(set(lp['slide_id'])) #+ list(set(test_set['slide_id']))
pts_map = {uniq_pts[i] : i for i in range(len(uniq_pts))}
                
#Just some checks to check that the patients don't overlap
assert set(val_paths[GROUP_COL]).intersection(set(train_paths[GROUP_COL])) == set()

# train_paths = train_paths.sample(n = 1000, random_state = random_state)
val_hosps = val_paths['source_id'].unique()
val_set_whole = kirk_whole_g24[kirk_whole_g24['source_id'].isin(val_hosps)]
# val_set_whole = balance_labels(val_set_whole, 'source_id', replace = replace)
# val_set_whole = balance_labels(val_set_whole, 'label', replace = replace)
val_paths = val_paths.sample(n = min(len(val_paths), config.num_val), random_state = random_state)
val_set_sample = preprocess_df(val_set_whole, num_hosps = 'all', num_tiles = 200, min_num_slides_p_hosp = 0, min_num_tiles_p_slide = 200, replace = False)

train_dataset = SlideDataset(
        paths=train_paths.full_path.values,
        slide_ids=train_paths.slide_id.values,
        labels=train_paths['label'].values,
        transform_compose=train_transform(full_size=512, crop_size=crop_size, s = train_s)
    )

val_dataset = SlideDataset(
        paths=val_paths.full_path.values,
        slide_ids=val_paths.slide_id.values,
        labels=val_paths['label'].values,
        transform_compose=eval_transform(full_size=512, crop_size=crop_size),
    ) 

val_whole_dataset = SlideDataset(
        paths=val_set_whole.full_path.values,
        slide_ids=val_set_whole.slide_id.values,
        labels=val_set_whole['label'].values,
        transform_compose=eval_transform(full_size = 512, crop_size = crop_size)
    )    

val_sample_dataset = SlideDataset(
        paths=val_set_sample.full_path.values,
        slide_ids=val_set_sample.slide_id.values,
        labels=val_set_sample['label'].values,
        transform_compose=eval_transform(full_size = 512, crop_size = crop_size)
    )

#Create Train Data Loader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)

#Create Val Data Loader
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#Create Test Data Loader
test_inf_dataloader = DataLoader(val_whole_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

#Create Test Data Loader
test_set_wsi_sample_dataloader = DataLoader(val_sample_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)

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
logger = WandbLogger(name = 'pred_grade', save_dir = '/home/jupyter/embeddings/logs', project = 'KIRC')

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

if config.train == 'True':
    #Fit the trainer and start training
    trainer.fit(model, train_dataloader, val_dataloader)
else:
    # kirc_inf_sweep_meta = pd.read_csv('0.75_abstain_kirc_4_wsi.csv')
    # runs = kirc_inf_sweep_meta['Name'].unique() #['elated-sweep-4', 'swept-sweep-5'] 
    # runs_dict = {i.split('-')[-1] : i for i in runs}
    run_name = 'elated-sweep-4' #runs_dict[EXP_VERSION.split('-')[-1]]
    print(run_name)
    model.load_state_dict(torch.load('/home/jupyter/LUAD/Lung/weights/' + run_name))

#Test the model on some unseen test data
# trainer.test(model, test_dataloaders = test_dataloader)
# trainer.test(model, test_dataloaders = val_dataloader)

#Save the model - good practice if you want to retrieve it

if config.save_weights == True:
    torch.save(model.state_dict(), f'/home/jupyter/LUAD/Lung/weights/{EXP_VERSION}')

# test_set.to_csv(f'/home/jupyter/LUAD/Lung/test_sets/{EXP_VERSION}.csv')
#Use these lines to empty the GPU memory
# torch.cuda.empty_cache()
# del model

model = nn.DataParallel(model).to('cuda')
if config.infer == 'True':
    ##############################################################################################################
    print("# Inferring the predictions on a hold out test set slide") 
    # Move the model to GPU
    
    #Create a softmax layer to convert the preds to probs
    smfx_lyr = nn.Softmax(dim=1)

    # Create an empty tensor to store the outputs

    task_store = torch.Tensor([])

    logits = torch.Tensor([])

    with torch.no_grad():
        for (imgs, labels, slide_ids) in tqdm(test_set_wsi_sample_dataloader):            
            # print(labels)
            model.eval()
            
            task_embeddings = model(imgs)   
            task_store = torch.cat([task_store, task_embeddings.cpu()])                        
                    
            # Convert the predictions to probabilities        
            pred_probs = smfx_lyr(model.module.classifier(task_embeddings))
            #Bring pred probs to cpu
            pred_probs = pred_probs.cpu()
            logits  = torch.cat([logits, pred_probs])

            # print('logits.shape', logits.shape) #should n x 2

    # Convert the predictions to a list and save it as a column of the data Frame
    val_set_sample['prob_g2'] = logits[:, 0].flatten().tolist()
    
    val_set_sample.to_csv(f'./preds/{EXP_VERSION}_sample.csv')

    ###
    # task_adata = run_embedding(encoding_array = task_store.detach().cpu().numpy(),
    #                     annotation_df = val_set_sample, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

    # task_adata.write('./embeddings/task_adata' + '_' + EXP_VERSION + '_sample_wsi.h5ad')

    # sc.pl.umap(task_adata, size = 120, projection = '2d', color = ['label'], 
    #     palette="tab20b", color_map=mpl.cm.tab20b, save = '_' + 'grade_' + model_type + '_' + EXP_VERSION + '_sample_wsi.png')

if config.infer_whole == 'True':
    ##############################################################################################################
    print("# Inferring the predictions on a hold out test set slide") 
    # Move the model to GPU
    #Create a softmax layer to convert the preds to probs
    smfx_lyr = nn.Softmax(dim=1)

    # Create an empty tensor to store the outputs
    logits = np.empty((1, config.num_mc_dropout), float)

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
    
    # print('mean_logits', mean_logits.shape)
    # print('std_logits', std_logits.shape)
    
    # print('mean_logits', mean_logits)
    # print('std_logits', std_logits)
    # Convert the predictions to a list and save it as a column of the data Frame
    val_set_whole['prob_healthy'] = mean_logits
    val_set_whole['std_logits'] = std_logits

    ###
    # task_adata = run_embedding(encoding_array = task_store.detach().cpu().numpy(),
    #                     annotation_df = val_set_whole, n_pcs = 50, n_neighbors = 25, use_rapids=False, run_louvain=False)

    # task_adata.write('./embeddings/task_adata' + '_' + EXP_VERSION + '_whole.h5ad')

    val_set_whole.to_csv(f'./preds/{EXP_VERSION}_whole.csv')

wandb.finish()

    # Clear the memory
    # torch.cuda.empty_cache()
    # del model