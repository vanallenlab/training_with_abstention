from imports import *

dataset = pd.read_csv('data.csv')

if filter_tumor == 'True':
    dataset = dataset[dataset['prob_tumor'] > 0.5]

#Assign labels
kirk_whole_g24.loc[kirk_whole_g24['grade'] == 'G2', 'label'] = 0
kirk_whole_g24.loc[kirk_whole_g24['grade'] == 'G4', 'label'] = 1
kirk_whole_g24['label'] = torch.from_numpy(kirk_whole_g24['label'].values.astype(float)).type(torch.LongTensor) 

kirk_whole_g24_sample = preprocess_df(kirk_whole_g24, num_hosps = num_unhealthy_hosps,
                                     num_tiles = num_tiles, 
                                     min_num_slides_p_hosp = min_num_slides_p_hosp, 
                                     min_num_tiles_p_slide = min_num_tiles_p_slide, 
                                     replace = replace
                                     )

all_hosps = kirk_whole_g24['source_id'].unique()

train_paths, val_paths = create_tp_vp(lp, GROUP_COL, train_size, random_state, label = 'healthy', num_classes = 2, replace = False)

#Keeping track of the sources
train_sources = sorted(list(set(train_paths['source_id'])))
val_sources = sorted(list(set(val_paths['source_id'])))
uniq_srcs = list(set(kirk_whole_g24['source_id']))


# train_paths = train_paths.sample(n = 1000, random_state = random_state)
val_hosps = val_paths['source_id'].unique()
val_set_whole = kirk_whole_g24[kirk_whole_g24['source_id'].isin(val_hosps)]

val_paths = val_paths.sample(n = min(len(val_paths), config.num_val), random_state = random_state)
val_set_sample = preprocess_df(val_set_whole, num_hosps = 'all', num_tiles = 200, min_num_slides_p_hosp = 0, min_num_tiles_p_slide = 200, replace = False)

#Create Train Data Loader
train_dataloader = DataLoader(SlideDataset(
        paths=train_paths.full_path.values,
        slide_ids=train_paths.slide_id.values,
        labels=train_paths['label'].values,
        transform_compose=train_transform(full_size=512, crop_size=crop_size, s = train_s)
    ), batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=10)

#Create Val Data Loader
val_dataloader = DataLoader(SlideDataset(
        paths=val_paths.full_path.values,
        slide_ids=val_paths.slide_id.values,
        labels=val_paths['label'].values,
        transform_compose=eval_transform(full_size=512, crop_size=crop_size),
    ) , batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=10)


monitor_var = 'val_task_loss'
if config.model_type == 'normal':
    # Create model #mc_lightning_be_cor.models.resnet.resnet_module.
    model = model_choices[config.normal_model](hparams = {'lr' : 1e-5, 
                                                'num_classes' : 2,
                                                'weight_decay' : 1e-5,
                                                'dropout' : 0.1,
                                                'confidence_threshold' : 0.9,
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
logger = WandbLogger(name = 'pred_grade', save_dir = '/home/jupyter/embeddings/logs', project = 'KIRC')

callbacks.append(early_stop_callback)

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

torch.save(model.state_dict(), f'/home/jupyter/LUAD/Lung/weights/{EXP_VERSION}')

model = nn.DataParallel(model).to('cuda')
