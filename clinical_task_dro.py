from imports import *

hyperparameter_defaults = dict(
    min_num_slides_p_hosp = 5,
    train_s = 0,
    col_sch = 'RGB',
    crop_size = 224,
    batch_size = 128,
    sample = "without_replacement",
    split = 'source_id',
    iter = 2,
    log_weight_decay = -5,
    patience = 5,
    NUM_HOLD_OUT_HOSPS = 1,
    normal_model = 'PretrainedResnet50FT_Hosp_DRO_abstain',
    log_lr = -5,
    min_num_tiles_p_slide = 100,
    dropout = 0.0,
    accumulate_grad_batches=1,
    val_check_interval=0.25,
    num_test = 1000,
    num_val = 1000,
    confidence_threshold = 0.5,
    max_steps = 1000,
    early_stop = 'True',
    temp_scale = 'True',
    filter_tumor = 'True',
  ) 

dataset = pd.read_csv('data.csv')

if config.filter_tumor == 'True':
    dataset = dataset[dataset['prob_tumor'] > 0.5]

#Assign labels

dataset['label'] = torch.from_numpy(dataset['label'].values.astype(float)).type(torch.LongTensor) 

dataset_sample = preprocess_df(dataset, num_hosps = config.num_unhealthy_hosps,
                                     num_tiles = config.num_tiles, 
                                     min_num_slides_p_hosp = config.min_num_slides_p_hosp, 
                                     min_num_tiles_p_slide = config.min_num_tiles_p_slide, 
                                     replace = False
                                )

train_paths, val_paths = create_tp_vp(lp, config.GROUP_COL, train_size = 0.7, random_state = config.iter, label = 'label', num_classes = 2, replace = False)

val_paths = val_paths.sample(n = min(len(val_paths), config.num_val), random_state = random_state)

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

model = PretrainedResnet50FT_Hosp_DRO_abstain(hparams = {'lr' : 10**config. log_lr, 'num_classes' : 2, 'weight_decay' : 10**config.log_weight_decay, 'dropout' : config.dropout, 'confidence_threshold' : config.confidence_threshold})

      
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
logger = WandbLogger(name = 'task', save_dir = '.', project = '')

callbacks = []
callbacks.append(early_stop_callback)

#Create a trainer obj
trainer = pl.Trainer(callbacks=callbacks, logger=logger, val_check_interval=config.val_check_interval, accumulate_grad_batches=config.accumulate_grad_batches, max_steps=config.max_steps)

#Fit the trainer and start training
trainer.fit(model, train_dataloader, val_dataloader)    