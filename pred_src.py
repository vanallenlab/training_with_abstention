from imports import *
from data_prep import data_prep
from umap_w_tiles_bdries import *

def pred_source():
    
    hyperparameter_defaults = dict(
        iter = 1,
        train = True,       
        num_srcs = 5,
        sp = 'slide_id',
        img = 'RGB',
        s = 1,
        dropout = 0.0,
        log_weight_decay = -5,
        infer = True,
        normalize = 'not_staintools',
        val_stain = 'diff',
        num_tiles = 100,
        num_slides = 5,
        include_all_val = 'False',
        include_num_confident = 'False',
        temp_scale = 'True',
        normal_model = 'PretrainedResnet50FT_Hosp_DRO_abstain',
        confidence_threshold = 0.9,
        infer_whole_slides = 'False',
        bw = 'None'
    )

    # Pass your defaults to wandb.init
    run = wandb.init(config=hyperparameter_defaults)
    config = wandb.config
    random_state = config.iter
    train = config.train
    num_srcs = config.num_srcs
    grp_col = config.sp
    col_sch = config.img
    s = config.s

    random.seed(a=random_state, version=2)
    seed_everything(random_state)

    #Keeping track of hyperparameter configuration
    EXP = f'CE,tsk:src,nrm:{config.normalize},val_stain:{config.val_stain},srcs:{num_srcs},sp:{config.sp},img:{config.img},s:{s},fold:{config.iter},num_tiles:{config.num_tiles},num_slides:{config.num_slides}'

    if col_sch == 'RGB':
        train_transform = RGBTrainTransform
        eval_transform = RGBEvalTransform
    elif col_sch == 'HSV':
        train_transform = HSVTrainTransform  
        eval_transform = HSVEvalTransform
    
    #Prepare the data
    lp =  data_prep(normalize = config.normalize, random_state = random_state, num_tiles = config.num_tiles, EXP = EXP, num_sources = num_srcs, num_slides = config.num_slides)
    
    #Create group splitting object
    gss = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=random_state)
    
    train_idx, val_idx = list(gss.split(X = list(range(len(lp))), groups = lp[grp_col]))[0]

    #Create train and val paths
    train_paths, val_paths = lp.iloc[train_idx] , lp.iloc[val_idx]

    #Balance by source site
    val_paths, train_paths = balance_labels(val_paths, 'source_id'), balance_labels(train_paths, 'source_id')
    
    #If stain normalizing, read file from previously written set of stain normalized tiles
    if (config.normalize == 'staintools') & (config.val_stain != 'same'): 
      val_paths['full_path'] = val_paths['full_path'].str[:-len('_t2.png')] + '_t1.png'

    #Just some checks to check that the patients don't overlap
    assert set(val_paths[grp_col]).intersection(set(train_paths[grp_col])) == set()

    #Create Train Dataset
    train_dataset = SlideDataset(
            paths=train_paths.full_path.values,
            slide_ids=train_paths.case_id.values,
            labels=train_paths.source_id.values,
            transform_compose=train_transform(full_size = 512, crop_size = 224, s = s),
            bw = config.bw
        )

    #Create Val Dataset
    val_dataset = SlideDataset(
            paths=val_paths.full_path.values,
            slide_ids=val_paths.case_id.values,
            labels=val_paths.source_id.values,
            transform_compose=eval_transform(full_size = 512, crop_size = 224),
            bw = config.bw
        )

    #Create Train Data Loader
    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, pin_memory=True, num_workers=10)

    #Create Val Data Loader
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, pin_memory=True, num_workers=10)

    #Create model
    
    model = model_choices[config.normal_model](hparams = {'lr' : 1e-5, 
                                                'num_classes' : num_srcs,
                                                'weight_decay' : 10 ** config.log_weight_decay,
                                                'dropout' : config.dropout,
                                                'confidence_threshold' : config.confidence_threshold,
                                                'include_all_val' : config.include_all_val,
                                                'include_num_confident' : config.include_num_confident
                                                })

    if (config.temp_scale == 'True') and (config.normal_model == 'PretrainedResnet50FT_Hosp_DRO_abstain'):    
        model.set_temperature(val_dataloader)

    #Create Early Stopping Callback
    early_stop_callback = EarlyStopping(
        monitor='val_task_loss',
        min_delta=0.00,
        patience=5,
        verbose=False,
        mode='min'
    )

    #Create a tb logger
    logger = WandbLogger(name = wandb.run.name, save_dir = './logs', project = 'pred_source')
    
    #Create a trainer obj
    trainer = pl.Trainer(gpus=-1, 
                        distributed_backend='ddp',
                        max_epochs=100, 
                        callbacks=[early_stop_callback], 
                        logger=logger,
                        limit_train_batches=1.0,
                        val_check_interval=0.25,
                        limit_val_batches=1.0,
                        log_every_n_steps=1,
                        max_steps=5000,
                        fast_dev_run=0)

    if train:
        #Fit the trainer and start training
        trainer.fit(model, train_dataloader, val_dataloader)

        #Save the model - good practice if you want to retrieve it 
        torch.save(model.state_dict(), './starfish_wandb_wts/starfish_wandb_wts' + '_' + EXP)

        val_paths.to_csv(f'/home/jupyter/LUAD/Lung/pred_source/val_paths/{EXP}.csv')


if __name__ == '__main__':
    pred_source()