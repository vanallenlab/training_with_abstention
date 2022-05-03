def sample_tiles_n_slides(x, num_slides, num_tiles):
        slides_sorted_by_size = x.groupby(['slide_id'])['full_path'].nunique().sort_values(ascending=False).index
        slides = slides_sorted_by_size[:num_slides]
        # slides = np.random.choice(x['slide_id'].unique(), size = min(num_slides, len(x['slide_id'].unique())), replace = False)
        x = x[x['slide_id'].isin(slides)]
        
        print(x.groupby(['slide_id']).agg('count'))

        return x.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=False)).reset_index(drop = True)

def sample_tiles_n_slides_from_source(x, num_slides = 5, num_tiles = 100):
    return x.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x = x, num_slides = num_slides, num_tiles = num_tiles)).reset_index(drop = True)

def create_tp_vp(lp, 
                GROUP_COL, 
                train_size, 
                random_state, 
                label = None, 
                num_classes = 2, 
                replace = False):
    
    gss = GroupShuffleSplit(n_splits=100, train_size=train_size, random_state=random_state)
    splits = list(gss.split(X = list(range(len(lp))), groups = lp[GROUP_COL]))

    splits_iterator = iter(splits)

    while any([(x not in y) for (x, y) in itertools.product(list(range(num_classes)), [td_ctr, vd_ctr])]):
        
        train_idx, val_idx = next(splits_iterator)

        #Create train and val paths
        train_paths, val_paths = lp.iloc[train_idx] , lp.iloc[val_idx]

        val_paths, train_paths = balance_labels(val_paths, label, replace = replace), balance_labels(train_paths, label, replace = replace)    
        
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


def sample_tiles_n_slides(x, num_slides, num_tiles):
        slides = np.random.choice(x['slide_id'].unique(), size = min(num_slides, len(x['slide_id'].unique())), replace = False)
        x = x[x['slide_id'].isin(slides)]
        x.sample(n=num_tiles, replace=False)
        return x.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=False)).reset_index(drop = True)

def find_largest_hosps(df, num_tiles = 1000, n = 5):  

    n_largest_hosps = df.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False).index[:n]
    
    # print('n largest hosps', n_largest_hosps)    

    # print('hospitals sorted by slide', df.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False))
    
    df_largest_h = df[df['source_id'].isin(n_largest_hosps)]
    
    filtered = df_largest_h.groupby('slide_id')['full_path'].filter(lambda x: len(x) >= num_tiles)

    # print('hospitals with at least num_tiles tiles', df_largest_h[df_largest_h['full_path'].isin(filtered)])

    return df_largest_h[df_largest_h['full_path'].isin(filtered)]

def get_tiles(normalize, unhealthy):
    #Get the case IDs for the patients
    unhealthy['case_id'] = unhealthy['slide_id'].str.slice(0, len('TCGA-44-6144'))
    #Get the IDs for the Sources
    unhealthy['source_id'] = unhealthy['slide_id'].str.slice(len('TCGA-'), len('TCGA-44'))  
    #Change the file paths
    unhealthy.full_path = unhealthy.full_path.str.replace('/mnt/disks/data_disk/', '/home/jupyter/')
    if normalize == 'staintools':
        imgs_we_have = [img[:-len('_t2.png')] + '.png' for x in tqdm(os.walk('/home/jupyter/LUAD/')) for img in glob(os.path.join(x[0], '*t2.png'))]
        unhealthy = unhealthy[unhealthy['full_path'].isin(imgs_we_have)]
        unhealthy['full_path'] = unhealthy['full_path'].str[:-len('.png')] + '_t2.png'
    return unhealthy

def data_prep(normalize = 'staintools', 
                random_state = 1,
                num_tiles = 10,
                pre_train_infer = True,
                post_train_infer = False, 
                EXP = 'wandb', 
                num_sources = 5, 
                num_slides = 5
            ):
    
    if EXP == 'wandb':
        run = wandb.init(config = {})
        EXP = wandb.run.name
        
    print('EXP', EXP)
 
    #Read the unhealthy dataset
    unhealthy = pd.read_csv('/home/jupyter/LUAD/Lung/data/lung_paths.csv')
    #Read the healthy dataset
    healthy = pd.read_csv('/home/jupyter/LUAD/Lung/data/lung_healthy_paths.csv')

    lp = pd.concat([unhealthy, healthy])

    lp = get_tiles(normalize, lp)    
    
    #find the largest hospitals and take only those slides for further processing
    lp = find_largest_hosps(df = lp, num_tiles= num_tiles, n = num_sources)  

    print(lp.groupby('source_id').nunique())
    
    lp.reset_index(inplace = True, drop = True)

    #Sample the same number of tiles from each dataset (evenly for each patient)
    # print(unhealthy.groupby(['slide_id']).size().sort_values(ascending=False))

    lp = lp.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x, num_slides, num_tiles)).reset_index(drop = True)

    print("lp.nunique")
    print(lp.nunique)
    lp.reset_index(drop = True, inplace = True)

    #Get the IDs as ordinal numbers
    enc = OrdinalEncoder()
    lp['source_id'] = enc.fit_transform(lp[['source_id']])    
    # lp['slide_id'] = enc.fit_transform(lp[['slide_id']])
    lp['source_id'] = torch.from_numpy(lp['source_id'].values.astype(float)).type(torch.LongTensor) 
    # lp['slide_id'] = torch.from_numpy(lp['slide_id'].values.astype(float)).type(torch.LongTensor) 

    #See the data so you know it is right
    # print(lp.columns)
    # print(lp.head())

    lp.to_csv('/home/jupyter/LUAD/Lung/lps/' + EXP + '.csv')
 
    #Pretraining embeddings
    if pre_train_infer == True:
        model = PretrainedResnet50FT(hparams = {'num_classes' : 5})
        model = nn.DataParallel(model).to('cuda')

        create_adata(paths = lp, run_name = EXP, model = model, sup = 'unsup')        
        # EXP = "astral-jazz-56"
        # create_umap_w_tiles(EXP, outer_box = 'num_srcs_clusters', inner_box = 'source_id')
        # create_umap_w_tiles(EXP, outer_box = 'num_slds_clusters', inner_box = 'slide_id')

        create_umap_w_tiles(run_name = EXP, outer_box = 'source_id', inner_box = 'slide_id', sup = 'unsup', outer_bw = 0)


    if post_train_infer == True:
        
        model = PretrainedResnet50FT_contrastive_multitask(hparams = {'lr' : 1e-7, 
        'num_classes' : num_sources, 
        'srcs_map' : None, 
        'include_ce_loss' : None})
        
        model_name = EXP #"ce_loss:False,tsk:src,srcs:5,sp:x,img:RGB,s:0,ts:1059|0407,fold:1"

        # # Load the pre-trained weights
        model.load_state_dict(torch.load('/home/jupyter/LUAD/Lung/weights/weights_' + model_name))

        create_adata(paths = lp, run_name = EXP, model = model, sup = 'sup_star')        

        create_umap_w_tiles(run_name = EXP, outer_box = 'source_id', inner_box = 'slide_id', sup = 'sup_star', outer_bw = 0)

        #######

        model = PretrainedResnet50FT(hparams = {'lr' : 1e-7, 
                                    'num_classes' : num_sources
                                    })
        
        model_name = EXP #"task=source,hosps=5largest,pts=atleast,tiles=1000,split=patient,balance=pt&label,val=0.3,image=RGB,sample=wo,,estop5,timestamp=1441|0404fold0"

        # # Load the pre-trained weights
        model.load_state_dict(torch.load('/home/jupyter/LUAD/Lung/weights/weights_' + model_name))

        create_adata(paths = lp, run_name = EXP, model = model, sup = 'sup_ce')

        create_umap_w_tiles(run_name = EXP, outer_box = 'source_id', inner_box = 'slide_id', sup = 'sup_ce', outer_bw = 0)
    
    return lp