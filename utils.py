def sample_tiles_n_slides(x, num_slides, num_tiles):
        slides_sorted_by_size = x.groupby(['slide_id'])['full_path'].nunique().sort_values(ascending=False).index
        slides = slides_sorted_by_size[:num_slides]
        x = x[x['slide_id'].isin(slides)]
        
        return x.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=False)).reset_index(drop = True)

def sample_tiles_n_slides_from_source(x, num_slides = 5, num_tiles = 100):
    return x.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x = x, num_slides = num_slides, num_tiles = num_tiles)).reset_index(drop = True)


#Code to create training paths and validation paths
def create_tp_vp(lp, 
                GROUP_COL, 
                train_size, 
                random_state, 
                label = None, 
                num_classes = 2, 
                replace = False):
    """
    lp: The DataFrame used 
    Group_col: The column used to split the data into training and validation- used to split the data by patient, for example
    train_size: What proportion / How many rows to keep in training 
    label: The column predicted
    
    """    
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
    #Sample num_slides    
    slides = np.random.choice(x['slide_id'].unique(), size = min(num_slides, len(x['slide_id'].unique())), replace = False)
    x = x[x['slide_id'].isin(slides)]
    x.sample(n=num_tiles, replace=False)
    return x.groupby(['slide_id']).apply(lambda x: x.sample(n=num_tiles, replace=False)).reset_index(drop = True)

def find_largest_hosps(df, num_tiles = 1000, num_sources = 5):  

    # Get the largest hospitals by number of slides
    n_largest_hosps = df.groupby(['source_id'])['slide_id'].nunique().sort_values(ascending=False).index[:num_sources]
    
    # Include slides only from these hospitals. Exclude slides from other hospitals
    df_largest_h = df[df['source_id'].isin(n_largest_hosps)]
    
    # Exclude slides having fewer than "num_tiles" tiles
    filtered = df_largest_h.groupby('slide_id')['full_path'].filter(lambda x: len(x) >= num_tiles)

    return df_largest_h[df_largest_h['full_path'].isin(filtered)]

def data_prep( 
                EXP = wandb.run.name(), 
                num_tiles = 100,
                num_sources = 5, 
                num_slides = 5
            ):
    
    """
    Returns a dataset for the source site prediction task. 
    Removes slides with less than num_tiles tiles. Samples 'num_sources' sources from the input dataframe,
    'num_slides' slides from each source and 'num_tiles' tiles from each slide. 
    """
    
    lp = pd.read_csv('dataset.csv')

    #find the largest hospitals by number of slides, and slides with at least 'num_tiles' tiles and take only those slides for further processing
    lp = find_largest_hosps(df = lp, num_tiles= num_tiles, num_sources = num_sources)  

    #Sample the same number of tiles from each dataset (evenly for each patient)
    lp = lp.groupby(['source_id']).apply(lambda x: sample_tiles_n_slides(x, num_slides, num_tiles)).reset_index(drop = True)

    #Get the IDs as ordinal numbers    
    enc = OrdinalEncoder()
    lp['source_id'] = enc.fit_transform(lp[['source_id']])    
    
    #Save the data
    lp.to_csv('/home/jupyter/LUAD/Lung/lps/' + EXP + '.csv')
 
    return lp