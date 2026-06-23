#region pandas
def print_shape(df,msg=None):
    ''' for chaining  '''
    if msg: print(msg)  
    print(df.shape)
    return df

def print_cols(df):
    ''' for chaining  '''  
    print(list(df.columns))
    return df

def clean_names(df): return df.rename(columns=clean_and_lower)

def trim_and_rename(_df,rename_dict) -> pd.DataFrame:
    ''' rename necessary columns, drop the others '''
    return _df.rename(columns=rename_dict)[rename_dict.values()]

def add_col(self,**kwargs):
    ''' assign new columns using itertuples, e.g. 
        df.add_col(
            new_col=lambda x: 'y' if 'F' in x.state_code else 'N'
            ,new_col2=lambda x: 'y' if 'G' in x.state_code else 'N'
        )
     '''
    _df = self.copy()
    for k, v in kwargs.items():
        _df[k] = list(map(v,_df.itertuples()))
    return _df

def get_dupes(df,column_names): 
    return df.loc[df.duplicated(subset=column_names, keep=False)]

def take_first(df,subset,by,ascending=True):
    return df.sort_values(by=by, ascending=ascending).drop_duplicates(subset=subset, keep="first").sort_values(subset)

def collapse_levels(_df):
    ''' removes multi index column names after pivot_table '''
    _df.columns = _df.columns.map('_'.join)
    return _df

def search_df(df,regex):
    ''' return matching rows & columns  '''
    
    # cast everything as string
    df = df.astype('str')
    regex = str(regex)

    # create a matrix of true/false and then filter by it
    filter_df = pd.DataFrame({c:df[c].str.contains(regex) for c in df.columns})
    df = df.where(filter_df)
    
    # which rows, cols are empty 
    nanrows = df.index[df.isna().all(axis=1)]
    nancols = df.columns[df.isna().all(axis=0)]

    return df.drop(index=nanrows).drop(columns=nancols).fillna('')

def filter_str(df,column_name,search_string,complement=False):
    """ query a string column of a dataframe """
    df = df.astype({column_name:'str'}) 
    criteria = df[column_name].str.contains(search_string)
    if complement:
        return df[~criteria]
    return df[criteria]

def remove_dup_cols(_df):
    ''' gets the last instance of column by clean name'''
    cols = {}
    for i,x in enumerate(_df.clean_names().columns):
        cols[x]=i
    return _df.iloc[:,list(cols.values())]

def my_clipboard(df):
    ''' for copying to excel '''
    df.rename(columns = lambda x: x.replace('_',' ').strip()).to_clipboard(index=False)

def view_short(df):
    ''' displays with condensed column info '''
    from IPython.display import display

    cols = {k:k.replace('_',' ')[:99] for k in df.columns}
    with pd.option_context(
        'display.max_colwidth', 25,  #default 50
        'display.float_format', '{:,.0f}'.format
    ):
        display(df.rename(columns=cols).replace(0,'-'))
    # return df  # 2022-11-11 otherwise jupyter shows twice

def change_dtypes(df,cols,dtype):
    ''' updates a column (or list of columns) to a specified data type e.g. int or float '''
    if 'list' not in str(type(cols)):
        cols = [cols]    
    return df.assign(**{c: df[c].astype(dtype) for c in cols})

def profile(df):
    ''' returns summary stats for a dataframe. also try df.info() '''    
    pd.options.display.float_format = '{:.0f}'.format

    df2 = pd.DataFrame()
    for c in df.columns:
        d = dict()
        d['col'] = c
        d['dtype'] = str(type(df.iloc[0][c])).replace("<class '",'').replace('numpy.','').replace("'>",'')
        d['nulls'] = df[df[c].isna()].shape[0]
        d['zeros'] = df[df[c].astype('str').replace('\.\d*','',regex=True) == '0'].shape[0]
        d['empty']=df[df[c].astype('str').replace(" ",'').apply(lambda x: len(x)==0)].shape[0]
        d['min'] = df[~df[c].isna()][c].min()
        d['max'] = df[~df[c].isna()][c].max()
        d['unique'] = df[c].nunique()

        lst = [len(x) for x in df[c].astype('str').unique()]
        d['minLength'] = min(lst)
        d['maxLength'] = max(lst)

        d['count'] = df[c].shape[0]

        df1 = pd.DataFrame.from_dict(d,orient='index') 
        df2 = pd.concat([df2,df1],axis=1)
    return df2.T.reset_index().drop('index',axis=1)

def coalesce(df,cols,target):
    outcome = df[cols].bfill(axis="columns").ffill(axis="columns").iloc[:, 0]
    return df.assign(**{target: outcome}).drop(columns=cols)
