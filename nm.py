import pandas as pd

#region jupyter

pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 1000) 
pd.set_option('display.max_columns', 30) 
pd.set_option('max_colwidth', 0)

# %load_ext autoreload
# %autoreload 2

# from IPython.core.display import HTML
# HTML("<style>.container{ width: 90% !important; }</style>")

# from matplotlib.pylab import rcParams
# rcParams['figure.figsize'] = 15, 4

#endregion

#region helpers
def matches_any(s,lst):
    ''' True if s contains any characters in lst, case-insensitve '''
    return any(x.lower() in str(s).lower() for x in lst)

def find_matching_element(s, string_list):
    ''' returns first matching element, case-insensitive'''
    return next((item for item in string_list if s.lower() in item.lower()), None)

def clean_and_lower(s):
    ''' for small capping column names '''
    return s.lower().replace('\n','').replace(' ','_').replace('(','_').replace(')','_')

def flatten(items):
    for i in items:
        if isinstance(i,(list,set,tuple)): yield from flatten(i)
        else: yield i

from functools import reduce
def thread_first(val, *forms):
    ''' credit: https://toolz.readthedocs.io/en/latest/_modules/toolz/functoolz.html#thread_last
        cytoolz is faster!
    '''
    def evalform_front(val, form):
        if callable(form):
            return form(val)
        if isinstance(form, tuple):
            func, args = form[0], form[1:]
            args = (val,) + args
            return func(*args)
    return reduce(evalform_front, forms, val)

def try_cast(s='str'):
    ''' cast as float otherwise zero '''
    i = 0.0
    if s != s:  # for nan
        return i
    try:
        i = float(s)
    except:
        pass
    return i
# thread last


#endregion

#region logging
import logging
from logging import handlers
import sys

from functools import wraps
def track_duration(fn):
    import time
    @wraps(fn)
    def log_duration_wrapper1(*args,**kwargs):
        MAX_FN_NAME_LEN=15
        # result = None
        # __name__=fn.__name__
        try:
            fn_name = fn.__name__.ljust(MAX_FN_NAME_LEN)[:MAX_FN_NAME_LEN] 
            ts = time.time()
            # print(fn_name + ' | starting ')
            result = fn(*args,**kwargs)
            te = time.time()
            print(f'{fn_name}records={result}, duration={str(datetime.timedelta(seconds=(te - ts)))}')
            return result
        except Exception as e:
            return e
        # return result
    return log_duration_wrapper1

# logger = logging.getLogger()
# logger.setLevel(logging.INFO)

# #log any unexpected errors
def handle_unhandled_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))
sys.excepthook = handle_unhandled_exception

def add_handler(handler,level=[]):
    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    handler.setFormatter(fmt)
    if level:
        handler.setLevel(level)
    logger.addHandler(handler)

# add_handler(logging.StreamHandler())
# add_handler(handlers.RotatingFileHandler(LOG_TO, mode='a+', maxBytes=10000, backupCount=0))

# EMAIL_ON_FAIL = {
#     'mailhost':'smtp@gmail.com'
#     ,'fromaddr':'description <name@gmail.com>'
#     ,'toaddrs':[
#             'name@gmail.com'
#         ]
#     ,'subject': 'ERROR:' + LOG_TO
# }
# # add_handler(handlers.SMTPHandler(**EMAIL_ON_FAIL),logging.ERROR)
#endregion logging

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

def clean_names(df):
    return df.rename(columns=clean_and_lower)

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

def exists(v):
    try:
        if len(v)>0:
            return True
    except:
        return False

    
def coalesce(df,cols,target):
    outcome = df[cols].bfill(axis="columns").ffill(axis="columns").iloc[:, 0]
    return df.assign(**{target: outcome}).drop(columns=cols)

#chaining warn
# with pd.option_context('mode.chained_assignment', None):
#     _df['testcoid'] = c
#     _df['alg_score'] = clf.decision_scores_.copy()


# https://stackoverflow.com/questions/34376896/pandas-dataframes-how-to-wrap-text-with-no-whitespace
# df['user_agent'] = df['user_agent'].str.wrap(100) #to set max line width of 100

# def truncate_columns(df):
#     return df.rename(columns={c:c[:10] for c in df.columns})

# @sqlalchemy.event.listens_for(TARGET_ENGINE, 'before_cursor_execute')
# def receive_before_cursor_execute(conn, cursor, statement, parameters,context, executemany):
#     if executemany:
#         cursor.fast_executemany = True

# def stack_files(files):
#     ''' @stack_files(list_of_files)
#         def parse(file):
#             ...
#         runs the parse function for each file and concats as a df
#      '''
#     def fx_wrapper(parse_fx):
#         def arg_wrapper(*args, **kwargs):
#             return reduce(lambda _,file: pd.concat([_,parse_fx(file)]),files,pd.DataFrame())
#         return arg_wrapper
#     return fx_wrapper

for fx in [
        collapse_levels,get_dupes,take_first,search_df,filter_str,remove_dup_cols,
        my_clipboard,view_short,change_dtypes,profile,trim_and_rename,
        add_col,print_shape,print_cols,clean_names
]:
    setattr(pd.DataFrame, fx.__name__, fx)
# from pandas.core.base import PandasObject
# PandasObject.print_shape = print_shape

#endregion pandas

#region datetime
import datetime
from dateutil.relativedelta import relativedelta
def start_of_quarter():
    d2 = datetime.datetime.today() + relativedelta(months=-1)  # shift ahead one month
    d3 = datetime.date(d2.year,((d2.month - 1)//3)*3+1,1)
    return d3.strftime("%Y-%m-%d") 

def str_days_ago(n):
    return (datetime.date.today() - datetime.timedelta(days=n)).strftime("%Y-%m-%d")

def iso(ts=datetime.datetime.now()):
    ''' return iso datetime string  '''
    return datetime.datetime.strftime(ts, "%Y%m%d%H%M%S")
#endregion datetime

#region file
def get_stat(file):
    import os
    return os.stat(file)

def file_size(file):
    '''file size bytes'''
    return int(get_stat(file).st_size)

def file_ts(file):
    '''file lastmodified datetime'''
    return iso(datetime.datetime.fromtimestamp(get_stat(file).st_mtime))

import concurrent.futures
from pathlib import Path
def get_files(base_path,pattern='*.*',workers=4):
    ''' gets files at bath_path, excludes files with ~ in name '''    
    sub_dirs=[d for d in Path(base_path).glob('*') if d.is_dir()]
    get_files = lambda path: [str(file) for file in Path(path).glob(pattern) if file.is_file() and '~' not in str(file.name)]
    with concurrent.futures.ThreadPoolExecutor(workers) as executor: results = executor.map(get_files, sub_dirs)
    return [file for sublist in results for file in sublist]
#endregion

#region excel
from copy import copy
def to_xltemplate(df,openpyxl_xlworkbook,sheetname='Sheet1',cell_range='A2',index=False,header=False):
    """ loads a dataframe into an excel template.  carries forward any numerical, horizontal and vertical styles of the first row:: 

        import openpyxl
        wb = openpyxl.load_workbook('template.xlsx')
        df.to_xltemplate(
            wb
            ,sheetname='Sheet2'
            ,cell_range='A2'
            ,index=False,header=False
        )
        wb.save('test3.xlsx')  # copy of template with data in it

    Args:
        df (pd.DataFrame): data in
        openpyxl_xlworkbook (openpyxl): excel template
        sheetname (str, optional): target tab in template. Defaults to 'Sheet1'.
        cell_range (str, optional): top-left cell of the range we're targeting. Defaults to 'A2'.
        index (bool, optional): whether or not to output the dataframe index. Defaults to False.
        header (bool, optional): whether or not to output the dataframe columns. Defaults to False.

    Returns:
        dataframe
    """    
    import openpyxl
    from openpyxl.utils.dataframe import dataframe_to_rows  # have to import this way due to their __init__
    ws = openpyxl_xlworkbook[sheetname]

    # translate excel range to starting position coordinates
    starting_row, starting_column = openpyxl.utils.coordinate_to_tuple(cell_range)

    # find row whose style we want to carry forward
    if header: 
        style_row = starting_row + 1
    else: 
        style_row = starting_row
    style_number_formats, style_alignments = [], []

    # loop through dataframe, updating values and applying formats
    rows = dataframe_to_rows(df, index=index, header=header)
    for r_idx, row in enumerate(rows, starting_row):
        for c_idx, value in enumerate(row, starting_column):
            c = ws.cell(r_idx,c_idx)
            c.value = value   # post the value
            if r_idx == style_row:  # save style row formats
                style_number_formats.append(c.number_format)
                style_alignments.append(copy(c.alignment))
            elif r_idx > style_row:  # apply style row formats
                c.number_format = style_number_formats[c_idx - starting_column]                
                c.alignment = style_alignments[c_idx - starting_column]

    return df
#endregion

#region active directory
def get_members(dn='CN=*,OU=g00958,OU=00958,DC=*,DC=*,DC=net'):
    from pyad import adquery
    q = adquery.ADQuery()
    idx = ['cn','sn','title']
    # idx += ['mail','givenname']
    q.execute_query(attributes=idx,where_clause=''' memberof='{}' '''.format(dn))
    return pd.DataFrame(q.get_results())[idx]
#endregion

#region meta
def print_functions():
    import inspect
    fx = [(name,obj) for name, obj in globals().items() if callable(obj) and obj.__module__ == __name__ ]
    for name, obj in fx:
        s = f'{name}'
        line = inspect.getsourcelines(obj)[1]
        # s += f'\t:{line}'
        docstring = inspect.getdoc(obj)
        # if docstring:
        #     s+='\t'+docstring[:15]
        if 127<=line<=302: 
            print(s)
#endregion

if __name__ == '__main__':
    # print(str_days_ago(4))
    print_functions()
    # print(pd.DataFrame({'COL':[1,2,3]}).print_cols().print_shape())
    # 
    # pass