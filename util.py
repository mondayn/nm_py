"""
miscellaneous functions

"""
#region
def shortcuts():
    print('''
pd.options.display.float_format = '{:,.2f}'.format
pd.set_option('display.max_rows', 1000) 
pd.set_option('display.max_columns', 30) 
pd.set_option('max_colwidth', 0)
          
with pd.option_context('display.max_rows', None):

%load_ext autoreload
%autoreload 2

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 15, 4
''')
    

def get_username():
    ''' on mac '''
    from pwd import getpwuid
    import os
    return getpwuid(os.getuid()).pw_name

def matches_any(s,lst):
    ''' True if s contains any characters in lst, case-insensitve '''
    return any(
        x.lower() in str(s).lower() 
        or str(s).lower() in x.lower()
    for x in lst)

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

def chunks(lst, n): 
    for i in range(0, len(lst), n): yield lst[i:i + n]

def batch(lst,n):
    size = int(len(lst)/n)+1
    return chunks(lst,size)

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

from functools import wraps
import datetime
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
            print(f'{fn_name}records={result}, {kwargs} duration={str(datetime.timedelta(seconds=(te - ts)))}')
            return result
        except Exception as e:
            return e
        # return result
    return log_duration_wrapper1

def fprint(x): print(x); return x;

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
        # if 127<=line<=302: 
        print(s)
#endregion

if __name__ == '__main__':
    print(matches_any('this',['that','ours','thisthat']))
    pass


import zipfile
from pathlib import Path
from date import iso

def run_zip():
    p = Path(r"...")
    dest=r''+iso()+'.zip'
    print(dest)
    with zipfile.ZipFile(dest,mode='w') as zf:
        for f in p.glob('*'):
            if f.is_file():
                print(f.name)
                zf.write(f,f.relative_to(p), compress_type = zipfile.ZIP_DEFLATED)

if __name__ == "__main__":
    run_zip()
