import os, datetime

#region file
def get_stat(file):
    return os.stat(file)

def file_size(file):
    '''file size bytes'''
    return int(get_stat(file).st_size)

def file_ts(file):
    '''file lastmodified datetime'''
    from date import iso
    return iso(datetime.datetime.fromtimestamp(get_stat(file).st_mtime))

def read_file(file): 
    with open(file,'r') as f: 
        return f.read()
    

    #region file
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

import os, concurrent.futures, fnmatch, glob
def get_files(base_path,pattern='*.*',workers=6):
    '''to recurse, pattern should begin with **/ '''
    with os.scandir(base_path) as sc: files = fnmatch.filter([f.path for f in sc if f.is_file()],pattern)
    with os.scandir(base_path) as sc: sub_dirs=[f for f in sc if f.is_dir()]    

    if len(sub_dirs)>0:
        def get_sub_files(path): return [os.path.abspath(os.path.join(path,x)) for x in glob.iglob(pattern,root_dir=path,recursive=True)]
        with concurrent.futures.ThreadPoolExecutor(workers) as executor: results = executor.map(get_sub_files, sub_dirs)
        files += [x for sublist in results for x in sublist]    

    return [f for f in files if '~' not in f]

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


# import concurrent.futures, fnmatch, glob
# def get_files(base_path,pattern='*.*',workers=6):
#     '''to recurse, pattern should begin with **/ '''
#     with os.scandir(base_path) as sc: files = fnmatch.filter([f.path for f in sc if f.is_file()],pattern)
#     with os.scandir(base_path) as sc: sub_dirs=[f for f in sc if f.is_dir()]    

#     if len(sub_dirs)>0:
#         def get_sub_files(path): return [os.path.abspath(os.path.join(path,x)) for x in glob.iglob(pattern,root_dir=path,recursive=True)]
#         with concurrent.futures.ThreadPoolExecutor(workers) as executor: results = executor.map(get_sub_files, sub_dirs)
#         files += [x for sublist in results for x in sublist]    

#     return [f for f in files if '~' not in f]

#endregion



# import os
# import datetime
# from pathlib import Path
# import concurrent.futures

# import re
# import pandas as pd

# from iacom import ia_util 

# def iso_name(file):
#     """puts a datetime at the end of filename

#     Args:
#         file (str): file path

#     Returns:
#         str: filename with datetime stamp
#     """
#     p = Path(file)
#     file = p.parent.joinpath(
#         str(p.stem) + ia_util.iso() + str(p.suffix) 
#     ) 
#     return file

# def _get_file_info(f):
#     stat = os.stat(f)
#     return {
#           'path': f
#             , 'ext': f.stem
#             , 'file_dir': f.parent
#             , 'fileName': f.name
#             , 'file_size': int(stat.st_size)
#             , 'modified': datetime.datetime.strftime(datetime.datetime.fromtimestamp(stat.st_mtime), "%Y-%m-%d %I:%M%p") 
#             , 'days_old': (datetime.datetime.now() - datetime.datetime.fromtimestamp(stat.st_mtime)).days
#            }

# # def get_directory_files(curdir,pattern='*.*'):
# #     """get contents of directory

# #     Args:
# #         curdir (str): directory
# #         pattern (str, optional): regex pattern.  prepending ``**/`` includes subdirectories. Defaults to *.* for all files.

# #     Returns:
# #         dataframe: of files
# #     """    
# #     df = pd.DataFrame([ _get_file_info(f.resolve()) for f in Path(curdir).glob(pattern)])			
# #     # print('# files found',df.shape[0])
# #     if df.shape[0]>0:
# #         return (
# #             df
# #             .sort_values('modified',ascending=False)
# #             .sort_values('file_dir')
# #         )

# def get_files(base_path,pattern='*.*',workers=4):
#     ''' gets files at bath_path, excludes files with ~ in name
#         pattern: prepending ``**/`` includes subdirectories. Defaults to *.* for all files
#         workers: default 4
#     '''    
#     sub_dirs=[d for d in Path(base_path).glob('*') if d.is_dir()]
#     get_files = lambda path: [str(file) for file in Path(path).glob(pattern) if file.is_file() and '~' not in str(file.name)]
#     with concurrent.futures.ThreadPoolExecutor(workers) as executor: results = executor.map(get_files, sub_dirs)
#     return [file for sublist in results for file in sublist]
    
# def get_directory_files(base_path,pattern='*.*',workers=4):    
#     ''' loads files at base_path into dataframe, excludings files with ~ in name
#         pattern: prepending ``**/`` includes subdirectories. Defaults to *.* for all files
#     '''    
#     df = pd.DataFrame([ _get_file_info(Path(f)) for f in get_files(base_path,pattern,workers)])
#     if df.shape[0]>0: return df.sort_values('modified',ascending=False).sort_values('file_dir')

# def keep_most_recent_top(path,pattern,n=1):
#     """keep most recent copies of files whos name matches pattern
    
#     Args:
#         path (str): [description]
#         pattern (regex): case insensitve, include subdirectories by prepending ``**/``
#         n (int, optional): number of files to keep. Defaults to 1.
#     """    
#     df = (
#         get_directory_files(path,pattern)
#         .sort_values('days_old')
#         .reset_index()
#     )
#     for i in range(n,df.shape[0]):
#         os.remove(df.iloc[i].path)

# def delete_files_older_than_n_days(path,pattern,n=31):
#     """remove old files

#     Args:
#         path (str): folder
#         pattern (regex): case insensitve, include subdirectories by prepending ``**/``
#         n (int, optional): minimum age of files kept. Defaults to 31.
#     """    
#     df = (
#         get_directory_files(path,pattern)
#         .query(''' days_old > @n ''')
#     )
#     for i in df.path:
#         os.remove(i)
    

if __name__ == '__main__':
    pass