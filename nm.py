import pandas as pd



#endregion

#region logging
import sys
import logging
from logging import handlers

LOG_TO = 'log.log' 
print('logging to ',LOG_TO)
logger = logging.getLogger()

format = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
handlers = [logging.StreamHandler(),logging.handlers.RotatingFileHandler(LOG_TO, mode='a+',maxBytes=10000, backupCount=4),]
logging.basicConfig(format=format,level=logging.INFO,handlers=handlers)

# emailhandler=logging.handlers.SMTPHandler(mailhost='smtp@gmail.com',fromaddr='description <name@gmail.com>',toaddrs=['name@gmail.com'],subject='ERROR:' + LOG_TO)
# emailhandler.setLevel(logging.ERROR)
# emailhandler.setFormatter(logging.Formatter(format))
# logger.addHandler(emailhandler)

sys.excepthook = lambda exc_type, exc_value, exc_traceback: logger.critical("Unhandled exception", exc_info=(exc_type, exc_value, exc_traceback))

def print_log(msg): print(msg); logger.info(msg)

#endregion logging


def exists(v):
    try:
        if len(v)>0:
            return True
    except:
        return False

    

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


for fx in [
        collapse_levels,get_dupes,take_first,search_df,filter_str,remove_dup_cols,
        my_clipboard,view_short,change_dtypes,profile,trim_and_rename,
        add_col,print_shape,print_cols,clean_names
]:
    setattr(pd.DataFrame, fx.__name__, fx)
# from pandas.core.base import PandasObject
# PandasObject.print_shape = print_shape

#endregion pandas


#endregion

if __name__ == '__main__':
    pass