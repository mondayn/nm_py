import datetime
from dateutil.relativedelta import relativedelta
from zoneinfo import ZoneInfo

def iso():
    ''' return iso datetime string  '''
    return datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d%H%M%S")

def start_of_quarter():
    d2 = datetime.datetime.today() + relativedelta(months=-1)  # shift ahead one month
    d3 = datetime.date(d2.year,((d2.month - 1)//3)*3+1,1)
    return d3.strftime("%Y-%m-%d") 

def str_days_ago(n):
    return (datetime.date.today() - datetime.timedelta(days=n)).strftime("%Y-%m-%d")

# https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes
def iso(ts=datetime.datetime.now()):
    ''' return iso datetime string  '''
    return datetime.datetime.strftime(ts, "%Y%m%d%H%M%S")

def utc_cst(dt): 
    return dt.astimezone(ZoneInfo("America/Chicago"))

def cst_str(dt):
    return dt.strftime("%Y-%m-%d %I:%M:%S %p CST")

def get_dte(months_stepsize):
    '''adhoc load is date inclusive both sides'''
    interval = relativedelta(months=months_stepsize)
    dt_from = datetime.date(2023,2,13)
    cutoff = datetime.date.today()
    while True:
        if dt_from >= cutoff: return
        dt_to = min(cutoff,(dt_from + interval) + relativedelta(days=-1)) 
        yield {'date_from':dt_from.strftime("%Y-%m-%d"),'date_to' : dt_to.strftime("%Y-%m-%d")}        
        dt_from += interval

if __name__ == '__main__':
    print([d for d in get_dte(12)])
