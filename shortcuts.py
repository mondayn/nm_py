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

# with warnings.catch_warnings(action="ignore",category=UserWarning):
#     fxn()