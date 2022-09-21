# 
# Dependancies setup
#

from lib.config.config import *
from lib.data_profiling import *
import pandas as pd
from numpy import log
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, savefig, subplots, Axes, title
from lib.config.ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_line_chart
from pandas import DataFrame, Series    
from seaborn import distplot, heatmap
from scipy.stats import norm, expon, lognorm
from lib.utils import *

register_matplotlib_converters()

#
# Loading the dataset
#

print("[!] Loading the dataset...")

missing_values = ["NA", "n/a", "na", "?", "--"]
data = pd.read_csv('./datasets/taiwan.csv', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, na_values = missing_values)
save_pd_as_csv(data, 'describe')

#
# 1. Data Prifiling
#

#check_record_count(data)
#check_dataframe_types(data)
#check_variable_types(data)
#check_missing_values(data)
#generate_granularity_single(data)
granularity_study_bins(data)

print("[!] Done.")