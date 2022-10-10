# 
# Dependancies setup
#

from lib.classification import perform_decision_trees_analysis, perform_knn_analysis, perform_naive_bayes_analysis, perform_random_forest_analysis, split_train_test_sets
from lib.config.config import *
from lib.data_preparation import check_data_balancing, equal_frequency_descretization, equal_width_descretization, perform_minmax_scaling, perform_oversample, perform_smote, perform_standard_scaling, perform_undersample
from lib.data_profiling import *
import pandas as pd
from numpy import log
from pandas.plotting import register_matplotlib_converters
from matplotlib.pyplot import figure, subplots, Axes, title
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
data = pd.read_csv('./datasets/snp.csv', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, na_values = missing_values)
save_pd_as_csv('snp', data, 'describe')

#
# 1. Data Prifiling
#

check_record_count('snp', data)
check_dataframe_types('snp', data)
check_variable_types('snp', data)
check_missing_values(data)
generate_granularity_single('snp', data)
granularity_study_bins('snp', data)
generate_boxplots('snp', data)
generate_outliers_plot('snp', data)
generate_sparsity_study('snp', data)


#
# Data preparation
#

#data_no_outliers = remove_outliers(data)
#print(f'[!] Original dataset shape: {data.shape}')
#print(f'[!] Data shape with no outliers: {data_no_outliers.shape}')

normalized_data_zscore = perform_standard_scaling('snp', data)
save_pd_as_csv('snp', normalized_data_zscore.describe(), "describe_normalized_zenscore")

normalized_data_minmax = perform_minmax_scaling('snp', normalized_data_zscore)
# restora a coluna 'Bankrupt?' do dataset original
normalized_data_minmax['Bankrupt?'] = data['Bankrupt?']

save_pd_as_csv('snp', normalized_data_minmax.describe(), "describe_normalized_minmax")

#generating plot for comparing the normalization algorithms
fig, axs = subplots(1, 3, figsize=(100,50),squeeze=False)
axs[0, 0].set_title('Original data')
data.boxplot(ax=axs[0, 0])
axs[0, 1].set_title('Z-score normalization')
normalized_data_zscore.boxplot(ax=axs[0, 1])
axs[0, 2].set_title('MinMax normalization')
normalized_data_minmax.boxplot(ax=axs[0, 2])
save_image('snp', 'boxplot_normalized_data_with_no_outliers')

check_data_balancing('snp', normalized_data_zscore, 'Bankrupt?')
undersampled_data = perform_undersample('snp', normalized_data_zscore, 'Bankrupt?')
oversampled_data = perform_oversample('snp', normalized_data_zscore, 'Bankrupt?')
smote_data = perform_smote('snp', normalized_data_zscore, 'Bankrupt?')

equal_width_data = equal_width_descretization('snp', data)
equal_freq_data = equal_frequency_descretization('snp', data)

#
# split training and test sets
#

split_train_test_sets('snp', data, 'snp', 'Bankrupt?')

#
# classification
#

perform_naive_bayes_analysis('snp', 'Bankrupt?')
perform_knn_analysis('snp', 'Bankrupt?')
#perform_decision_trees_analysis('snp', 'Bankrupt?')
#perform_random_forest_analysis('snp', 'Bankrupt?')

print("[!] Done :)")
