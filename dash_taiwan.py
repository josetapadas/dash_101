# 
# Dependancies setup
#

from lib.classification import *
from lib.config.config import *
from lib.data_preparation import *
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
data = pd.read_csv('./datasets/taiwan.csv', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, na_values = missing_values)
save_pd_as_csv('taiwan', data, 'describe')

#
# 1. Data Profiling
#

# check_record_count('taiwan', data)
# check_dataframe_types('taiwan', data)
# check_variable_types('taiwan', data)
# check_missing_values(data)
# generate_granularity_single('taiwan', data)
#granularity_study_bins('taiwan', data)
#generate_textual_correlation_table('taiwan', data)
#generate_correlation_heatmap('taiwan', data)
# generate_boxplots('taiwan', data)
# generate_outliers_plot('taiwan', data)
# generate_sparsity_study('taiwan', data)

#
# Data preparation
#

# remove list of variables
#columns_to_drop = []  // choose this line if do not wish to remove any variable from the analysis
# Features to remove due to IRRELEVANCE
#columns_to_drop = np.array([ 6, 7, 8, 9, 10, 14, 21, 24, 25, 26, 27, 28, 30, 31, 33, 34, 36, 41, 46, 47, 51, 53, 58, 59, 62, 63, 67, 69, 71, 73, 75, 76, 87, 88, 92, 93 , 94 ])
# Features to remove due to REDUNDANCY
#columns_to_drop = np.array([ 2, 3, 5, 8, 10, 17, 18, 23, 27, 38, 42, 43, 61, 66, 77, 88, 90 ])
# Features to remove due to IRRELEVANCE + REDUNDANCY
columns_to_drop = np.array([ 2, 3, 5, 6, 7, 8, 9, 10, 14, 17, 18, 21, 23, 24, 25, 26, 27, 28, 30, 31, 
                            33, 34, 36, 38, 41, 42, 43, 46, 47, 51, 53, 58, 59, 61, 62, 63, 66, 67, 
                            69, 71, 73, 75, 76, 77, 87, 88, 90, 92, 93 , 94 ])

data_subset = drop_columns_at_position(data, columns_to_drop)
save_pd_as_csv('taiwan', data_subset, "data_subset")

# data_no_outliers = remove_outliers(data)
# print(f'[!] Original dataset shape: {data.shape}')
# print(f'[!] Data shape with no outliers: {data_no_outliers.shape}')

normalized_data_zscore = perform_standard_scaling('taiwan', data_subset)
# save_pd_as_csv('taiwan', normalized_data_zscore.describe(), "describe_normalized_zenscore")

normalized_data_minmax = perform_minmax_scaling('taiwan', normalized_data_zscore)
# # # restora a coluna 'Bankrupt?' do dataset original
#normalized_data_minmax['Bankrupt?'] = data['Bankrupt?']
#save_pd_as_csv('taiwan', normalized_data_minmax.describe(), "describe_normalized_minmax")

#generating plot for comparing the normalization algorithms
# fig, axs = subplots(1, 3, figsize=(100,50),squeeze=False)
# axs[0, 0].set_title('Original data')
# data.boxplot(ax=axs[0, 0])
# axs[0, 1].set_title('Z-score normalization')
# normalized_data_zscore.boxplot(ax=axs[0, 1])
# axs[0, 2].set_title('MinMax normalization')
# normalized_data_minmax.boxplot(ax=axs[0, 2])
# save_image('taiwan', 'boxplot_normalized_data_with_no_outliers')

# check_data_balancing('taiwan', normalized_data_zscore, 'Bankrupt?')
# undersampled_data = perform_undersample('taiwan', normalized_data_zscore, 'Bankrupt?')
# oversampled_data = perform_oversample('taiwan', normalized_data_zscore, 'Bankrupt?')
smote_data = perform_smote('taiwan', normalized_data_zscore, 'Bankrupt?')

# equal_width_data = equal_width_descretization('taiwan', data)
# equal_freq_data = equal_frequency_descretization('taiwan', data)

#
# split training and test sets
#

split_train_test_sets('taiwan', smote_data, 'taiwan', 'Bankrupt?')

#
# classification
#

# perform_naive_bayes_analysis('taiwan', 'Bankrupt?')
# perform_knn_analysis('taiwan', 'Bankrupt?')
perform_decision_trees_analysis('taiwan', 'Bankrupt?')
perform_random_forest_analysis('taiwan', 'Bankrupt?')
perform_multi_layer_perceptrons('taiwan', 'Bankrupt?')
perform_gradient_boosting('taiwan', 'Bankrupt?')

print("[!] Done :)")
