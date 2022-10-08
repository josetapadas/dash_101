# 
# Dependancies setup
#

from lib.classification import perform_decision_trees_analysis, perform_knn_analysis, perform_naive_bayes_analysis, split_train_test_sets
from lib.config.config import *
from lib.data_preparation import check_data_balancing, equal_frequency_descretization, equal_width_descretization, perform_minmax_scaling, perform_oversample, perform_smote, perform_standard_scaling, perform_undersample
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
#granularity_study_bins(data)
#generate_boxplots(data)
#generate_outliers_plot(data)

#
# Data preparation
#

#data_no_outliers = remove_outliers(data)
#print(f'[!] Original dataset shape: {data.shape}')
#print(f'[!] Data shape with no outliers: {data_no_outliers.shape}')

# normalized_data_zscore = perform_standard_scaling(data)
# save_pd_as_csv(normalized_data_zscore.describe(), "describe_normalized_zenscore")

# normalized_data_minmax = perform_minmax_scaling(normalized_data_zscore)
# # restora a coluna 'Bankrupt?' do dataset original
# normalized_data_minmax['Bankrupt?'] = data['Bankrupt?']

# save_pd_as_csv(normalized_data_minmax.describe(), "describe_normalized_minmax")

# # função que dropa uma coluna com base no (X-index)
# def drop_column_at_position(data, index):
#     new_data = data
#     new_data.drop(new_data.columns[index - 1], axis = 1, inplace = True)
#     return new_data
  
# novo_data_set_sem_x1 = drop_column_at_position(normalized_data_minmax, 1)
# print(novo_data_set_sem_x1.head())

# # função que dropa uma lista de colunsa com base no array de indices (X-index)
# def drop_columns_at_position(data, array_of_indices):
#     new_data = data
#     new_data.drop(new_data.columns[array_of_indices], axis = 1, inplace = True)
#     return new_data
# sem_varios = drop_column_at_position(normalized_data_minmax, np.array([1, 3, 4]))
# print(sem_varios.head())

# generating plot for comparing the normalization algorithms
# fig, axs = subplots(1, 3, figsize=(100,50),squeeze=False)
# axs[0, 0].set_title('Original data')
# data.boxplot(ax=axs[0, 0])
# axs[0, 1].set_title('Z-score normalization')
# normalized_data_zscore.boxplot(ax=axs[0, 1])
# axs[0, 2].set_title('MinMax normalization')
# normalized_data_minmax.boxplot(ax=axs[0, 2])
# savefig('./output/images/boxplot_normalized_data_with_no_outliers.png')

#check_data_balancing(normalized_data_zscore, 'Bankrupt?')
#undersampled_data = perform_undersample(normalized_data_zscore, 'Bankrupt?')
#oversampled_data = perform_oversample(normalized_data_zscore, 'Bankrupt?')
#smote_data = perform_smote(normalized_data_zscore, 'Bankrupt?')

#equal_width_data = equal_width_descretization(data)
#equal_freq_data = equal_frequency_descretization(data)

split_train_test_sets(data, 'taiwan', 'Bankrupt?')
#perform_naive_bayes_analysis('taiwan', 'Bankrupt?')
#perform_knn_analysis('taiwan', 'Bankrupt?')
perform_decision_trees_analysis('taiwan', 'Bankrupt?')

print("[!] Done :)")
