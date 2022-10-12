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
data = pd.read_csv('./datasets/snp.csv', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, na_values = missing_values)
save_pd_as_csv('snp', data, 'describe')

#
# 1. Data Prifiling
#

# check_record_count('snp', data)
# check_dataframe_types('snp', data)
# check_variable_types('snp', data)
# check_missing_values(data)
# generate_granularity_single('snp', data)
# granularity_study_bins('snp', data)
# generate_boxplots('snp', data)
# generate_outliers_plot('snp', data)
# generate_sparsity_study('snp', data)
# generate_textual_correlation_table('snp', data)
# generate_correlation_heatmap('snp', data)

#
# Data preparation
#

#
# 1) dataset sem feature engineering
#

new_data_no_feat_eng = pd.DataFrame(data)
new_data_no_feat_eng.drop(new_data_no_feat_eng.iloc[:, 56:], inplace=True, axis=1)

# removendo algumas percentagens
new_data_no_feat_eng = clean_empty_excel_value(new_data_no_feat_eng, 'DPRIME', '#VALUE!')
new_data_no_feat_eng = convert_percentage_for_column(new_data_no_feat_eng, 'DPRIME')

# removendo a data
new_data_no_feat_eng = drop_column_at_position(new_data_no_feat_eng, 2)

# resultado
save_pd_as_csv('snp', new_data_no_feat_eng, "no_feature_engineered")


#data_no_outliers = remove_outliers(data)
#print(f'[!] Original dataset shape: {data.shape}')
#print(f'[!] Data shape with no outliers: {data_no_outliers.shape}')

normalized_data_zscore = perform_standard_scaling('snp', new_data_no_feat_eng)
save_pd_as_csv('snp', normalized_data_zscore.describe(), "describe_normalized_zenscore")

normalized_data_minmax = perform_minmax_scaling('snp', normalized_data_zscore)
# restora a coluna 'UPDOWN_SnP' do dataset original
normalized_data_minmax['UPDOWN_SnP'] = data['UPDOWN_SnP']

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

check_data_balancing('snp', normalized_data_zscore, 'UPDOWN_SnP')
undersampled_data = perform_undersample('snp', normalized_data_zscore, 'UPDOWN_SnP')
oversampled_data = perform_oversample('snp', normalized_data_zscore, 'UPDOWN_SnP')
smote_data = perform_smote('snp', normalized_data_zscore, 'UPDOWN_SnP')

equal_width_data = equal_width_descretization('snp', new_data_no_feat_eng)
equal_freq_data = equal_frequency_descretization('snp', new_data_no_feat_eng)

#
# split training and test sets
#

split_train_test_sets('snp', new_data_no_feat_eng, 'snp', 'UPDOWN_SnP')

#
# classification
#

perform_naive_bayes_analysis('snp', 'UPDOWN_SnP')
perform_knn_analysis('snp', 'UPDOWN_SnP')
#perform_decision_trees_analysis('snp', 'UPDOWN_SnP')
#perform_random_forest_analysis('snp', 'UPDOWN_SnP')
perform_multi_layer_perceptrons('snp', 'UPDOWN_SnP')
perform_gradient_boosting('snp', 'UPDOWN_SnP')

print("[!] Done :)")
