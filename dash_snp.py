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

missing_values = ["NA", "n/a", "na", "?", "--", "#VALUE!"]
data = pd.read_csv('./datasets/snp.csv', sep=',', decimal='.', parse_dates=True, infer_datetime_format=True, na_values = missing_values)
save_pd_as_csv('snp', data, 'describe')


#
# 0) Data cleanup
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
new_data_no_feat_eng.drop(['DATE'], axis=1, inplace=True)


# resultado
save_pd_as_csv('snp', new_data_no_feat_eng, "no_feature_engineered")

# prepara a variável a usar abaixo
prepared_dataset = new_data_no_feat_eng

#### end of preparation of dataset (1)

#
# 2) dataset com feature engineering
#

# new_data_with_feat_eng = pd.DataFrame(data)
# percentage_columns = [
#     'DPRIME',
#     'Var.DGS30',
#     'Var.DEXUSEU',
#     'Var. DEXUSUK',
#     'Var.VIX',
#     'Var.Volume S&P500',
#     'PX_LAST CCMP.1',
#     'PX_LAST INDU.1',
#     'px last eurostoxx.2',
#     'px last footsie.1',
#     'price nikkei.1',
#     'price HSI.1',
#     'price energy.1',
#     'volume energy.1',
#     'price information.1',
#     'volume inform.1',
#     'health price.1',
#     'volume health.1',
#     'price cons.disc.1',
#     'volume cons.disc',
#     'price utility',
#     'vol.utility',
#     'finant.price',
#     'vol.finant',
#     'indust.price',
#     'vol.industr',
#     'cons.price',
#     'volume consum',
#     'telec.price',
#     'vol.telec',
#     'mater.price',
#     'vol.mater'
# ]


# new_data_with_feat_eng = convert_percentage_for_columns(new_data_with_feat_eng, percentage_columns)

# # removendo as datas
# new_data_with_feat_eng.drop(['DATE', 'Date changes DPRIME', 'Calendar.days.without.trade'], axis=1, inplace=True)

# # remove rows with any values that are not finite
# new_data_with_feat_eng = new_data_with_feat_eng[np.isfinite(new_data_with_feat_eng).all(1)]

# # resultado
# save_pd_as_csv('snp', new_data_with_feat_eng, "with_feature_engineered")

# # prepara a variável a usar abaixo
# prepared_dataset = new_data_with_feat_eng

#### end of preparation of dataset (2)

#
# 1. Data Profiling
#

check_record_count('snp', prepared_dataset)
check_dataframe_types('snp', prepared_dataset)
check_variable_types('snp', prepared_dataset)
check_missing_values(prepared_dataset)
generate_granularity_single('snp', prepared_dataset)
granularity_study_bins('snp', prepared_dataset)
generate_boxplots('snp', prepared_dataset)
generate_outliers_plot('snp', prepared_dataset)
generate_sparsity_study('snp', prepared_dataset)
generate_textual_correlation_table('snp', prepared_dataset)
generate_correlation_heatmap('snp', prepared_dataset)

#data_no_outliers = remove_outliers(prepared_dataset)
#print(f'[!] Original dataset shape: {prepared_dataset.shape}')
#print(f'[!] Data shape with no outliers: {prepared_dataset.shape}')

normalized_data_zscore = perform_standard_scaling('snp', prepared_dataset)
save_pd_as_csv('snp', normalized_data_zscore.describe(), "describe_normalized_zenscore")

normalized_data_minmax = perform_minmax_scaling('snp', prepared_dataset)

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

equal_width_data = equal_width_descretization('snp', prepared_dataset)
equal_freq_data = equal_frequency_descretization('snp', prepared_dataset)

#
# split training and test sets
#

split_train_test_sets('snp', prepared_dataset, 'snp', 'UPDOWN_SnP')

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
