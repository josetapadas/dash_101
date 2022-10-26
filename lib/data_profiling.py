from lib.config.ds_charts import bar_chart, get_variable_types, HEIGHT
from matplotlib.pyplot import figure, subplots, subplots_adjust, tight_layout, title
from seaborn import heatmap
from lib.utils import *
import numpy as np

def get_numeric_variables(data):
    return get_variable_types(data)['Numeric']

def check_missing_values(data):
    mv = {}
    for var in data:
        nr = data[var].isnull().sum()
        if nr > 0:
            print(f'[!] Found {nr} missing values in {var}')
            mv[var] = nr

    if mv == {}:
        print("[!] No missing values found in dataset.")

def check_record_count(dataset, data):
    figure()
    values = {'# records': data.shape[0], '# variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='# of records vs # variables')
    save_image(dataset, 'records_variables')

def check_dataframe_types(dataset, data):
    cat_vars = data.select_dtypes(include='object')
    data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    save_pd_as_csv(dataset, data.dtypes, 'data_types')

def check_variable_types(dataset, data):
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    save_image(dataset, 'variable_types')


def group_numeric_variables(variables, N = 2):
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    return [variables[n:n+N] for n in range(0, len(variables), N)]

def generate_rows_cols(number_of_variables, N):
    return (number_of_variables // N, N) if number_of_variables % N == 0 else (number_of_variables // N + 1, N)

def get_variable_number(data, key):
    return f'X{list(data.keys()).index(key) + 1}'

def generate_granularity_single(dataset, data):
    print('[+] Generating granularity for each single variables')
    plots_per_line = 3
    grouped_variables = group_numeric_variables(get_variable_types(data)['Numeric'], plots_per_line)

    group_index = 0
    for group in grouped_variables:
        group_index += 1
        number_of_variables = len(group)
        rows, cols = generate_rows_cols(number_of_variables, plots_per_line)
        
        fig, axs = subplots(rows, cols, figsize=(cols*8, rows*6), squeeze=False)
        i, j = 0, 0
        
        print(f'[!] Plotting group {group_index} of {len(grouped_variables)} ...')

        for single_variable in group:
            axs[i, j].set_title(f'Histogram for {get_variable_number(data, single_variable)}')
            axs[i, j].set_xlabel(single_variable)
            axs[i, j].set_ylabel('nr records')
            axs[i, j].hist(data[single_variable].values, bins=100)
            j += 1
        
        save_image(dataset, f'granularity_single_g{group_index}')

def granularity_study_bins(dataset, data):
    print('[+] Generating general granularity histograms for different bins.')
    plots_per_line = 3
    grouped_variables = group_numeric_variables(get_variable_types(data)['Numeric'], plots_per_line)

    group_index = 0

    for group in grouped_variables:
        group_index += 1
        number_of_variables = len(group)
        rows, cols = generate_rows_cols(number_of_variables, plots_per_line)

        bins = (10, 100, 1000)
        cols = len(bins)
        
        fig, axs = subplots(rows, cols, figsize=(cols*8, rows*6), squeeze=False)
        i, j = 0, 0
        
        print(f'[!] Plotting group {group_index} of {len(grouped_variables)} ...')

        for single_variable in group:
            for j in range(cols):
                axs[i, j].set_title(f'Histogram for {get_variable_number(data, single_variable)} and bin {bins[j]}')
                axs[i, j].set_xlabel(single_variable)
                axs[i, j].set_ylabel('Nr records')
                axs[i, j].hist(data[single_variable].values, bins=bins[j])
                j += 1

            save_image(dataset, f'granularity_study_x{get_variable_number(data, single_variable)}')
           
def generate_boxplots(dataset, data):
    print('[+] Generating the bloxplots for the dataset')

    plots_per_line = 3
    grouped_variables = group_numeric_variables(get_variable_types(data)['Numeric'], plots_per_line)
    group_index = 0
    for group in grouped_variables:
        group_index += 1
        number_of_variables = len(group)
        rows, cols = generate_rows_cols(number_of_variables, plots_per_line)
        
        fig, axs = subplots(rows, cols, figsize=(cols*8, rows*6), squeeze=False)
        i, j = 0, 0
        
        print(f'[!] Plotting group {group_index} of {len(grouped_variables)}.')

        for single_variable in group:
            axs[i, j].set_title(f'{get_variable_number(data, single_variable)}')
            axs[i, j].boxplot(data[single_variable].dropna().values)
            j += 1
        
        save_image(dataset, f'boxplot_g{group_index}')

def generate_outliers_plot(dataset, data):
    print('[+] Generating the plots to identify the outliers')

    NR_STDEV: int = 2
    outliers_iqr = []
    outliers_stdev = []
    summary5 = data.describe(include='number')
    plots_per_line = 3
    grouped_variables = group_numeric_variables(get_variable_types(data)['Numeric'], plots_per_line)

    group_index = 0
    for group in grouped_variables:
        group_index += 1
        number_of_variables = len(group)
        rows, cols = generate_rows_cols(number_of_variables, plots_per_line)
        
        fig, axs = subplots(rows, cols, figsize=(cols*8, rows*6), squeeze=False)
        i, j = 0, 0
        width = 0.3
        
        print(f'[!] Plotting group {group_index} of {len(grouped_variables)}.')

        for single_variable in group:
            iqr = 1.5 * (summary5[single_variable]['75%'] - summary5[single_variable]['25%'])
            outliers_iqr = data[data[single_variable] > summary5[single_variable]['75%']  + iqr].count()[single_variable] + data[data[single_variable] < summary5[single_variable]['25%']  - iqr].count()[single_variable]
            std = NR_STDEV * summary5[single_variable]['std']
            outliers_stdev = data[data[single_variable] > summary5[single_variable]['mean'] + std].count()[single_variable] + data[data[single_variable] < summary5[single_variable]['mean'] - std].count()[single_variable]

            axs[i, j].set_title(f'{get_variable_number(data, single_variable)}')
            axs[i, j].bar(1 + width/2, outliers_iqr, width, label='iqr')
            axs[i, j].bar(1 - width/2, outliers_stdev, width, label='stdev')
            axs[i, j].legend()
            j += 1
        
        save_image(dataset, f'outliers_g{group_index}')

def remove_outliers_for_feature(feature, feature_name, dataset):
    # identify the 25th and 75th quartiles

    q25, q75 = np.percentile(feature, 25), np.percentile(feature, 75)
    print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
    feat_iqr = q75 - q25
    print('iqr: {}'.format(feat_iqr))
    # valor do cut-off para retirar os outliers. Originalmente 1.5
    feat_cut_off = feat_iqr * 1.5
    feat_lower, feat_upper = q25 - feat_cut_off, q75 + feat_cut_off
    print('Cut Off: {}'.format(feat_cut_off))
    print(feature_name +' Lower: {}'.format(feat_lower))
    print(feature_name +' Upper: {}'.format(feat_upper))
    
    outliers = [x for x in feature if x < feat_lower or x > feat_upper]
    print(feature_name + ' outliers: {}'.format(len(outliers)))

    dataset = dataset.drop(dataset[(dataset[feature_name] > feat_upper) | (dataset[feature_name] < feat_lower)].index)
    print('-' * 65)
    
    return dataset

def remove_outliers(data):
    print('[-] Removing outliers...')
    for col in data:
        data_no_outliers = remove_outliers_for_feature(data[col], str(col), data)

    return data_no_outliers


def generate_sparsity_study(dataset, data):
    print('[+] Generating the sparsity plot...')

    numeric_vars = get_variable_types(data)['Numeric']
    rows, cols = len(numeric_vars)-1, len(numeric_vars)-1
    figure()
    fig, axs = subplots(rows, cols, figsize=(cols * HEIGHT/2, rows *HEIGHT/2), squeeze=False)
# using padding to allow space for the names of variables not to cross from a graph to the next        
    fig.tight_layout(pad=3.0)
    for i in range(len(numeric_vars)):
        var1 = numeric_vars[i]
        for j in range(i+1, len(numeric_vars)):
            var2 = numeric_vars[j]
#            axs[i, j-1].set_title("%s x %s"%(var1,var2))
            axs[i, j-1].set_title("x(X%d) x y(X%d)"%((i+2),(j+2)))
            axs[i, j-1].set_xlabel(var1)
            axs[i, j-1].set_ylabel(var2)
            axs[i, j-1].scatter(data[var1], data[var2])
    save_image(dataset, f'sparsity_study')

def generate_textual_correlation_table(dataset, data):
    print('[+] Generating the textual correlation table...')
    corr_mtx = abs(data.corr())
    corr_mtx.style.background_gradient(cmap='coolwarm')
    corr_mtx.to_excel(f'./output/{dataset}/tables/{generate_timestamp()}textual_correlation_table.xlsx', engine='openpyxl', index=False)

def generate_correlation_heatmap(dataset, data):
    corr_mtx = abs(data.corr())
    fig = figure(figsize=[38, 38])
    fig.tight_layout(pad=0)
#    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
    heatmap(abs(corr_mtx), xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, cmap='Blues')
    title('Correlation analysis')
    save_image(dataset, f'{dataset}_rf_ranking')
