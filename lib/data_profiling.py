from lib.config.ds_charts import bar_chart, get_variable_types, choose_grid, HEIGHT, multiple_line_chart
from matplotlib.pyplot import figure, savefig, subplots, Axes, title
from lib.utils import *

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

def check_record_count(data):
    figure()
    values = {'# records': data.shape[0], '# variables': data.shape[1]}
    bar_chart(list(values.keys()), list(values.values()), title='# of records vs # variables')
    savefig('./output/images/records_variables.png')

def check_dataframe_types(data):
    cat_vars = data.select_dtypes(include='object')
    data[cat_vars.columns] = data.select_dtypes(['object']).apply(lambda x: x.astype('category'))
    save_pd_as_csv(data.dtypes, 'data_types')

def check_variable_types(data):
    variable_types = get_variable_types(data)
    counts = {}
    for tp in variable_types.keys():
        counts[tp] = len(variable_types[tp])
    figure(figsize=(4,2))
    bar_chart(list(counts.keys()), list(counts.values()), title='Nr of variables per type')
    savefig('./output/images/variable_types.png')

def group_numeric_variables(variables, N = 2):
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    return [variables[n:n+N] for n in range(0, len(variables), N)]

def generate_rows_cols(number_of_variables, N):
    return (number_of_variables // N, N) if number_of_variables % N == 0 else (number_of_variables // N + 1, N)

def generate_granularity_single(data):
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
            axs[i, j].set_title(f'Histogram for {single_variable}')
            axs[i, j].set_xlabel(single_variable)
            axs[i, j].set_ylabel('nr records')
            axs[i, j].hist(data[single_variable].values, bins=100)
            j += 1
        
        savefig(f'./output/images/granularity_single_g{group_index}.png')

def granularity_study_bins(data):
    print('[+] Generating general granularity histograms for different bins.')
    variables = get_variable_types(data)['Numeric']
    if [] == variables:
        raise ValueError('There are no numeric variables.')

    number_of_variables = len(variables)

    rows, cols = (number_of_variables // 2, 2) if number_of_variables % 2 == 0 else (number_of_variables // 2 + 1, 2)
    bins = (10, 100, 1000)
    cols = len(bins)
    fig, axs = subplots(rows, cols, figsize=(cols*10, rows*8), squeeze=False)
    for i in range(rows):
        for j in range(cols):
            axs[i, j].set_title('Histogram for %s %d bins'%(variables[i], bins[j]))
            axs[i, j].set_xlabel(variables[i])
            axs[i, j].set_ylabel('Nr records')
            axs[i, j].hist(data[variables[i]].values, bins=bins[j])
    savefig('./output/images/granularity_study.png')
