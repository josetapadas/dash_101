from sklearn.preprocessing import StandardScaler, MinMaxScaler, KBinsDiscretizer
from pandas import DataFrame, concat, Series
from lib.config.ds_charts import bar_chart, get_variable_types, multiple_bar_chart
from lib.utils import save_image, save_pd_as_csv
from matplotlib.pyplot import figure
from imblearn.over_sampling import SMOTE

def perform_standard_scaling(dataset, data):
    print('[+] Performing standard scaling to dataset...')
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    transf = StandardScaler(with_mean=True, with_std=True, copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_zscore = concat([tmp, df_sb,  df_bool], axis=1)
    save_pd_as_csv(dataset, norm_data_zscore, 'scaled_zscore', False)

    return norm_data_zscore

def perform_minmax_scaling(dataset, data):
    print('[+] Performing MinMax scaling to dataset')

    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    transf = MinMaxScaler(feature_range=(0, 100), copy=True).fit(df_nr)
    tmp = DataFrame(transf.transform(df_nr), index=data.index, columns= numeric_vars)
    norm_data_minmax = concat([tmp, df_sb,  df_bool], axis=1)
    save_pd_as_csv(dataset, norm_data_minmax, 'scaled_minimax', False)
    return norm_data_minmax

def check_data_balancing(dataset, original, class_var):
    print(f'[+] Checking data balancing based on the "{class_var}"...')
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()
    print('\n\tMinority class =', positive_class, ':', target_count[positive_class])
    print('\tMajority class =', negative_class, ':', target_count[negative_class])
    print('\tProportion:', round(target_count[positive_class] / target_count[negative_class], 2), ': 1\n')
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    figure()
    bar_chart(target_count.index, target_count.values, title='Class balance')
    save_image(dataset, 'check_data_balancing')

def perform_undersample(dataset, original, class_var):
    print(f'[+] Undersample the data set based on the "{class_var}"...')
    
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    df_neg_sample = DataFrame(df_negatives.sample(len(df_positives)))
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    df_under = concat([df_positives, df_neg_sample], axis=0)
    save_pd_as_csv(dataset, df_under, 'undersampled_data', False)

    values['UnderSample'] = [len(df_positives), len(df_neg_sample)]
    print('\n\tMinority class =', positive_class, ':', len(df_positives))
    print('\tMajority class =', negative_class, ':', len(df_neg_sample))
    print('\tProportion: ', round(len(df_positives) / len(df_neg_sample), 2), ': 1\n')

    return df_under

def perform_oversample(dataset, original, class_var):
    print(f'[+] Oversample the data set based on the "{class_var}"...')
    
    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    df_positives = original[original[class_var] == positive_class]
    df_negatives = original[original[class_var] == negative_class]
    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    df_pos_sample = DataFrame(df_positives.sample(len(df_negatives), replace=True))
    df_over = concat([df_pos_sample, df_negatives], axis=0)
    save_pd_as_csv(dataset, df_over, 'oversampled_data', False)
    
    values['OverSample'] = [len(df_pos_sample), len(df_negatives)]
    print('\n\tMinority class =', positive_class, ':', len(df_pos_sample))
    print('\tMajority class =', negative_class, ':', len(df_negatives))
    print('\tProportion:', round(len(df_pos_sample) / len(df_negatives), 2), ': \n')

    return df_over

def perform_smote(dataset, original, class_var):
    RANDOM_STATE = 42

    print(f'[+] Performing SMOTE the data set based on the "{class_var}"...')

    target_count = original[class_var].value_counts()
    positive_class = target_count.idxmin()
    negative_class = target_count.idxmax()

    values = {'Original': [target_count[positive_class], target_count[negative_class]]}

    smote = SMOTE(sampling_strategy='minority', random_state=RANDOM_STATE)
    y = original.pop(class_var).values
    X = original.values
    smote_X, smote_y = smote.fit_resample(X, y)
    df_smote = concat([DataFrame(smote_X), DataFrame(smote_y)], axis=1)
    df_smote.columns = list(original.columns) + [class_var]
    save_pd_as_csv(dataset, df_smote, 'smote_data', False)

    smote_target_count = Series(smote_y).value_counts()
    values['SMOTE'] = [smote_target_count[positive_class], smote_target_count[negative_class]]
    print('\n\tMinority class=', positive_class, ':', smote_target_count[positive_class])
    print('\tMajority class=', negative_class, ':', smote_target_count[negative_class])
    print('\tProportion:', round(smote_target_count[positive_class] / smote_target_count[negative_class], 2), ': 1\n')

    figure()
    multiple_bar_chart([positive_class, negative_class], values, title='Target', xlabel='frequency', ylabel='Class balance')
    save_image(dataset, 'data_balancing_comparison')

    return df_smote

def equal_width_descretization(dataset, data):
    print('[+] Performing equal width descretizastion')
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    N_BINS = 5
    discretization = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="uniform")
    discretization.fit(df_nr)
    eq_width = DataFrame(discretization.transform(df_nr), index=data.index)

    df = DataFrame(df_sb, index=data.index)
    df = concat([df, df_bool, eq_width], axis=1)
    df.columns = symbolic_vars + boolean_vars + numeric_vars
    save_pd_as_csv(dataset, df, 'equal_width_descretization', True)

    return df

def equal_frequency_descretization(dataset, data):
    print('[+] Performing equal frequency descretizastion')
    variable_types = get_variable_types(data)
    numeric_vars = variable_types['Numeric']
    symbolic_vars = variable_types['Symbolic']
    boolean_vars = variable_types['Binary']

    df_nr = data[numeric_vars]
    df_sb = data[symbolic_vars]
    df_bool = data[boolean_vars]

    N_BINS = 5
    discretization = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal", strategy="quantile")
    discretization.fit(df_nr)
    eq_width = DataFrame(discretization.transform(df_nr), index=data.index)

    df = DataFrame(df_sb, index=data.index)
    df = concat([df, df_bool, eq_width], axis=1)
    df.columns = symbolic_vars + boolean_vars + numeric_vars
    save_pd_as_csv(dataset, df, 'equal_freq_descretization')

    return df