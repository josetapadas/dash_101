from datetime import datetime
from matplotlib.pyplot import savefig
import pandas as pd

def generate_timestamp():
    now = datetime.now() # current date and time
    return now.strftime("%d%m%Y%H%M%S")

def save_pd_as_csv(dataset, df, table_name, defaultIndex = True):
    filename = f'{generate_timestamp()}__{table_name}'
    df.to_csv(f'./output/{dataset}/tables/{filename}.csv', index = defaultIndex)
    print(f'[+] saving {filename} table as csv...')

def save_image(dataset, image_name):
    filename = f'{generate_timestamp()}__{image_name}'
    savefig(f'./output/{dataset}/images/{filename}.png')
    print(f'[+] saving {filename} image as png...')

# função que dropa uma coluna com base no (X-index)
# exemplo: 
# novo_data_set_sem_x1 = drop_column_at_position(normalized_data_minmax, 1)
# print(novo_data_set_sem_x1.head())
def drop_column_at_position(data, index):
    new_data = data
    new_data.drop(new_data.columns[index - 1], axis = 1, inplace = True)
    return new_data


# função que dropa uma lista de colunsa com base no array de indices (X-index)
# exemplo:
# sem_varios = drop_column_at_position(normalized_data_minmax, np.array([1, 3, 4]))
# print(sem_varios.head())
def drop_columns_at_position(data, array_of_indices):
    new_data = data
    new_data.drop(new_data.columns[array_of_indices], axis = 1, inplace = True)
    return new_data

def convert_percentage_for_column(data, col):
    df = pd.DataFrame(data)
    df[col] = df[col].str.replace(' ','')
    df[col] = df[col].str.replace('%','').astype('float') / 100.0
    pd.to_numeric(df[col], errors='coerce')
    return df

def convert_percentage_for_columns(data, columns):
    df = pd.DataFrame(data)

    for current_column in columns:
        df = convert_percentage_for_column(df, current_column)
    
    return df

def clean_empty_excel_value(data, col, dirty_value):
    df = pd.DataFrame(data)
    df = df[df[col].str.contains(dirty_value)==False]
    return df