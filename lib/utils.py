def save_pd_as_csv(df, table_name, defaultIndex = True):
    df.to_csv(f'./output/tables/{table_name}.csv', index = defaultIndex)
    print(f'[+] saving {table_name} table as csv...')