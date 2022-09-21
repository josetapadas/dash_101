def save_pd_as_csv(df, table_name):
    csv_data = df.to_csv(f'./output/tables/{table_name}.csv', index = True)
    print(f'[+] saving {table_name} table as csv...')