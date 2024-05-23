import pandas as pd 


class HandleDateAndTime():
    
    def convert_to_datetime(df, column):
        df[column] = pd.to_datetime(df[column], errors='coerce')
        return df
    
    def extract_date_parts(df, column):
        df[f'{column}_year'] = df[column].dt.year
        df[f'{column}_month'] = df[column].dt.month
        df[f'{column}_day'] = df[column].dt.day
        
        return df
