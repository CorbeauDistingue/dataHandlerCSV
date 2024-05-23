import pandas as pd 

class HandleMissingValue:
    
    def fill_as_mean(df, column):
        df[column].fillna(df[column].mean(), inplace=True)
        return df
    
    def fill_as_median(df, column):
        df[column].fillna(df[column].median(), inplace=True)
        return df
    
    def fill_as_constant(df, column, constantVal):
        df[column].fillna(constantVal, inplace=True)
        return df
    
    def drop(df, column):
        df.dropna(subset=[column], inplace=True)
        return df
