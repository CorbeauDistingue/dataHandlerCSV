import pandas as pd 
import numpy as np

class HandleScaler:
    
    def max_scaled(df, column):
        df[column] = df[column] / df[column].abs().max()
        return df
    
    def min_max_scaled(df, column):
        df[column] = ( df[column] - df[column].min() ) / ( df[column].max() - df[column].min() )
        return df
    
    def z_method_scaled(df, column):
        df[column] = ( df[column] - df[column].mean() ) / df[column].std()
        return df