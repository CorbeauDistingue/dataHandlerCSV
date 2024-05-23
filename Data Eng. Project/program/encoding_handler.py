import pandas as pd 
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class HandleEncode():
    
    @staticmethod
    def one_hot_encode(df, column):
        encoder = OneHotEncoder(sparse_output=False)
        encoded = encoder.fit_transform(df[[column]])
        encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
        return pd.concat([df, encoded_df], axis=1).drop(columns=[column])

    @staticmethod
    def label_encode(df, column):
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        return df
    
    