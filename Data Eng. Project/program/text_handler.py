import re
import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download('wordnet')
from nltk.corpus import stopwords as sw
from nltk.tokenize import word_tokenize as wt
from nltk.stem import WordNetLemmatizer as wnl


class HandleTexts:
    
    def __init__(self):
        self.stop = set(sw.words("english"))
        self.lemm = wnl()
    
    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text) #remove punctuation
        text = re.sub(r'\d+', '', text) #remove numbers
        
        stop_words = set(sw.words('english'))
        words = text.split()
        text = ' '.join(word for word in words if word not in stop_words)
        
        lemm = wnl() 
        text = [lemm.lemmatize(word) for word in text.split()] #kök haline getir
        return text
    
    def clean_column(self, df, column):
        df[column] = df[column].apply(self.clean_text)
        return df