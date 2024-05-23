from program.missing_value_handler import HandleMissingValue as hmv
from program.dataType_handler import HandlerDataType as hdt
from program.dateTime_handler import HandleDateAndTime as hdTime
from program.encoding_handler import HandleEncode as he
from program.outlier_handler import HandleOutliers as ho
from program.scaler_handler import HandleScaler as hs
from program.text_handler import HandleTexts as ht
import pandas as pd 
import numpy as np
import re
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

