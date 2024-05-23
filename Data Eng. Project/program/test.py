import pandas as pd 

from missing_value_handler import HandleMissingValue as hmv
from outlier_handler import HandleOutliers as ho
from scaler_handler import HandleScaler as hs
from text_handler import HandleTexts as ht

def generate_df():
        df = pd.read_csv("synthetic_sample_data.csv")
        return df


df = generate_df()

#Missing Value Handler
print("|||||||||||||||||||| MISSING VALUE HANDLER |||||||||||||||||||||||||||")

print(f"""
Original Database (only drama): 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Drama"]}
-----------------------------------------------------------
        """)

hmv.fill_as_mean(df, "Budget in USD")

print(f"""
        Database After filled with mean : 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Drama"]}
-----------------------------------------------------------
        """)

df = generate_df()
hmv.fill_as_median(df, "Budget in USD")

print(f"""
Database After filled with median : 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Drama"]}
-----------------------------------------------------------
        """)

df = generate_df()
hmv.fill_as_constant(df, "Budget in USD", 123456)

print(f"""
Database After removing rows which contains missing value : 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Drama"]}
-----------------------------------------------------------
        """)

df = generate_df()
hmv.drop(df, "Budget in USD")

print(f"""
Database After removing rows which contains missing value : 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Drama"]}
-----------------------------------------------------------
        """)

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

#Outlier Handler
print("||||||||||||||||||||||| OUTLIER HANDLER ||||||||||||||||||||||||||||||")

df = generate_df().sort_values(['Budget in USD'], ascending=[False])

print(f"""
Original Database (only comedy): 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Comedy"]}
-----------------------------------------------------------
        """)

ho.identify_outliers(df, "Budget in USD")
print(f"""
Identified Outliers: 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Comedy"]}
-----------------------------------------------------------
        """)

ho.correct_outliers(df, "Budget in USD")
print(f"""
Corrected Outliers: 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Comedy"]}
-----------------------------------------------------------
        """)

df = generate_df().sort_values(['Budget in USD'], ascending=[False])

ho.remove_outliers(df, "Budget in USD")
print(f"""
Removed Outliers: 
-----------------------------------------------------------
{df.head(500).loc[df["Genre"] == "Comedy"]}
-----------------------------------------------------------
        """)

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")

#Outlier Handler
print("||||||||||||||||||||||| SCALER HANDLER |||||||||||||||||||||||||||||||")

df = generate_df().sort_values(['Budget in USD'], ascending=[False])
print(f"""
Original Database sorted accordingly to budget (higher to lower) : 
-----------------------------------------------------------
{df.head(50)["Budget in USD"]}
-----------------------------------------------------------
        """)

hs.max_scaled(df, "Budget in USD")
print(f"""
Original Database sorted accordingly to budget (higher to lower) 
Normalized or Scaled accordingly to maximum absolute : 
-----------------------------------------------------------
{df.head(50)["Budget in USD"]}
-----------------------------------------------------------
        """)

df = generate_df().sort_values(['Budget in USD'], ascending=[False])
hs.min_max_scaled(df, "Budget in USD")
print(f"""
Original Database sorted accordingly to budget (higher to lower) 
Normalized or Scaled accordingly to min-max frequency : 
-----------------------------------------------------------
{df.head(50)["Budget in USD"]}
-----------------------------------------------------------
        """)

df = generate_df().sort_values(['Budget in USD'], ascending=[False])
hs.z_method_scaled(df, "Budget in USD")
print(f"""
Original Database sorted accordingly to budget (higher to lower) 
Normalized or Scaled accordingly to Z method : 
-----------------------------------------------------------
{df.head(50)["Budget in USD"]}
-----------------------------------------------------------
        """)

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||| TEXT HANDLER |||||||||||||||||||||||||||||||||")

df = generate_df()
hmv.drop(df, "Budget in USD")

print(f"""
Original Database (summary) : 
-----------------------------------------------------------
{df.head(50)["Summary"]}
-----------------------------------------------------------
        """)

ht = ht()
ht.clean_column(df=df, column="Summary")

print(f"""
Tokenized Version of Database (summary) : 
-----------------------------------------------------------
{df.head(50)["Summary"]}
-----------------------------------------------------------
        """)

print("||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||")
print("||||||||||||||||||||||| DATA TYPE HANDLER ||||||||||||||||||||||||||||")


