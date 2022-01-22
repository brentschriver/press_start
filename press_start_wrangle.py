import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.preprocessing
from sklearn.model_selection import train_test_split
from scipy import stats

# modeling methods
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures

import warnings
warnings.filterwarnings("ignore")

# Create a function that will acquire the data and prepare it for Exploratory Data Analysis
def acquire_and_prep():
    games = pd.read_csv('vgsales.csv')
    # Drop missing values.
    games.dropna(inplace=True)
    # Convert 'Year' column into integer datatype.
    games.Year = games.Year.astype(int)
    # Lowercase all columns.
    columns = [col.lower() for col in games.columns]
    games.columns = columns
    # Drop all observations where global sales are less than one million.
    games = games[games.global_sales > 1.0]
    # Create a column that gives the age of the game as opposed to the year it was released.
    games['age'] = 2022 - games.year
    # Create a column that combines all sales outside of North America
    games['combined_sales'] = games.eu_sales + games.jp_sales + games.other_sales
    # Create age_bins for the games.
    games['age_bins'] = pd.cut(games.year, bins = [0, 2002, 2009, 2022], labels = ['old_af','middle_aged','noob'])
    # Create two separate dataframes. One for quantitative values and the other for qualitative values.
    return games

# Create a function that takes in a df and splits it to prepare for EDA
def split_data(df):
    # split test off, 20% of original df size. 
    train_validate, test = train_test_split(df, test_size=.2, 
                                            random_state=123)
    
    # split validate off, 30% of what remains (24% of original df size)
    # thus train will be 56% of original df size. 
    train, validate = train_test_split(train_validate, test_size=.3, 
                                       random_state=123)
    
    return train, validate, test


# Create a function that takes in a dataframe and iterates through the categorical columns and plots a Seaborn barplot.
def qualitative_boxplot(df):
    qualitative_values = df.select_dtypes(include=['object', 'category']).columns
    plt.figure(figsize=(36,56))
    for i, col in enumerate(qualitative_values[1:]):
        plot_number = i + 1
        plt.subplot(4,1,plot_number)
        plt.title(col)
        sns.barplot(x=col, y="na_sales", data=df)
        na_sales_rate = df.na_sales.mean()
        plt.axhline(na_sales_rate, label="North American Sales Rate")
        plt.xticks(rotation=45)
        plt.grid(False)
        plt.tight_layout()



# Create a function that iterates through the categorical features and runs the proper statistical test.
def qualitative_stats_test(df):
    top_pubs = get_top_publishers(df)
    alpha = 0.50
    for pub in top_pubs:
        publisher_mean = df[df.publisher == pub].na_sales
        overall_mean = df.na_sales.mean()
    
        t, p = stats.ttest_1samp(publisher_mean, overall_mean)
    
        print(t, p/2)
        
        if p/2 > alpha:
            print("We fail to reject the null hypotheis.")
        elif t < 0:
            print("We fail to reject null hypothesis.")
        else:
            print(f"We reject the null hypothesis. There is sufficient evidence to move forward with the understanding that {pub}'s average sales are greater than the population average.")

# Create a function that returns a list of the top publishers.
def get_top_publishers(df):
    contenders = []
    for pubs in df.publisher.unique():    
        if (df.publisher == pubs).sum() > 10:
            contenders.append(pubs)
    top_pubs = []
    for publisher in contenders:
        if df[df['publisher'] == publisher].na_sales.mean() > df.na_sales.mean():
            top_pubs.append(publisher)
    return top_pubs