import joblib
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler


from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import os
os.makedirs("Tables/", exist_ok=True)

import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv('laptop_price_dataset.csv')
# Veri setindeki ilk 5 satırı görüntüle
df.head()
# Veri türlerini kontrol et
df.dtypes
# Eksik veri oranlarını kontrol et
df.isnull().sum() / len(df) #eksik verimiz bulunmamaktadır

################################################
# 1. Exploratory Data Analysis
################################################

def check_df(dataframe, head= 5):
    """
    This function provides an overview of a DataFrame including its shape, data types,
    the first 'head' rows, the last 'head' rows, the count of missing values, and selected quantiles.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    head : int, optional
        Number of rows to display from the beginning and end of the DataFrame (default is 5).

    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.99, 1]).T)

check_df(df)

#ANALYSIS OF CATEGORICAL VARIABLES
def cat_summary(dataframe, col_name, plot=True, save_path="Tables/"):
    """
        Display a summary of a categorical variable in a DataFrame, including value counts and ratios.
        Save the plot to a specified directory if plot is True.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the categorical variable.
        col_name : str
            The name of the categorical column to be analyzed.
        plot : bool, optional
            If True, display a countplot to visualize the distribution and save the plot (default is True).
        save_path : str, optional
            The directory where the plot will be saved (default is "Tables/").
        """
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.savefig(f"{save_path}cat_summary_plot.png")
        plt.show(block=True)

cat_summary(df, "Price (Euro)")
#Price (Euro): Bu sütun, Price (Euro) değişkenindeki farklı fiyat değerlerini ifade eder.
#Count (veya tekrar sayısı): Her bir fiyatın kaç kez tekrarlandığını belirtir.
#Ratio (Oran): Her bir fiyatın, toplam gözlem sayısına göre yüzdesini ifade eder.

#ANALYSIS OF NUMERICAL VARIABLES
def num_summary(dataframe, numerical_col, plot=True, save_path="Tables/"):
    """
        Display a summary of a numerical variable in a DataFrame, including descriptive statistics and an optional histogram.
        Save the plot to a specified directory if plot is True.

        Parameters
        ----------
        dataframe : DataFrame
            The DataFrame containing the numerical variable.
        numerical_col : str
            The name of the numerical column to be analyzed.
        plot : bool, optional
            If True, display a histogram to visualize the distribution and save the plot (default is True).
        save_path : str, optional
            The directory where the plot will be saved (default is "Tables/").
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    # Ensure the directory exists
    os.makedirs(save_path, exist_ok=True)

    if plot:
        # Create histogram plot
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)

        # Save the plot to the specified path
        plt.savefig(f"{save_path}num_summary_plot.png")
        plt.close()  # Close the plot after saving it

        # Display the plot
        plt.show(block=True)
num_summary(df, "Price (Euro)")

#ANALYSIS OF NUMERIC VARIABLES BY TARGET
def target_summary_with_num(dataframe, target, numerical_col):
    """
    Display the mean of a numerical variable grouped by the target variable in a DataFrame.

    Parameters
    ----------
    dataframe (DataFrame): The DataFrame containing the data.
    target (str): The name of the target variable.
    numerical_col (str): The name of the numerical column to be analyzed.

    """
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n#################################\n\n")

for col in num_cols:
    target_summary_with_num(df, "Price (Euro)", col)

#ANALYSIS OF CATEGORICAL VARIABLES BY TARGET

def target_summary_with_cat(dataframe, target, categorical_col):
    """
        Calculate the mean of the target variable grouped by the specified categorical column.

        Parameters
        ----------
        dataframe : pandas.DataFrame
            The input DataFrame containing the data.
        target : str
            The name of the target variable for which the mean will be calculated.
        categorical_col : str
            The name of the categorical column used for grouping.

        """
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

#target_summary_with_cat(df, "Price (Euro)", cat_cols)

def correlation_matrix(dataframe, cols):
    """
    Generate and display a correlation matrix heatmap for the specified columns.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        The input DataFrame containing the data.
    cols : list of str
        The list of column names for which the correlation matrix will be calculated and visualized.
    """
    fig = plt.gcf()
    fig.set_size_inches(10, 8)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    fig = sns.heatmap(dataframe[cols].corr(), annot=True, linewidths=0.5, annot_kws={'size': 12}, linecolor='w', cmap='RdBu')
    plt.show(block=True)
correlation_matrix(df, num_cols)

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Provides the names of categorical, numeric, and categorical but cardinal variables in the dataset.
    Note: Numeric-appearing categorical variables are also included in the categorical variables.

    Parameters
    ----------
    dataframe : DataFrame
        The DataFrame to be analyzed.
    cat_th : int, optional
        The threshold value for variables that are numerical but categorical (default is 10).
    car_th : int, optional
        The threshold value for categorical but cardinal variables (default is 20).

    Returns
    -------
    cat_cols : list
        List of categorical variables.
    num_cols : list
        List of numeric variables.
    cat_but_car : list
        List of categorical-appearing cardinal variables.

    Notes
    -----
    cat_cols + num_cols + cat_but_car = total number of variables
    num_but_cat is within cat_cols.
    """
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]


    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

check_df(df)
cat_cols, num_cols, cat_but_car = grab_col_names(df, cat_th=5, car_th=20)
