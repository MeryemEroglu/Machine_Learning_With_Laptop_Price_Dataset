####################################
# Laptop Price Feature Engineering
####################################

#Company: Laptop üreticisinin adı (örneğin, Apple, HP).
#Product: Laptop modelinin adı veya serisi (örneğin, MacBook Pro, 250 G6).
#TypeName: Laptop türü veya sınıfı (örneğin, Ultrabook, Notebook).
#Inches: Ekran boyutu (inç cinsinden).
#ScreenResolution: Ekran çözünürlüğü ve ekran panel türü bilgisi.
#CPU_Company: İşlemci üreticisi (örneğin, Intel, AMD).
#CPU_Type: İşlemci modeli veya serisi (örneğin, Core i5, Core i7).
#CPU_Frequency (GHz): İşlemci hızını gösteren frekans değeri (GHz cinsinden).
#RAM (GB): Laptopun sahip olduğu RAM miktarı (GB cinsinden).
#Memory: Depolama türü ve kapasitesi (örneğin, 128GB SSD, 256GB Flash Storage).
#GPU_Company: Grafik işlemcisi üreticisi (örneğin, Intel, AMD, NVIDIA).
#GPU_Type: Grafik işlemcisinin modeli veya türü.
#OpSys: Yüklü işletim sistemi (örneğin, macOS, No OS).
#Weight (kg): Laptopun ağırlığı (kilogram cinsinden).
#Price (Euro): Laptopun fiyatı (Euro cinsinden).


# GÖREV 1: KEŞİFSEL VERİ ANALİZİ
       # Adım 1: Genel resmi inceleyin.
       # Adım 2: Sayısal ve kategorik değişkenleri yakalayın.
       # Adım 3: Sayısal ve kategorik değişkenleri analiz edin.
       # Adım 4: Hedef değişken analizini gerçekleştirin.
                # (Kategorik değişkenlere göre hedef değişkenin ortalaması, sayısal değişkenlere göre hedef değişkenin ortalaması)
       # Adım 5: Aykırı değer analizini gerçekleştirin.
       # Adım 6: Eksik veri analizini gerçekleştirin.
       # Adım 7: Korelasyon analizini gerçekleştirin.

# GÖREV 2: ÖZELLİK MÜHENDİSLİĞİ
       # Adım 1: Eksik ve aykırı değerler için gerekli işlemleri gerçekleştirin.
                # Veri setinde eksik gözlem bulunmuyor, ancak Glucose, Insulin gibi değişkenlerde 0 değerleri eksik gözlem göstergesi olabilir.
                # Örneğin, bir kişinin glikoz veya insülin değeri 0 olamaz.
                # İlgili değişkenlerdeki 0 değerlerini NaN ile değiştirmeyi ve ardından eksik değer işlemlerini uygulamayı düşünebilirsiniz.
       # Adım 2: Yeni değişkenler oluşturun.
       # Adım 3: Kodlama (encoding) işlemlerini gerçekleştirin.
       # Adım 4: Sayısal değişkenleri standartlaştırın.
       # Adım 5: Bir model oluşturun.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
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

file_name = "df_head_table.xlsx"
df.head().to_excel(file_name, index=False)

print(f"Tablo '{file_name}' olarak kaydedildi.")
df.dtypes
# Eksik veri oranlarını kontrol et
df.isnull().sum() / len(df) #eksik verimiz bulunmamaktadır

######################################
# GÖREV 1: KEŞİFSEL VERİ ANALİZİ
######################################
# Adım 1: Genel resmi inceleyin.
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
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)


