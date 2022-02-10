#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING FOR TELCO
#############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#pip install catboost
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name,q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

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

def load_telco():
    return pd.read_csv("Bootcamp/6.Hafta-Feature Engineering/Telco-Customer-Churn.csv")

df = load_telco()

df.head()
df.shape
df.describe().T
df.nunique()
df.info()

# TotalCharges ve decision variable sayısal bir değişken olmalı
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

df["Churn"] = df["Churn"].apply(lambda x: 1 if x=='Yes' else 0)

cat_cols, num_cols, cat_but_car = grab_col_names(df)

######################################
# Kategorik Değişken Analizi
######################################

def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()


for col in cat_cols:
    cat_summary(df, col)
    print("#####################################")

######################################
# Sayısal Değişken Analizi
######################################

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist(bins=50)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

    print("#####################################")

for col in num_cols:
    num_summary(df, col)

######################################
# Korelasyon Analizi
######################################

corr = df[num_cols].corr()
corr

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, annot=True, cmap="magma")
plt.show()

def high_correlated_cols(dataframe, plot=False, corr_th=0.70):
    corr = dataframe.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(np.bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, annot=True, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)

###################
# Feature Engineering
# Eksik Değer Analizi ve Median veya Mean ile Doldurulması
###################

# Tenure değeri 0 olan değerler ile total charge nan değerler aynı deolayısıyla ya yeni başlayan kullanıcılar ya da hata var.
# Datadan çıkarmalı ve ya editlemeliyiz.
df.sort_values("tenure",ascending=True).head(20)
df.sort_values("TotalCharges",ascending=False).head(20)

df.isnull().sum()

def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])
    print(missing_df, end="\n")
    if na_name:
        return na_columns

missing_values_table(df)

# Değer doldurmayı genelleme ile yapılmamalı çünkü burdaki nan değerler yeni başlayan kullanıcılar olabilir, median yazılamaz.
def missing_filler(data, num_method="median", target="Outcome"):
    variables_with_na = [col for col in data.columns if data[col].isnull().sum() > 0]
    # Eksik değere sahip olan değişkenler listelenir

    temp_target = data[target]

    print("# BEFORE, # NAN")
    print(data[variables_with_na].isnull().sum())
    # Uygulama öncesi değişkenlerin eksik değerlerinin sayısı ve oranı
    print("Ratio")
    print(data[variables_with_na].isnull().sum() / data.shape[0] * 100, "\n\n")
    # num_method mean ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    if num_method == "mean":
        data = data.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)
    # num_method median ise tipi object olmayan değişkenlerin boş değerleri ortalama ile dolduruluyor
    elif num_method == "median":
        data = data.apply(lambda x: x.fillna(x.median()) if x.dtype != "O" else x, axis=0)

    data[target] = temp_target

    print("# AFTER")
    print(" Imputation method is '" + num_method.upper() + "' for numeric variables! \n")
    print(data[variables_with_na].isnull().sum(), "\n\n")

    return data

# df = missing_filler(df, num_method="median", target="Churn")

# Tenure değeri 0 olanları 1 aylıkmış gibi düşünüp na olan totalcharge değerine aylık ortalama ödeme yerleştirildi.
df["tenure"] = df["tenure"].apply(lambda x: 1 if x == 0 else x)
df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"].median())

df.describe().T

######################################
# Aykırı Değerlerin Kendilerine Erişmek ve baskılamak
######################################


def grab_outliers(dataframe, col_name, q1=0.05, q3=0.95, index=False):
    low, up = outlier_thresholds(dataframe, col_name, q1, q3)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index

grab_outliers(df, "tenure")
grab_outliers(df, "MonthlyCharges")
grab_outliers(df, "TotalCharges")

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

for col in num_cols:
    print(col, check_outlier(df, col))
    if check_outlier(df, col):
        replace_with_thresholds(df, col)

# Aykırı değerimiz yok

###################
# Özellik Çıkarımı
###################

df.head()
df.shape
df.describe().T
df.nunique()
df.info()

# MonthlyCharges değerlerini müşteri ekonomisine göre segmente edebiliriz
df.loc[(df["MonthlyCharges"]<=45),"NEW_MONTHLY_SEGMENT"] = "Economic"
df.loc[(df["MonthlyCharges"]>45) & (df["MonthlyCharges"]<=80),"NEW_MONTHLY_SEGMENT"] = "Standard"
df.loc[(df["MonthlyCharges"]>80),"NEW_MONTHLY_SEGMENT"] = "Premium"

# Kontratı taahütlü olan kullanıcıları ayırma
df["NEW_Engaged"] = df["Contract"].apply(lambda x: 0 if x in ["Month-to-month"] else 1)

# Tenure  değişkeninden yıllık kategorik değişken oluşturma
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
df.loc[(df["tenure"]>12) & (df["tenure"]<=24),"NEW_TENURE_YEAR"] = "1-2 Year"
df.loc[(df["tenure"]>24) & (df["tenure"]<=36),"NEW_TENURE_YEAR"] = "2-3 Year"
df.loc[(df["tenure"]>36) & (df["tenure"]<=48),"NEW_TENURE_YEAR"] = "3-4 Year"
df.loc[(df["tenure"]>48) & (df["tenure"]<=60),"NEW_TENURE_YEAR"] = "4-5 Year"
df.loc[(df["tenure"]>60) & (df["tenure"]<=72),"NEW_TENURE_YEAR"] = "5-6 Year"

# Tenure değişkenine ve ekonomik paketine göre ayrım
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Standard") & ((df["NEW_TENURE_YEAR"]=="0-1 Year") | (
        df["NEW_TENURE_YEAR"]=="1-2 Year")),"NEW_TENURE_SEGMENT"] = "NewStandard"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Economic") & ((df["NEW_TENURE_YEAR"]=="0-1 Year") | (
        df["NEW_TENURE_YEAR"]=="1-2 Year")),"NEW_TENURE_SEGMENT"] = "NewEconomic"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Premium") & ((df["NEW_TENURE_YEAR"]=="0-1 Year") | (
        df["NEW_TENURE_YEAR"]=="1-2 Year")),"NEW_TENURE_SEGMENT"] = "NewPremium"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Standard") & ((df["NEW_TENURE_YEAR"]=="2-3 Year") | (
        df["NEW_TENURE_YEAR"]=="3-4 Year")),"NEW_TENURE_SEGMENT"] = "OrganicStandard"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Economic") & ((df["NEW_TENURE_YEAR"]=="2-3 Year") | (
        df["NEW_TENURE_YEAR"]=="3-4 Year")),"NEW_TENURE_SEGMENT"] = "OrganicEconomic"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Premium") & ((df["NEW_TENURE_YEAR"]=="2-3 Year") | (
        df["NEW_TENURE_YEAR"]== "3-4 Year")),"NEW_TENURE_SEGMENT"] = "OrganicPremium"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Standard") & ((df["NEW_TENURE_YEAR"]=="4-5 Year") | (
        df["NEW_TENURE_YEAR"]== "5-6 Year")),"NEW_TENURE_SEGMENT"] = "LoyalStandard"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Economic") & ((df["NEW_TENURE_YEAR"]=="4-5 Year") | (
        df["NEW_TENURE_YEAR"]=="5-6 Year")),"NEW_TENURE_SEGMENT"] = "LoyalEconomic"
df.loc[(df["NEW_MONTHLY_SEGMENT"] == "Premium") & ((df["NEW_TENURE_YEAR"]=="4-5 Year") | (
        df["NEW_TENURE_YEAR"]=="5-6 Year")),"NEW_TENURE_SEGMENT"] = "LoyalPremium"

# Dahil olduğu hizmet sayısı
df['NEW_TotalServices'] = (df[['PhoneService', 'InternetService', 'OnlineSecurity','OnlineBackup',
                               'DeviceProtection','TechSupport','StreamingTV','StreamingMovies']] == 'Yes').sum(axis=1)

# Ödeme talimatı olup olmaması
df["NEW_FLAG_AutoPayment"] = df["PaymentMethod"].apply(
    lambda x: 1 if x in ["Bank transfer (automatic)","Credit card (automatic)"] else 0)

###################
# ENCODING
###################

#Tekrar Kategorik - Numerik - Kardinal olarak ayrıldı.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in ["Churn"]]
# LABEL ENCODING

def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

# sadece 2 farklı string değer olan değişkenler
df.head()
binary_cols = [col for col in df.columns if (df[col].dtype == "O" and df[col].nunique() == 2)]

# Binary yapımı 1-0
for col in binary_cols:
    label_encoder(df, col)


# one-hot encode
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe
df = one_hot_encoder(df, cat_cols, drop_first=True)
# tekrar
cat_cols, num_cols, cat_but_car = grab_col_names(df)

###################
# Standardization
###################
num_cols

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

###################
# Model
###################

y = df["Churn"]
X = df.drop(["Churn","customerID"], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

catboost_model = CatBoostClassifier(verbose=False, random_state=12345).fit(X_train, y_train)
y_pred = catboost_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),2)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

###################
# Feature Importance
###################


def plot_feature_importance(importance,names,model_type):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {'feature_names': feature_names, 'feature_importance': feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=['feature_importance'], ascending=False, inplace=True)

    # Define size of bar plot
    plt.figure(figsize=(15, 10))
    # Plot Searborn bar chart
    sns.barplot(x=fi_df['feature_importance'], y=fi_df['feature_names'])
    # Add chart labels
    plt.title(model_type + ' FEATURE IMPORTANCE')
    plt.xlabel('FEATURE IMPORTANCE')
    plt.ylabel('FEATURE NAMES')
    plt.show()

plot_feature_importance(catboost_model.get_feature_importance(), X.columns, 'CATBOOST')



