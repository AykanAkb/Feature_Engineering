#############################################
# FEATURE ENGINEERING & DATA PRE-PROCESSING FOR DIABET
#############################################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
# !pip install missingno
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name,q1=0.25, q3=0.75):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name,q1, q3)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

def load_diabet():
    data = pd.read_csv("Bootcamp/6.Hafta-Feature Engineering/diabetes.csv")
    return data

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

df=load_diabet()
df.head(10)
df.shape
df.describe().T
df.nunique()
df.info()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

for col in num_cols:
    print(col, check_outlier(df, col))
    print(col, outlier_thresholds(df, col))


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
# Hedef Değişken Analizi (Analysis of Target Variable)
######################################

def target_summary_with_cat(dataframe, target, categorical_col):
    print("#####################################")
    print(target,"\n",pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}))



for col in cat_cols:
    for col2 in num_cols:
        print(target_summary_with_cat(df,col2,col))

######################################
# Korelasyon Analizi (Analysis of Correlation)
######################################

corr = df[num_cols].corr()
corr

sns.set(rc={'figure.figsize': (12, 12)})
sns.heatmap(corr, cmap="RdBu")
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
        sns.heatmap(corr, cmap="RdBu")
        plt.show()
    return drop_list

high_correlated_cols(df, plot=True)
###################
# Feature Engineering
# Eksik Değer Analizi ve Median veya Mean ile Doldurulması
###################

def convert_to_nan(dataframe, excluded_cols, convert_from=0):
    zero_cols = [col for col in dataframe.columns if (dataframe[col].min() == convert_from and col not in excluded_cols)]

    for col in zero_cols:
        dataframe[col] = np.where(dataframe[col] == 0, np.nan, dataframe[col])

    return dataframe

 # applymap Denemeye çalıştım patladı
 #for col in zero_cols:
 #   df[col].applymap(lambda x: np.nan if x == O else x, axis=0)


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

# Hamilelik ve bağımlı değişken dışındaki parametrelerin 0 değeri içermesi imkansız dolayısıyla bu değerleri değiştiricez
exc_cols=["Pregnancies", "Outcome"]
convert_to_nan(df,exc_cols,convert_from=0)
df = missing_filler(df, num_method="median", target="Outcome")

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

grab_outliers(df, "Pregnancies")
grab_outliers(df, "Glucose")
grab_outliers(df, "BloodPressure")
grab_outliers(df, "SkinThickness")
grab_outliers(df, "Insulin")
grab_outliers(df, "BMI")
grab_outliers(df, "DiabetesPedigreeFunction")
grab_outliers(df, "Age")

def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1, q3)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


for col in df.columns:
    print(col, check_outlier(df, col,q1=0.05, q3=0.95))
    if check_outlier(df, col,q1=0.05, q3=0.95):
        replace_with_thresholds(df, col,q1=0.05, q3=0.95)

for col in df.columns:
    print(col, check_outlier(df, col,q1=0.05, q3=0.95))

###################
# Feature Extraction
###################

df["NEW_GLUCOSE_GROUP"] = pd.cut(x=df["Glucose"], bins=[0, 140, 200, 300], labels=["Normal", "Prediabetes", "Diabetes"]).astype('object')

df.loc[(df["Age"]) <= 35.0, 'NEW_AGE_GROUP'] = 'young'
df.loc[(df["Age"] > 35.0) & (df["Age"] <= 55.0), 'NEW_AGE_GROUP'] = 'mature'
df.loc[(df["Age"] > 55.0), 'NEW_AGE_GROUP'] = 'old'


df.loc[(df["BMI"] <= 18.4), 'NEW_BODY_MASS'] = 'Underweight'
df.loc[(df["BMI"] > 18.4) & (df["BMI"] <= 24.9), 'NEW_BODY_MASS'] = 'Healthy'
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), 'NEW_BODY_MASS'] = 'Overweight'
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 44.9), 'NEW_BODY_MASS'] = 'Obese'
df.loc[(df["BMI"] > 44.9), 'NEW_BODY_MASS'] = 'Critical_Obese'

# Normal insulin değerleri.
def set_insulin(dataframe):
    if (dataframe["Insulin"] >= 100) and (dataframe["Insulin"] <= 126):
        return "Normal"
    else:
        return "Abnormal"
df["NEW_INSULIN_SCORE"] = df.apply(set_insulin, axis=1)

###################
# ENCODING
###################

#Tekrar Kategorik - Numerik - Kardinal olarak ayrıldı.
cat_cols, num_cols, cat_but_car = grab_col_names(df)

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

# String içeren kategorik değişkenleri seçelim
df.head()
one_hot_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
# one-hot encode ettik bu değerleri.
def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, one_hot_cols, drop_first=True)

# tekrar
cat_cols, num_cols, cat_but_car = grab_col_names(df)

###################
# Standartization
###################
num_cols

scaler = StandardScaler()

df[num_cols] = scaler.fit_transform(df[num_cols])

df[num_cols].head()

###################
# Model
###################
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score,roc_auc_score
from sklearn.ensemble import RandomForestClassifier
y = df["Outcome"]
X = df.drop("Outcome", axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=17)

rf_model = RandomForestClassifier(random_state=46).fit(X_train, y_train)
y_pred = rf_model.predict(X_test)

print(f"Accuracy: {round(accuracy_score(y_pred, y_test), 2)}")
print(f"Recall: {round(recall_score(y_pred,y_test),3)}")
print(f"Precision: {round(precision_score(y_pred,y_test), 2)}")
print(f"F1: {round(f1_score(y_pred,y_test), 2)}")
print(f"Auc: {round(roc_auc_score(y_pred,y_test), 2)}")

###################
# Feature Importance
###################

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    print(feature_imp.sort_values("Value",ascending=False))
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(rf_model, X)