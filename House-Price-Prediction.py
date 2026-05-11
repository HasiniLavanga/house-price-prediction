# =====================================================
# HOUSE PRICE PREDICTION
# Exploratory Data Analysis & Data Understanding
# =====================================================
# =====================================================
# IMPORT LIBRARIES
# =====================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
# =====================================================
# DISPLAY SETTINGS
# =====================================================
pd.set_option('display.max_columns', None)
sns.set_style("whitegrid")
print("✅ Libraries Imported Successfully")
# =====================================================
# LOAD DATASET
# =====================================================
# Make sure AmesHousing.csv is in the SAME folder
# as this Python file
df = pd.read_csv("AmesHousing.csv")
print("\n✅ Ames Housing Dataset Loaded Successfully")
# =====================================================
# DISPLAY DATASET
# =====================================================
print("\n================ FIRST 5 ROWS ================")
print(df.head())
print("\n📌 Dataset Shape:", df.shape)
print("\n📌 Number of Rows:", df.shape[0])
print("📌 Number of Columns:", df.shape[1])
# =====================================================
# DISPLAY COLUMN NAMES
# =====================================================
print("\n================ COLUMN NAMES ================")
print(df.columns.tolist())
# =====================================================
# DATASET INFORMATION
# =====================================================
print("\n================ DATASET INFORMATION ================")
print(df.info())
# =====================================================
# DATA TYPES
# =====================================================
print("\n================ DATA TYPES ================")
print(df.dtypes)
# =====================================================
# MISSING VALUE ANALYSIS
# =====================================================
print("\n================ MISSING VALUE ANALYSIS ================")
missing_values = df.isnull().sum()
missing_values = missing_values[missing_values > 0]
missing_values = missing_values.sort_values(ascending=False)
print(missing_values)
# =====================================================
# MISSING VALUE PERCENTAGE
# =====================================================
missing_percentage = (
    (df.isnull().sum() / len(df)) * 100
)
missing_percentage = missing_percentage[
    missing_percentage > 0
]
missing_percentage = missing_percentage.sort_values(
    ascending=False
)
print("\n================ MISSING VALUE PERCENTAGE ================")
print(missing_percentage)
# =====================================================
# DUPLICATE VALUE ANALYSIS
# =====================================================
print("\n================ DUPLICATE VALUE ANALYSIS ================")
duplicates = df.duplicated().sum()
print("📌 Duplicate Rows:", duplicates)
# =====================================================
# DESCRIPTIVE STATISTICS
# =====================================================
print("\n================ DESCRIPTIVE STATISTICS ================")
print(df.describe())
# =====================================================
# CHECK TARGET COLUMN
# =====================================================
if "SalePrice" not in df.columns:
    print("\n❌ ERROR: 'SalePrice' column not found.")
    print("Please check your dataset column names.")
else:
    # =================================================
    # TARGET VARIABLE ANALYSIS
    # =================================================
    print("\n================ SALE PRICE DISTRIBUTION ================")
    plt.figure(figsize=(10,5))
    sns.histplot(df['SalePrice'], kde=True)
    plt.title("SalePrice Distribution")
    plt.xlabel("SalePrice")
    plt.ylabel("Frequency")
    plt.show()
    # =================================================
    # LOG TRANSFORMATION
    # =================================================
    df['LogSalePrice'] = np.log1p(df['SalePrice'])
    plt.figure(figsize=(10,5))
    sns.histplot(df['LogSalePrice'], kde=True)
    plt.title("Log Transformed SalePrice Distribution")
    plt.xlabel("Log SalePrice")
    plt.ylabel("Frequency")
    plt.show()
    print("\n✅ Log Transformation Applied Successfully")
    # =================================================
    # CORRELATION ANALYSIS
    # =================================================
    print("\n================ CORRELATION HEATMAP ================")
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr()
    plt.figure(figsize=(18,14))
    sns.heatmap(
        corr_matrix,
        cmap='coolwarm',
        center=0
    )
    plt.title("Correlation Heatmap")
    plt.show()
    # =================================================
    # TOP CORRELATED FEATURES
    # =================================================
    sale_corr = corr_matrix['SalePrice'].sort_values(
        ascending=False
    )
    print("\n================ TOP FEATURES CORRELATED WITH SALEPRICE ================")
    print(sale_corr.head(20))
    # =================================================
    # IMPORTANT FEATURES
    # =================================================
    important_features = [
        'Overall Qual',
        'Gr Liv Area',
        'Garage Cars',
        'Garage Area',
        'Total Bsmt SF',
        'Year Built'
    ]
    # =================================================
    # CHECK FEATURE EXISTENCE
    # =================================================
    valid_features = []
    for feature in important_features:
        if feature in df.columns:
            valid_features.append(feature)
    # =================================================
    # FEATURE RELATIONSHIPS
    # =================================================
    print("\n================ FEATURE RELATIONSHIPS ================")
    for feature in valid_features:
        plt.figure(figsize=(7,5))
        sns.scatterplot(
            x=df[feature],
            y=df['SalePrice']
        )
        plt.title(f"{feature} vs SalePrice")
        plt.xlabel(feature)
        plt.ylabel("SalePrice")
        plt.show()
    # =================================================
    # OUTLIER ANALYSIS
    # =================================================
    print("\n================ OUTLIER ANALYSIS ================")
    for feature in valid_features:
        plt.figure(figsize=(7,4))
        sns.boxplot(x=df[feature])
        plt.title(f"Boxplot - {feature}")
        plt.show()
    # =================================================
    # CATEGORICAL FEATURE ANALYSIS
    # =================================================
    print("\n================ CATEGORICAL FEATURE ANALYSIS ================")
    categorical_columns = df.select_dtypes(
        include='object'
    ).columns
    print("\n📌 Number of Categorical Columns:")
    print(len(categorical_columns))
    print(categorical_columns)
    # =================================================
    # NEIGHBORHOOD DISTRIBUTION
    # =================================================
    if "Neighborhood" in df.columns:
        plt.figure(figsize=(14,6))
        sns.countplot(x=df['Neighborhood'])
        plt.xticks(rotation=90)
        plt.title("Neighborhood Distribution")
        plt.show()
    # =================================================
    # TRAIN TEST SPLIT
    # =================================================
    print("\n================ TRAIN TEST SPLIT ================")
    X = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )
    print("✅ Training Data Shape:", X_train.shape)
    print("✅ Testing Data Shape:", X_test.shape)
    # =================================================
    # SAVE PROCESSED DATASET
    # =================================================
    df.to_csv("processed_housing_data.csv", index=False)
    print("\n✅ Processed Dataset Saved Successfully")
    # =================================================
    # FINAL MESSAGE
    # =================================================
    print("\n🎯 Exploratory Data Analysis Completed Successfully")

# DAY-2
# =====================================================
# FEATURE ENGINEERING & DATA PREPROCESSING
# =====================================================
print("\n=================================================")
print("FEATURE ENGINEERING & DATA PREPROCESSING")
print("=================================================")
# =====================================================
# CREATE NEW FEATURES
# =====================================================
print("\n✅ Creating New Features")
# Total Square Footage
df['TotalSF'] = df['Gr Liv Area'] + df['Total Bsmt SF']
# Total Bathrooms
df['TotalBathrooms'] = (
    df['Full Bath'] +
    (0.5 * df['Half Bath']) +
    df['Bsmt Full Bath'] +
    (0.5 * df['Bsmt Half Bath'])
)
# House Age
df['HouseAge'] = df['Yr Sold'] - df['Year Built']
# Remodel Age
df['RemodelAge'] = df['Yr Sold'] - df['Year Remod/Add']
# Total Porch Area
df['TotalPorchSF'] = (
    df['Open Porch SF'] +
    df['Enclosed Porch'] +
    df['3Ssn Porch'] +
    df['Screen Porch']
)
# Garage Ratio
df['GarageRatio'] = (
    df['Garage Area'] / (df['Gr Liv Area'] + 1)
)
print("✅ New Features Created Successfully")
# =====================================================
# POLYNOMIAL FEATURES
# =====================================================
print("\n✅ Creating Polynomial Features")
df['OverallQual_Sq'] = df['Overall Qual'] ** 2
df['GrLivArea_Sq'] = df['Gr Liv Area'] ** 2
print("✅ Polynomial Features Added")
# =====================================================
# CHECK SKEWNESS
# =====================================================
print("\n================ SKEWNESS ANALYSIS ================")
numeric_features = df.select_dtypes(include=[np.number])
skewness = numeric_features.skew().sort_values(
    ascending=False
)
print(skewness.head(15))
# =====================================================
# HANDLE SKEWED FEATURES
# =====================================================
print("\n✅ Applying Log Transformation on Skewed Features")
skewed_features = skewness[skewness > 0.75].index
for feature in skewed_features:
    if feature != 'SalePrice':
        df[feature] = np.log1p(df[feature])
print("✅ Skewed Features Transformed")
# =====================================================
# ORDINAL ENCODING
# =====================================================
print("\n================ ORDINAL ENCODING ================")
quality_mapping = {
    'Ex': 5,
    'Gd': 4,
    'TA': 3,
    'Fa': 2,
    'Po': 1,
    np.nan: 0
}
ordinal_columns = [
    'Exter Qual',
    'Exter Cond',
    'Kitchen Qual',
    'Heating QC'
]
for col in ordinal_columns:
    if col in df.columns:
        df[col] = df[col].map(quality_mapping)
print("✅ Ordinal Encoding Applied")
# =====================================================
# MISSING VALUE HANDLING
# =====================================================
print("\n================ MISSING VALUE HANDLING ================")
# Numerical Columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())
# Categorical Columns
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    df[col] = df[col].fillna("None")
print("✅ Missing Values Handled Successfully")
# =====================================================
# OUTLIER DETECTION
# =====================================================
print("\n================ OUTLIER DETECTION ================")
for feature in [
    'Gr Liv Area',
    'Total Bsmt SF',
    'Garage Area'
]:
    if feature in df.columns:
        mean = df[feature].mean()
        std = df[feature].std()
        outliers = df[
            (df[feature] > mean + 5 * std)
        ]
        print(f"{feature} Outliers:", len(outliers))
# =====================================================
# FEATURE DISTRIBUTION VISUALIZATION
# =====================================================
print("\n================ FEATURE DISTRIBUTIONS ================")
feature_list = [
    'TotalSF',
    'TotalBathrooms',
    'HouseAge',
    'OverallQual_Sq'
]
for feature in feature_list:
    if feature in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df[feature], kde=True)
        plt.title(f"{feature} Distribution")
        plt.xlabel(feature)
        plt.ylabel("Frequency")
        plt.show()
# =====================================================
# FEATURE VS SALEPRICE
# =====================================================
print("\n================ FEATURE RELATIONSHIPS ================")
for feature in feature_list:
    if feature in df.columns:
        plt.figure(figsize=(7,5))
        sns.scatterplot(
            x=df[feature],
            y=df['SalePrice']
        )
        plt.title(f"{feature} vs SalePrice")
        plt.xlabel(feature)
        plt.ylabel("SalePrice")
        plt.show()
# =====================================================
# UPDATED CORRELATION ANALYSIS
# =====================================================
print("\n================ UPDATED CORRELATION HEATMAP ================")
updated_numeric_df = df.select_dtypes(include=[np.number])
updated_corr = updated_numeric_df.corr()
plt.figure(figsize=(18,14))
sns.heatmap(
    updated_corr,
    cmap='coolwarm',
    center=0
)
plt.title("Updated Correlation Heatmap")
plt.show()
# =====================================================
# TOP FEATURES AFTER ENGINEERING
# =====================================================
updated_sale_corr = updated_corr['SalePrice'].sort_values(
    ascending=False
)
print("\n================ UPDATED TOP FEATURES ================")
print(updated_sale_corr.head(20))
# =====================================================
# SAVE UPDATED DATASET
# =====================================================
df.to_csv("feature_engineered_housing_data.csv", index=False)
print("\n✅ Feature Engineered Dataset Saved Successfully")
# =====================================================
# FINAL MESSAGE
# =====================================================
print("\n🎯 Feature Engineering & Data Preprocessing Completed Successfully")
