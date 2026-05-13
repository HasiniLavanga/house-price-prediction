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

# =====================================================
# MODEL BUILDING & MACHINE LEARNING PIPELINE
# =====================================================
print("\n=================================================")
print("MODEL BUILDING & MACHINE LEARNING PIPELINE")
print("=================================================")
# =====================================================
# IMPORT MACHINE LEARNING LIBRARIES
# =====================================================
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    cross_val_score,
    KFold,
    train_test_split
)
# =====================================================
# UPDATED TRAIN TEST SPLIT
# =====================================================
print("\n✅ Updating Train-Test Split After Feature Engineering")
X = df.drop(["SalePrice"], axis=1)
y = df["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("✅ Updated Training Shape:", X_train.shape)
print("✅ Updated Testing Shape:", X_test.shape)
# =====================================================
# IDENTIFY COLUMN TYPES
# =====================================================
numeric_features = X.select_dtypes(
    include=[np.number]
).columns
categorical_features = X.select_dtypes(
    include=['object']
).columns
print("\n📌 Numerical Features:", len(numeric_features))
print("📌 Categorical Features:", len(categorical_features))
# =====================================================
# NUMERICAL PIPELINE
# =====================================================
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),

        (
            "scaler",
            StandardScaler()
        )
    ]
)
# =====================================================
# CATEGORICAL PIPELINE
# =====================================================
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),

        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
# =====================================================
# COLUMN TRANSFORMER
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),

        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
print("\n✅ Preprocessing Pipeline Created")
# =====================================================
# DEFINE MODELS
# =====================================================
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.001),
    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        random_state=42
    )
}
# =====================================================
# MODEL TRAINING & EVALUATION
# =====================================================
results = []
print("\n================ MODEL TRAINING ================")
for model_name, model in models.items():
    print(f"\n🚀 Training {model_name}")
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )
    # =================================================
    # TRAIN MODEL
    # =================================================
    pipeline.fit(
        X_train,
        np.log1p(y_train)
    )
    # =================================================
    # PREDICTIONS
    # =================================================
    predictions = pipeline.predict(X_test)
    # =================================================
    # RMSE
    # =================================================
    rmse = np.sqrt(
        mean_squared_error(
            np.log1p(y_test),
            predictions
        )
    )
    # =================================================
    # R2 SCORE
    # =================================================
    r2 = r2_score(
        np.log1p(y_test),
        predictions
    )
    print(f"✅ RMSE: {rmse:.4f}")
    print(f"✅ R2 Score: {r2:.4f}")
    # =================================================
    # CROSS VALIDATION
    # =================================================
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    cv_scores = cross_val_score(
        pipeline,
        X,
        np.log1p(y),
        cv=cv,
        scoring='neg_root_mean_squared_error'
    )
    mean_cv_rmse = -cv_scores.mean()
    print(f"✅ Cross Validation RMSE: {mean_cv_rmse:.4f}")
    # =================================================
    # STORE RESULTS
    # =================================================
    results.append({
        "Model": model_name,
        "RMSE": rmse,
        "R2 Score": r2,
        "CV RMSE": mean_cv_rmse
    })
# =====================================================
# RESULTS DATAFRAME
# =====================================================
results_df = pd.DataFrame(results)
print("\n================ MODEL COMPARISON ================")
print(results_df)
# =====================================================
# MODEL COMPARISON GRAPH
# =====================================================
plt.figure(figsize=(10,6))
sns.barplot(
    data=results_df,
    x="Model",
    y="CV RMSE"
)
plt.title("Model Comparison - Cross Validation RMSE")
plt.xticks(rotation=15)
plt.ylabel("RMSE")
plt.show()
# =====================================================
# BEST MODEL
# =====================================================
best_model = results_df.sort_values(
    by="CV RMSE"
).iloc[0]
print("\n🏆 BEST MODEL")
print(best_model)
# =====================================================
# RANDOM FOREST PIPELINE
# =====================================================
rf_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                random_state=42
            )
        )
    ]
)
rf_pipeline.fit(
    X_train,
    np.log1p(y_train)
)
# =====================================================
# LEARNING CURVE ANALYSIS
# =====================================================
train_sizes = []
train_scores = []
test_scores = []
sizes = np.linspace(
    0.1,
    1.0,
    5
)
for size in sizes:
    subset_size = int(len(X_train) * size)
    X_subset = X_train.iloc[:subset_size]
    y_subset = np.log1p(
        y_train.iloc[:subset_size]
    )
    rf_pipeline.fit(
        X_subset,
        y_subset
    )
    train_pred = rf_pipeline.predict(
        X_subset
    )
    test_pred = rf_pipeline.predict(
        X_test
    )
    train_rmse = np.sqrt(
        mean_squared_error(
            y_subset,
            train_pred
        )
    )
    test_rmse = np.sqrt(
        mean_squared_error(
            np.log1p(y_test),
            test_pred
        )
    )
    train_sizes.append(subset_size)
    train_scores.append(train_rmse)
    test_scores.append(test_rmse)
# =====================================================
# LEARNING CURVE GRAPH
# =====================================================
plt.figure(figsize=(10,6))
plt.plot(
    train_sizes,
    train_scores,
    marker='o',
    label='Training RMSE'
)
plt.plot(
    train_sizes,
    test_scores,
    marker='o',
    label='Testing RMSE'
)
plt.title("Learning Curve Analysis")
plt.xlabel("Training Size")
plt.ylabel("RMSE")
plt.legend()
plt.show()
# =====================================================
# SAVE RESULTS
# =====================================================
results_df.to_csv(
    "model_results.csv",
    index=False
)
print("\n✅ Model Results Saved Successfully")
# =====================================================
# FINAL MESSAGE
# =====================================================
print("\n🎯 Model Building & Evaluation Completed Successfully")
# =====================================================
# DAY 4 - XGBOOST & OPTUNA OPTIMIZATION
# =====================================================
print("\n=================================================")
print("XGBOOST & OPTUNA HYPERPARAMETER OPTIMIZATION")
print("=================================================")
# =====================================================
# IMPORT LIBRARIES
# =====================================================
import optuna
from xgboost import XGBRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler
)
from sklearn.metrics import (
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    KFold
)
# =====================================================
# REMOVE DATA LEAKAGE COLUMNS
# =====================================================
print("\n✅ Removing Leakage Columns")
columns_to_remove = []
if "SalePrice" in df.columns:
    columns_to_remove.append("SalePrice")
if "LogSalePrice" in df.columns:
    columns_to_remove.append("LogSalePrice")
X = df.drop(columns=columns_to_remove)
y = np.log1p(df["SalePrice"])
print("✅ Features Prepared Successfully")
# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("✅ Train Test Split Completed")
# =====================================================
# IDENTIFY COLUMN TYPES
# =====================================================
numeric_features = X.select_dtypes(
    include=[np.number]
).columns
categorical_features = X.select_dtypes(
    include=['object']
).columns
print("\n📌 Numerical Features:", len(numeric_features))
print("📌 Categorical Features:", len(categorical_features))
# =====================================================
# NUMERIC PIPELINE
# =====================================================
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)
# =====================================================
# CATEGORICAL PIPELINE
# =====================================================
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)
# =====================================================
# COLUMN TRANSFORMER
# =====================================================
preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numeric_transformer,
            numeric_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)
print("✅ Preprocessing Pipeline Created")
# =====================================================
# OPTUNA OBJECTIVE FUNCTION
# =====================================================
def objective(trial):
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            50,
            150
        ),
        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            8
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.2
        ),
        "subsample": trial.suggest_float(
            "subsample",
            0.6,
            1.0
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.6,
            1.0
        ),
        "objective": "reg:squarederror",
        "random_state": 42,
        "verbosity": 0
    }
    model = XGBRegressor(**params)
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                model
            )
        ]
    )
    pipeline.fit(
        X_train,
        y_train
    )
    predictions = pipeline.predict(
        X_test
    )
    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )
    return rmse
# =====================================================
# START OPTUNA STUDY
# =====================================================
print("\n🚀 Starting Hyperparameter Optimization")
study = optuna.create_study(
    direction="minimize"
)
# Faster execution
study.optimize(
    objective,
    n_trials=5
)
print("\n✅ Optimization Completed")
# =====================================================
# BEST PARAMETERS
# =====================================================
print("\n================ BEST PARAMETERS ================")
print(study.best_params)
print("\n✅ Best RMSE:", study.best_value)
# =====================================================
# FINAL XGBOOST MODEL
# =====================================================
best_params = study.best_params
best_params["objective"] = "reg:squarederror"
best_params["random_state"] = 42
best_params["verbosity"] = 0
final_model = XGBRegressor(
    **best_params
)
final_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            final_model
        )
    ]
)
# =====================================================
# TRAIN FINAL MODEL
# =====================================================
print("\n🚀 Training Final XGBoost Model")
final_pipeline.fit(
    X_train,
    y_train
)
# =====================================================
# PREDICTIONS
# =====================================================
predictions = final_pipeline.predict(
    X_test
)
# =====================================================
# EVALUATION
# =====================================================
rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)
r2 = r2_score(
    y_test,
    predictions
)
print("\n================ FINAL MODEL PERFORMANCE ================")
print("✅ RMSE:", rmse)
print("✅ R2 Score:", r2)
# =====================================================
# CROSS VALIDATION
# =====================================================
cv = KFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)
cv_scores = cross_val_score(
    final_pipeline,
    X,
    y,
    cv=cv,
    scoring='neg_root_mean_squared_error'
)
cv_rmse = -cv_scores.mean()
print("✅ Cross Validation RMSE:", cv_rmse)
# =====================================================
# FEATURE IMPORTANCE
# =====================================================
print("\n================ FEATURE IMPORTANCE ================")
xgb_model = final_pipeline.named_steps["model"]
feature_importance = xgb_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature Index": range(
        len(feature_importance)
    ),
    "Importance": feature_importance
})
importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)
print(importance_df.head(10))
# =====================================================
# FEATURE IMPORTANCE GRAPH
# =====================================================
plt.figure(figsize=(10,6))
sns.barplot(
    data=importance_df.head(10),
    x="Importance",
    y="Feature Index"
)
plt.title("Top 10 XGBoost Feature Importances")
plt.savefig("xgboost_feature_importance.png")
plt.close()
print("✅ Feature Importance Graph Saved")
# =====================================================
# OPTUNA PARAMETER IMPORTANCE
# =====================================================
optuna_importance = optuna.importance.get_param_importances(
    study
)
importance_names = list(
    optuna_importance.keys()
)
importance_values = list(
    optuna_importance.values()
)
plt.figure(figsize=(10,6))
sns.barplot(
    x=importance_values,
    y=importance_names
)
plt.title("Optuna Hyperparameter Importance")
plt.savefig("optuna_parameter_importance.png")
plt.close()
print("✅ Optuna Importance Graph Saved")
# =====================================================
# SAVE BEST PARAMETERS
# =====================================================
best_params_df = pd.DataFrame(
    [study.best_params]
)
best_params_df.to_csv(
    "best_xgboost_parameters.csv",
    index=False
)
print("✅ Best Parameters Saved")
# =====================================================
# FINAL MESSAGE
# =====================================================
print("\n🎯 XGBoost & Hyperparameter Optimization Completed Successfully")
print("\n✅ Feature Engineered Dataset Saved Successfully")
# =====================================================
# FINAL MESSAGE
# =====================================================
print("\n🎯 Feature Engineering & Data Preprocessing Completed Successfully")
