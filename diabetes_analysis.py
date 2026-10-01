# %% [markdown]
# # Diabetes Dataset — Exploratory Analysis & Classification Models

# %%
import numpy as np
import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold
import missingno as msno
import warnings
warnings.simplefilter(action="ignore")

# %% [markdown]
# ## Load data

# %%
df = pd.read_csv("diabetes.csv")
print(df.head())
print(df)
print(df.info())
print(df.shape)
print(df.describe())
print(df.Outcome.value_counts())

# %% [markdown]
# ## Age distribution

# %%
df["Age"].hist(edgecolor="black")
plt.title("Distribution of Patient Age")
plt.xlabel("Age (Years)")
plt.ylabel("Frequency")
plt.show()
print("Max Age: " + str(df["Age"].max()) + "Min Age: " + str(df["Age"].min()))

# %% [markdown]
# ## Distribution of all numeric features

# %%
fig, ax = plt.subplots(4, 2, figsize=(16, 16), layout="constrained")

sns.distplot(df.Age, bins=20, ax=ax[0, 0])
ax[0, 0].set_title("Distribution of Age", fontsize=12, fontweight='bold')
ax[0, 0].set_xlabel("Age (Years)", fontsize=12, fontweight='bold', labelpad=5)
ax[0, 0].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.Pregnancies, bins=20, ax=ax[0, 1])
ax[0, 1].set_title("Distribution of Pregnancies", fontsize=12, fontweight='bold')
ax[0, 1].set_xlabel("Number of Pregnancies", fontsize=12, fontweight='bold', labelpad=5)
ax[0, 1].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.Glucose, bins=20, ax=ax[1, 0])
ax[1, 0].set_title("Distribution of Glucose Concentration", fontsize=12, fontweight='bold')
ax[1, 0].set_xlabel("Glucose Concentration (mg/dL)", fontsize=12, fontweight='bold', labelpad=5)
ax[1, 0].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.BloodPressure, bins=20, ax=ax[1, 1])
ax[1, 1].set_title("Distribution of Diastolic Blood Pressure", fontsize=12, fontweight='bold')
ax[1, 1].set_xlabel("Diastolic Blood Pressure (mm Hg)", fontsize=12, fontweight='bold', labelpad=5)
ax[1, 1].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.SkinThickness, bins=20, ax=ax[2, 0])
ax[2, 0].set_title("Distribution of Skin Thickness", fontsize=12, fontweight='bold')
ax[2, 0].set_xlabel("Triceps Skin Fold Thickness (mm)", fontsize=12, fontweight='bold', labelpad=5)
ax[2, 0].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.Insulin, bins=20, ax=ax[2, 1])
ax[2, 1].set_title("Distribution of Insulin", fontsize=12, fontweight='bold')
ax[2, 1].set_xlabel("2-Hour Serum Insulin (mu U/ml)", fontsize=12, fontweight='bold', labelpad=5)
ax[2, 1].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.DiabetesPedigreeFunction, bins=20, ax=ax[3, 0])
ax[3, 0].set_title("Distribution of Diabetes Pedigree Function", fontsize=12, fontweight='bold')
ax[3, 0].set_xlabel("Diabetes Pedigree Function Score", fontsize=12, fontweight='bold', labelpad=5)
ax[3, 0].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

sns.distplot(df.BMI, bins=20, ax=ax[3, 1])
ax[3, 1].set_title("Distribution of BMI", fontsize=12, fontweight='bold')
ax[3, 1].set_xlabel("Body Mass Index (BMI in kg/m²)", fontsize=12, fontweight='bold', labelpad=5)
ax[3, 1].set_ylabel("Density", fontsize=12, fontweight='bold', labelpad=5)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.3, hspace=0.6)
plt.show()

# %% [markdown]
# ## Outcome breakdown by other features

# %%
print(df.groupby("Outcome").agg({"Pregnancies": "mean"}))
print(df.groupby("Outcome").agg({"Age": "mean"}))
print(df.groupby("Outcome").agg({"Age": "max"}))
print(df.groupby("Outcome").agg({"Insulin": "mean"}))
print(df.groupby("Outcome").agg({"Insulin": "max"}))
print(df.groupby("Outcome").agg({"Glucose": "mean"}))
print(df.groupby("Outcome").agg({"Glucose": "max"}))
print(df.groupby("Outcome").agg({"BMI": "mean"}))

# %% [markdown]
# ## Target (Outcome) balance

# %%
f, ax = plt.subplots(1, 2, figsize=(18, 8))

df['Outcome'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f%%', ax=ax[0], shadow=True)
ax[0].set_title('Proportion of Diabetes Outcome')
ax[0].set_ylabel('')
ax[0].set_xlabel('')

sns.countplot(x=df['Outcome'], ax=ax[1])
ax[1].set_title('Count of Diabetes Outcome')
ax[1].set_xlabel('Outcome (0 = No Diabetes, 1 = Diabetes)')
ax[1].set_ylabel('Number of Patients')

plt.show()

# %% [markdown]
# ## Correlation analysis

# %%
print(df.corr())

f, ax = plt.subplots(figsize=[20, 15])
sns.heatmap(df.corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
ax.set_xlabel("Features", fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel("Features", fontsize=14, fontweight='bold', labelpad=10)
plt.show()

# %% [markdown]
# ## Checking for impossible zero values

# %%
print(df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']]) == df[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']].replace(0, np.nan)
print(df.head())
print(df.isnull().sum())

# %% [markdown]
# ## Outlier detection (IQR method)

# %%
for feature in df:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if df[(df[feature] > upper)].any(axis=None):
        print(feature, "yes")
    else:
        print(feature, "no")

sns.boxplot(x=df["Insulin"])
plt.title("Patient Insulin Levels Distribution")
plt.xlabel("2-Hour Serum Insulin (mu U/ml)")
plt.ylabel("Insulin")
plt.show()

# %% [markdown]
# ## Cap Insulin outliers

# %%
Q1 = df.Insulin.quantile(0.25)
Q2 = df.Insulin.quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df.loc[df["Insulin"] > upper, "Insulin"] = int(round(upper))

# %% [markdown]
# ## Feature engineering: BMI category

# %%
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9, "NewBMI"] = NewBMI[5]
print(df.head())

# %% [markdown]
# ## Missing value overview

# %%
msno.bar(df)
plt.title("Non-Missing Value Counts per Column")
plt.xlabel("Columns")
plt.ylabel("Count of Non-Missing Values")
plt.show()

# %% [markdown]
# ## Replace impossible zeros with NaN and impute by outcome-group median

# %%
cols = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols] = df[cols].replace(0, np.nan)
print(df.isnull().sum())


def median_target(var):
    return df.groupby("Outcome")[var].median().reset_index()


columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for i in columns:
    if df[i].isnull().sum() == 0:
        continue
    med = median_target(i)
    df.loc[(df["Outcome"] == 0) & (df[i].isnull()), i] = med.loc[0, i]
    df.loc[(df["Outcome"] == 1) & (df[i].isnull()), i] = med.loc[1, i]

print(df.isnull().sum())

# %% [markdown]
# ## Re-check outliers across all numeric columns

# %%
numeric_columns = df.select_dtypes(include=np.number).columns

for feature in numeric_columns:
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    if (df[feature] > upper).any():
        print(feature, "yes")
    else:
        print(feature, "no")

# %% [markdown]
# ## Feature engineering: BMI, Insulin, Glucose categories

# %%
NewBMI = pd.Series(["Underweight", "Normal", "Overweight", "Obesity 1", "Obesity 2", "Obesity 3"], dtype="category")
df["NewBMI"] = NewBMI
df.loc[df["BMI"] < 18.5, "NewBMI"] = NewBMI[0]
df.loc[(df["BMI"] > 18.5) & (df["BMI"] <= 24.9), "NewBMI"] = NewBMI[1]
df.loc[(df["BMI"] > 24.9) & (df["BMI"] <= 29.9), "NewBMI"] = NewBMI[2]
df.loc[(df["BMI"] > 29.9) & (df["BMI"] <= 34.9), "NewBMI"] = NewBMI[3]
df.loc[(df["BMI"] > 34.9) & (df["BMI"] <= 39.9), "NewBMI"] = NewBMI[4]
df.loc[df["BMI"] > 39.9, "NewBMI"] = NewBMI[5]


def set_insulin(row):
    if row["Insulin"] >= 16 and row["Insulin"] <= 166:
        return "Normal"
    else:
        return "Abnormal"


df = df.assign(NewInsulinScore=df.apply(set_insulin, axis=1))

NewGlucose = pd.Series(["Low", "Normal", "Overweight", "Secret", "High"], dtype="category")
df["NewGlucose"] = NewGlucose
df.loc[df["Glucose"] <= 70, "NewGlucose"] = NewGlucose[0]
df.loc[(df["Glucose"] > 70) & (df["Glucose"] <= 99), "NewGlucose"] = NewGlucose[1]
df.loc[(df["Glucose"] > 99) & (df["Glucose"] <= 126), "NewGlucose"] = NewGlucose[2]
df.loc[df["Glucose"] > 126, "NewGlucose"] = NewGlucose[3]

df = pd.get_dummies(df, columns=["NewBMI", "NewInsulinScore", "NewGlucose"], drop_first=True)
print(df.head())
print(df.columns.tolist())

# %% [markdown]
# ## Prepare features and target

# %%
categorical_columns = [
    col for col in df.columns
    if col.startswith("NewBMI_")
    or col.startswith("NewInsulinScore_")
    or col.startswith("NewGlucose_")
]
categorical_df = df[categorical_columns]
print("Categorical columns:")
print(categorical_df.head())

y = df["Outcome"]
X = df.drop(columns=["Outcome"] + categorical_columns)
cols = X.columns
index = X.index
print("X:")
print(X.head())

# %% [markdown]
# ## Scale numeric features

# %%
from sklearn.preprocessing import RobustScaler

transformer = RobustScaler().fit(X)
X = transformer.transform(X)
X = pd.DataFrame(X, columns=cols, index=index)
X = pd.concat([X, categorical_df], axis=1)
print("Final X shape:", X.shape)
print(X.head())

# %% [markdown]
# ## Baseline model comparison (10-fold cross-validation)

# %%
models = []
models.append(('LR', LogisticRegression(random_state=12345)))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier(random_state=12345)))
models.append(('RF', RandomForestClassifier(random_state=12345)))
models.append(('SVM', SVC(gamma='auto', random_state=12345)))
models.append(('XGB', GradientBoostingClassifier(random_state=12345)))

results = []
names = []

kfold = KFold(n_splits=10, shuffle=True, random_state=12345)

for name, model in models:
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring="accuracy")
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)

fig = plt.figure(figsize=(15, 10))
fig.suptitle("Algorithm Comparison")
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_title("10-Fold Cross-Validation Accuracy by Model")
ax.set_xticklabels(names)
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Accuracy")
plt.show()

# %% [markdown]
# ## Hyperparameter tuning: Logistic Regression

# %%
lr_params = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"], "solver": ["liblinear"]}
lr_model = LogisticRegression(random_state=12345, max_iter=1000)
lr_grid = GridSearchCV(estimator=lr_model, param_grid=lr_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=2)
lr_grid.fit(X, y)
print("Logistic Regression Best Parameters:")
print(lr_grid.best_params_)
print("Logistic Regression Best CV Accuracy:")
print(lr_grid.best_score_)

lr_tuned = LogisticRegression(**lr_grid.best_params_, random_state=12345, max_iter=1000)
lr_tuned.fit(X, y)
lr_cv_score = cross_val_score(lr_tuned, X, y, cv=10, scoring="accuracy")
print("Tuned Logistic Regression Accuracy:")
print(lr_cv_score.mean())
print("Tuned Logistic Regression Std:")
print(lr_cv_score.std())

# %% [markdown]
# ## Hyperparameter tuning: Random Forest

# %%
rf_params = {"n_estimators": [100, 200, 500], "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 10], "max_depth": [3, 5, 8, None]}
rf_model = RandomForestClassifier(random_state=12345)
rf_grid = GridSearchCV(estimator=rf_model, param_grid=rf_params, cv=10, scoring="accuracy", n_jobs=-1, verbose=2)
rf_grid.fit(X, y)
print("Random Forest Best Parameters:")
print(rf_grid.best_params_)
print("Random Forest Best CV Accuracy:")
print(rf_grid.best_score_)

rf_tuned = RandomForestClassifier(**rf_grid.best_params_, random_state=12345)
rf_tuned.fit(X, y)
rf_cv_score = cross_val_score(rf_tuned, X, y, cv=10, scoring="accuracy")
print("Tuned Random Forest Accuracy:")
print(rf_cv_score.mean())
print("Tuned Random Forest Std:")
print(rf_cv_score.std())

# %% [markdown]
# ## Final comparison and feature importance

# %%
tuned_results = pd.DataFrame({
    "Model": ["Tuned Logistic Regression", "Tuned Random Forest"],
    "Accuracy": [lr_cv_score.mean(), rf_cv_score.mean()],
    "Std": [lr_cv_score.std(), rf_cv_score.std()]
})
print("Final Tuned Model Comparison:")
print(tuned_results)

feature_imp = pd.Series(rf_tuned.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_imp.values, y=feature_imp.index)
plt.title("Random Forest Feature Importance")
plt.xlabel("Feature Importance")
plt.ylabel("Features")
plt.show()

tuned_results = pd.DataFrame({
    "Model": ["Tuned Logistic Regression", "Tuned Random Forest"],
    "Accuracy": [lr_cv_score.mean(), rf_cv_score.mean()],
    "Std": [lr_cv_score.std(), rf_cv_score.std()]
})

print("FINAL TUNED MODEL COMPARISON")
print(tuned_results.to_string(index=False))
print("Logistic Regression Accuracy:", round(lr_cv_score.mean(), 4))
print("Random Forest Accuracy:", round(rf_cv_score.mean(), 4))
print("Logistic Regression Std:", round(lr_cv_score.std(), 4))
print("Random Forest Std:", round(rf_cv_score.std(), 4))
