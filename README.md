# EDA_EXP_6 WINE QUALITY DATASET

## NAME : YOGESHWARAN A
## REG NUMBER: 212223040249

**Aim**

To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal.

**Algorithm**

1)Import pandas, numpy, seaborn, matplotlib, sklearn libraries.

**Program**
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

sns.set(style="whitegrid")

# 'all'   -> remove rows that are outliers in ANY numeric feature (aggressive)
# 'focused' -> remove rows that are outliers in ANY of ['alcohol','volatile acidity','pH'] (safer)
# Default is 'all' per request, but the script will AUTO-FALLBACK to 'focused' if 'all' removes too many rows.

REMOVE_MODE = 'all'  # set to 'all' or 'focused'

#LOAD DATA
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pd.read_csv(url, sep=';')
print("Loaded rows,cols:", df.shape)

# create binary target (good >=7)
df['good_wine'] = (df['quality'] >= 7).astype(int)


print("\n--- Quick head & shape ---")
display(df.head())
print("Shape:", df.shape)

# Univariate
brief_feats = ['alcohol', 'volatile acidity', 'pH']
print("\n--- Quick stats (mean,median,std) for key features ---")
print(df[brief_feats].agg(['mean','median','std']).round(3))

plt.figure(figsize=(9,3))
for i, col in enumerate(brief_feats, 1):
    plt.subplot(1,3,i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(col)
plt.tight_layout()
plt.show()

# Bivariate
plt.figure(figsize=(8,3))
for i, col in enumerate(brief_feats[:2], 1):
    plt.subplot(1,2,i)
    sns.boxplot(x='quality', y=col, data=df)
    plt.title(f"{col} vs quality")
plt.tight_layout()
plt.show()

# Multivariate
print("\n--- Correlation (selected features + quality) ---")
print(df[brief_feats + ['quality']].corr().round(3))

plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='alcohol', y='volatile acidity', hue='quality', palette='viridis', s=25)
plt.title('alcohol vs volatile acidity (color=quality)')
plt.show()

# FULL OUTLIER DETECTION (IQR) - ALL numeric features
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
# exclude target columns from detection
exclude = ['quality','good_wine']
num_features = [c for c in num_cols if c not in exclude]

print("\nRunning IQR outlier detection on every numeric feature...")
outlier_summary = []

for col in num_features:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    mask = (df[col] < lower) | (df[col] > upper)
    cnt = int(mask.sum())
    pct = cnt / df.shape[0] * 100
    outlier_summary.append({
        'feature': col,
        'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
        'lower': lower, 'upper': upper,
        'outlier_count': cnt,
        'outlier_pct': pct
    })

out_df = pd.DataFrame(outlier_summary).sort_values(by='outlier_count', ascending=False)
pd.set_option('display.float_format', lambda x: f'{x:.3f}')
print("\n--- Outlier summary (feature, count, %): ---")
display(out_df[['feature','outlier_count','outlier_pct']])

# Top features by outlier count
top_k = min(6, len(out_df))
top_features = out_df.head(top_k)['feature'].tolist()
print(f"\nTop {top_k} features with most outliers: {top_features}")

#VISUALIZE OUTLIERS (boxplots for top features)
plt.figure(figsize=(12, 2*len(top_features)))
for i, col in enumerate(top_features, 1):
    plt.subplot(len(top_features), 1, i)
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot — {col} (outliers beyond whiskers)")
plt.tight_layout()
plt.show()

# 5) SHOW EXAMPLE OUTLIER ROWS (up to 5 per top feature)
print("\nExample outlier rows for top features (up to 5 rows each):")
for col in top_features:
    row = out_df.loc[out_df['feature'] == col].iloc[0]
    lower, upper = float(row['lower']), float(row['upper'])
    mask = (df[col] < lower) | (df[col] > upper)
    if mask.sum() > 0:
        print(f"\n--- {col} : {mask.sum()} outliers (showing up to 5) ---")
        display(df[mask].head(5))
    else:
        print(f"\n--- {col} : no outliers by IQR rule ---")


'''Outlier rule used: value < Q1 - 1.5*IQR or > Q3 + 1.5*IQR.
We detected outliers for every numeric feature above. Record counts & % for discussion.
Below we will compare model performance BEFORE vs AFTER removing rows that are outliers
Removal MODE is set at top (REMOVE_MODE). Default: 'all' (aggressive). Script will auto-fallback if needed.'''

# MODEL: BEFORE OUTLIER REMOVAL

selected_features = ['alcohol','volatile acidity','pH']  # features used for modeling comparison
X = df[selected_features]
y = df['good_wine']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)
pred_before = model_before.predict(X_test)

acc_before = accuracy_score(y_test, pred_before)
print("\n--- Model BEFORE outlier removal ---")
print("Rows used:", df.shape[0])
print("Accuracy:", round(acc_before,4))
print("Confusion Matrix:\n", confusion_matrix(y_test, pred_before))
print(classification_report(y_test, pred_before, digits=4))


'''OUTLIER REMOVAL
    If REMOVE_MODE == 'all' remove any row that is an outlier in ANY numeric feature.
    If that removal leaves the dataset unusable for modeling, auto-fallback to 'focused'
    which removes rows that are outliers in alcohol|volatile acidity|pH only.'''

def remove_outliers_any_feature(df_in, features_list):
    df_tmp = df_in.copy()
    remove_mask = np.zeros(len(df_tmp), dtype=bool)
    for col in features_list:
        row = out_df[out_df['feature'] == col]
        if row.empty:
            continue
        lower = float(row['lower']); upper = float(row['upper'])
        remove_mask = remove_mask | ((df_tmp[col] < lower) | (df_tmp[col] > upper))
    return df_tmp.loc[~remove_mask].reset_index(drop=True), int(remove_mask.sum())

# attempt removal according to REMOVE_MODE
if REMOVE_MODE == 'all':
    df_removed, removed_count = remove_outliers_any_feature(df, num_features)
    print(f"\nAttempting aggressive removal on ALL numeric features -> rows removed: {removed_count}")
    # If too many removed or classes broken, fallback
    if df_removed.shape[0] < 100 or df_removed['good_wine'].nunique() < 2:
        print("Warning: aggressive removal removed too many rows or left only one class. Falling back to 'focused' removal.")
        df_removed, removed_count = remove_outliers_any_feature(df, selected_features)
        print(f"Focused removal on {selected_features} -> rows removed: {removed_count}")
        removal_mode_used = 'focused (fallback from all)'
    else:
        removal_mode_used = 'all'
else:
    df_removed, removed_count = remove_outliers_any_feature(df, selected_features)
    print(f"\nFocused removal on {selected_features} -> rows removed: {removed_count}")
    removal_mode_used = 'focused'

print("Rows after removal:", df_removed.shape[0])

#QUICK BEFORE vs AFTER CHECK (compact)
plt.figure(figsize=(9,6))
for i, col in enumerate(selected_features,1):
    plt.subplot(3,2,2*i-1)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"{col} - original")
    plt.subplot(3,2,2*i)
    sns.histplot(df_removed[col], kde=True, bins=20)
    plt.title(f"{col} - after removal")
plt.tight_layout()
plt.show()

means_before = df[selected_features].mean().round(3)
means_after  = df_removed[selected_features].mean().round(3)
print("\nFeature means before vs after (selected features):")
display(pd.DataFrame({'before_mean': means_before, 'after_mean': means_after}))

#MODEL: AFTER OUTLIER REMOVAL
X2 = df_removed[selected_features]
y2 = df_removed['good_wine']

if y2.nunique() < 2 or df_removed.shape[0] < 30:
    print("\nAfter removal dataset is too small or single-class — cannot train reliable model. Stopping after showing removal info.")
else:
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)
    model_after = LogisticRegression(max_iter=1000)
    model_after.fit(X_train2, y_train2)
    pred_after = model_after.predict(X_test2)

    acc_after = accuracy_score(y_test2, pred_after)
    print("\n--- Model AFTER outlier removal ---")
    print("Removal mode used:", removal_mode_used)
    print("Rows used:", df_removed.shape[0])
    print("Accuracy:", round(acc_after,4))
    print("Confusion Matrix:\n", confusion_matrix(y_test2, pred_after))
    print(classification_report(y_test2, pred_after, digits=4))

# COMPARISON SUMMARY
print(f"Accuracy BEFORE: {acc_before:.4f}")
print(f"Accuracy AFTER : {acc_after:.4f}")
if acc_after > acc_before:
    print("-> Accuracy improved after outlier removal.")
elif acc_after < acc_before:
    print("-> Accuracy decreased after outlier removal.")
else:
    print("-> Accuracy unchanged after outlier removal.")
```


**Output**
<img width="1681" height="479" alt="Screenshot 2025-11-23 151951" src="https://github.com/user-attachments/assets/682c5580-0d17-4a2b-98e5-bf66f7e49e6b" />
<img width="883" height="284" alt="download" src="https://github.com/user-attachments/assets/0b89d002-e19c-4c8e-9b6b-186080ecded9" />
<img width="784" height="284" alt="download" src="https://github.com/user-attachments/assets/7706c48c-6b8b-425e-9be9-ccc2c60fa27e" />
<img width="753" height="701" alt="Screenshot 2025-11-23 152134" src="https://github.com/user-attachments/assets/a3d6f6f6-339a-4a45-98c7-c96ba1dc808e" />
<img width="643" height="556" alt="Screenshot 2025-11-23 152220" src="https://github.com/user-attachments/assets/e1e49900-67fc-4e78-bed4-758dd7d96d16" />
<img width="1354" height="27" alt="Screenshot 2025-11-23 152257" src="https://github.com/user-attachments/assets/c53df218-92b3-4bcf-a8e8-8d9626d43d0b" />
<img width="1181" height="1184" alt="download" src="https://github.com/user-attachments/assets/39d9d72d-bfc7-403b-870b-d4bd933b688c" />
<img width="1647" height="344" alt="Screenshot 2025-11-23 152349" src="https://github.com/user-attachments/assets/a52c2825-70d2-45c2-bb28-8927a12cddf5" />
<img width="1633" height="302" alt="Screenshot 2025-11-23 152418" src="https://github.com/user-attachments/assets/ce592718-5d3e-4b13-a57d-611b454e345d" />
<img width="1630" height="301" alt="Screenshot 2025-11-23 152441" src="https://github.com/user-attachments/assets/ed8e3818-9cd0-424d-a689-de966854a9f7" />
<img width="1625" height="305" alt="Screenshot 2025-11-23 152505" src="https://github.com/user-attachments/assets/fa8bd2bd-1bf0-4765-96bf-0da343625da7" />
<img width="1639" height="289" alt="Screenshot 2025-11-23 152531" src="https://github.com/user-attachments/assets/99db0681-4ac5-4060-99ca-7535a149af24" />
<img width="1627" height="298" alt="Screenshot 2025-11-23 152717" src="https://github.com/user-attachments/assets/b2e65660-8fd6-4ce0-9d05-a06337cc8cd4" />
<img width="1627" height="298" alt="Screenshot 2025-11-23 152717" src="https://github.com/user-attachments/assets/88268e5d-3bf5-45c7-b7ca-a8ea506f782e" />
<img width="691" height="327" alt="Screenshot 2025-11-23 152847" src="https://github.com/user-attachments/assets/4b38b075-1bc1-4747-bd05-17257c71612e" />
<img width="1676" height="101" alt="Screenshot 2025-11-23 152955" src="https://github.com/user-attachments/assets/24405649-5e15-4961-aacc-4014e084aaba" />
<img width="883" height="584" alt="download" src="https://github.com/user-attachments/assets/beaa7ae8-08d2-424b-b41f-7212fa02c1c0" />
<img width="595" height="627" alt="Screenshot 2025-11-23 153251" src="https://github.com/user-attachments/assets/810df855-52a4-4c4c-9b73-c46abac201e7" />


**Result**

Thus, To perform complete Exploratory Data Analysis (EDA) on the Wine Quality dataset, detect and remove outliers using the IQR method, and compare the performance of a classification model (Logistic Regression) before and after outlier removal has successfully completed.
