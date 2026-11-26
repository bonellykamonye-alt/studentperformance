import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("StudentPerformanceFactors.csv")
print("the head of the dataset:",df.head(10))
print(df.info())
print("the number of missing values:",df.isnull().sum())
df.drop(columns=["Distance_from_Home","Parental_Education_Level","Teacher_Quality"],inplace=True)

print("shape",df.shape)
print(df.columns.tolist())
print("dupliacted values",df.duplicated().sum())
print(df.isnull().sum())
print(df.describe())

plt.figure(figsize=(10,8))
sns.histplot(data=df,x="Hours_Studied",y="Exam_Score")
plt.title("Hours_studied vs exam_score")
plt.tight_layout()
plt.show()

num_df=df.select_dtypes(
    include="number"
)
print(num_df)
corr_matrix=num_df.corr()
plt.figure(figsize=(8,8))
sns.heatmap(corr_matrix,annot=True,cmap="coolwarm",fmt=".2f",linewidths=0.5)
plt.title("correlation heatmap")
plt.tight_layout()
plt.show()

print(df.columns.tolist())
uniq_fam=df["Family_Income"].unique()
print(uniq_fam)
print(df["Family_Income"].dtypes)
print(df["Family_Income"].value_counts())

peer=df["Peer_Influence"].value_counts()
print(peer)
print(df["Peer_Influence"].info)


