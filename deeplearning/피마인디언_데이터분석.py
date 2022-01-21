import pandas as pd
df=pd.read_csv("C:/Users/yjs61/Desktop/python/dataset/pima-indians-diabetes.csv",
names=["pregnant","plasma","pressure","thickness","insulin","BMI","pedigree","age","class"])

print(df.head(5))
print(df.info())
print(df.describe())
print(df[['pregnant','class']])
print(df[['pregnant','class']].groupby(['pregnant'],as_index=False).mean().sort_values(by='pregnant',ascending=True))
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),linewidths=0.1,vmax=0.5, cmap=plt.cm.gist_heat,linecolor='white',annot=True)

plt.show()

grid = sns.FacetGrid(df, col ='class')
grid.map(plt.hist,'plasma',bins=10)
plt.show()