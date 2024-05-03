import pandas as pd
df=pd.read_csv('Placement_Data_Full_Class.csv')
print("Complete dataset!!")
print(df)

print("\n Printing randomly 5 sample")
print(df.sample(5))

print("\n Last 5 rows")
print(df.tail(5))

print("\n First 5 rows")
print(df.head(5))

print("\n Finding missing values")
miss_value=df.isna().sum()
print(miss_value)

print("\n Initial statics in data")
initial_statistics=df.describe()
print(initial_statistics)

print("\n Variable description")
var_desc=df.dtypes
print(var_desc)

print("\n Check data frame dimension")
dimension = df.shape
print(dimension)

print(pd.isna)
print("\n Proper data type conversion")
df['Sno']=df['Sno'].astype(float)
print(df['Sno'])

print("\n Convert categoric variable into numeric")
df=pd.get_dummies(df,columns=['Gender'])
print(df)

print(df.isna())
print(df.isna().sum())

print(df.isnull())
print(df.isnull().sum())

print(df.describe())

df=df.replace(['M','F'],[1,0])
print(df)