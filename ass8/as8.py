import pandas as pd
df = pd.read_csv('titanic_test.csv')
print('Titanic Dataset is successfully loaded....')
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
choice = 1
while(choice != 10):
    print('--------Menu----------')
    print('1.Display information of dataset')
    print('2.Find Missing values')
    print('3.Fill Missing values')
    print('4.Histogram of 1-variable (Age & Fare)') 
    print('5.Histogram  of 2-variables')
    print('6. Exit')
    choice = int(input('Enter your choice: '))
    if choice==1:
        print('Information of Dataset:\n', df.info)
        print('Shape of Dataset (row x column):', df.shape)
        print('Columns Name:', df.columns)
        print('Total elements in dataset:', df.size)
        print('Datatype of attributes (columns):', df.dtypes)
        print('First 5 rows:\n', df.head().T)
        print('Last 5 rows:\n', df.tail().T) 
        print('Any 5 rows:\n',df.sample(5).T)

    if choice == 2:
        print('Total Number of Null Values in Dataset:', df.isna().sum())

    if choice == 3:
        df['age'].fillna(df['age'].median(), inplace = True)
        print('Null values are: \n',df.isna().sum())

    if choice == 4:
        fig, axes = plt.subplots(1,2)
        fig.suptitle('Histogram  of 1-variables (Age & Fare)')
        sns.histplot(data = df, x = 'age', ax=axes[0]) 
        sns.histplot(data = df, x = 'fare', ax=axes[1])
        plt.show()

    if choice == 5:
        fig, axes = plt.subplots(2,2) 
        fig.suptitle('Histogram of 2-variables')
        sns.histplot(data = df, x = 'age',hue='pclass',multiple='dodge', ax=axes [0,0]) 
        sns.histplot(data = df, x = 'fare', hue='pclass',multiple='dodge', ax=axes [0,1])
        sns.histplot(data = df, x = 'age', hue='sex',multiple='dodge' ,ax=axes [1,0]) 
        sns.histplot(data = df, x = 'fare',  hue='sex',multiple='dodge', ax=axes [1,1])
        plt.show()

    if(choice == 6):
	    break