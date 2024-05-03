def DetectOutlier(df,var): 
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    high, low = Q3+1.5*IQR, Q1-1.5*IQR
    
    print("Highest allowed in variable:", var, high)
    print("lowest allowed in variable:", var, low)
    count = df[(df[var] > high) | (df[var] < low)][var].count()
    print('Total outliers in:',var,':',count)

    df = df[((df[var] >= low) & (df[var] <= high))]
    print('Outliers removed in', var)
    return df

def Display(y_test,y_pred):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print('confusion_matrix\n',cm)
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm,annot=True,linewidths=.3,cmap="Blues")
    plt.show()
    import warnings
    warnings.filterwarnings('always')
    from sklearn.metrics import classification_report
    print(classification_report(y_test, y_pred))

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv('IRIS.csv')
print('Iris dataset is successfully loaded .....')
choice = 1
while(choice != 9):
    print('-------------- Menu --------------------')
    print('1. Display information of dataset')
    print('2. create histogram for each feature of dataset')
    print('3. create boxplot for each feature of dataset')
    print('4. Exit')
    choice = int(input('Enter your choice: '))

    if choice == 1:
        columns = ['sepal_length','sepal_width','petal_length','petal_width']
        groupbycolumnname = ['Variety']
        print('Information about the dataset:', df.info())
        print(df.head().T) 
        print(df.columns)

    if choice == 2:
        fig, axes = plt.subplots(2, 2, figsize=(16, 9))
        sns.histplot(df['sepal_length'], ax = axes[0,0])
        sns.histplot(df['sepal_width'], ax = axes[0,1])
        sns.histplot(df['petal_length'], ax = axes[1,0])
        sns.histplot(df['petal_width'], ax = axes[1,1])
        plt.show()

    if choice == 3:
        Variety = ['sepal_length','sepal_width','petal_length','petal_width']
        species = ['Setosa', 'Versicolor', 'Virginica'] 
        fig, axes = plt.subplots(2,2)
        fig.suptitle('Before removing Outliers')
        sns.boxplot(data = df, x ='sepal_length', ax=axes[0,0])
        sns.boxplot(data = df, x = 'sepal_width', ax=axes[0,1]) 
        sns.boxplot(data = df, x = 'petal_length', ax=axes[1,0])
        sns.boxplot(data = df, x = 'petal_width', ax=axes[1,1])
        plt.show()

        print('Identifying overall outliers in feature variables.....')
        for var in Variety :
            df = DetectOutlier(df, var)
        fig, axes = plt.subplots(2,2)
        fig.suptitle('After removing Outliers')
        sns.boxplot(data = df, x ='sepal_length', ax=axes[0,0])
        sns.boxplot(data = df, x = 'sepal_width', ax=axes[0,1]) 
        sns.boxplot(data = df, x = 'petal_length', ax=axes[1,0])
        sns.boxplot(data = df, x = 'petal_width', ax=axes[1,1])
        fig.tight_layout()
        plt.show()

    if choice == 4:
        break