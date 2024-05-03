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
    print('2. Find Missing values')
    print('3. Detect and remove outliers')
    print('4. Encoding using label encoder')
    print('5. Find correlation matrix')
    print('6. Train and Test the model using Bernoulli Naive Bayes')
    print('7. Train and Test the model using Guassian Naive Bayes')
    print('8. Prediction on user input ')
    print('9. Exit')
    choice = int(input('Enter your choice: '))

    if choice == 1:
        columns = ['sepal_length','sepal_width','petal_length','petal_width']
        groupbycolumnname = ['Variety']
        print('Information about the dataset:', df.info())
        print(df.head().T) 
        print(df.columns)

    if choice == 2:
        print(df.isnull().sum())

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
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score
        import warnings

        warnings.filterwarnings("ignore")
        df = pd.read_csv('IRIS.csv')
        print(df.head().T)
        print("\nColumn Names: \n")
        print(df.columns)
        encode = LabelEncoder()
        df.species = encode.fit_transform(df.species)
        print(df.head(10))

    if choice == 5:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.heatmap(df.corr(), annot=True) 
        plt.show()

    if choice == 6:
        x = df.iloc[:, [0, 1, 2, 3]].values 
        y = df.iloc[:, 4].values
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)
        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.fit_transform(x_test)

        from sklearn.naive_bayes import BernoulliNB
        classifier = BernoulliNB()
        classifier.fit(x_train, y_train)

        y_pred = classifier.predict(x_test)
        print('y_pred=', y_pred)
        from sklearn.metrics import accuracy_score

        print("Accuracy of the BernoulliNB model:")
        print(accuracy_score(y_pred, y_test))
        Display(y_pred, y_test)

    if choice == 7:
        x = df.iloc[:, [0, 1, 2, 3]].values 
        y = df.iloc[:, 4].values

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state=0)

        from sklearn.preprocessing import StandardScaler
        sc_x = StandardScaler()
        x_train = sc_x.fit_transform(x_train)
        x_test = sc_x.fit_transform(x_test)


        from sklearn.naive_bayes import GaussianNB
        model = GaussianNB()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test) 
        print('y_pred=', y_pred)

        from sklearn.metrics import accuracy_score
        print('Accuracy of the GaussianNB model:') 
        print(accuracy_score(y_test, y_pred))
        Display(y_pred, y_test)

    if choice == 8:
        import numpy as np
        from sklearn.svm import SVC

        x = df.iloc[:, [0, 1, 2, 3]].values 
        y=df.iloc[:, 4].values

        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=0)
        model = SVC() 
        svm = SVC()
        model.fit(x_train, y_train) 
        pred = model.predict(x_test)
        svm.fit(x_train, y_train) 
        features = np.array([[4.0, 2.0, 4.1, 0.2]])
        print(np.array)
        prediction = svm.predict(features)
        print('Prediction: ()'.format(prediction))

    if choice == 9:
        break