import pandas as pd
df = pd.read_csv('Employee_Salary.csv')
print(' Employee_Salary Dataset is successfully loaded!!')
df2 = pd.read_csv('IRIS.csv')
print('Iris Dataset is successfully loaded!!')

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

choice = 1
while(choice != 9):
    print('--------------MENU-----------')
    print('1. Display information of employee salary dataset')
    print('2. Display statistical information of numerical columns of employee salary dataset')
    print('3. Groupwise statistical of employee salary dataset')
    print('4. Bar plot for all statistics of employee salary dataset')
    print('5. Display information of Iris dataset')
    print('6. Display statistical information of numerical columns of iris dataset')
    print('7. Groupwise statistical of iris dataset')
    print('8. Bar plot for all statistics of iris dataset')
    print('9. Exit')
    choice = int(input("Enter your choice: "))
    if choice == 1:
        print('Information of dataset:\n', df.info)
        print('Shape of Dataset (row x column): ', df.shape)
        print('Columns Name: ', df.columns)
        print('Total elements in dataset: ', df.size)
        print('Datatypes of attributes (columns): ', df.dtypes)
        print('First 5 rows:\n', df.head().T)
        print('Last 5 rows:\n', df.tail().T)
        print('Any 5 rows:\n', df.sample(5).T)

    if choice == 2:
        print('Statistical information of numerical columns: \n')
        columns = ['Experience_Years','Age','Salary']
        print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
        for column in columns:
            m1,m2,m3 = df[column].min(),df[column].max(),df[column].mean()
            m4,s = df[column].median(),df[column].std() 
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format(column,m1,m2,m3,m4,s))

    if choice == 3:
        print('Groupwise statistical summary..')
        columns = ['Experience_Years','Age','Salary']
        for column in columns:
            print('\n---------------',column,'--------------\n')
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
            m1 = df[column].groupby(df['Gender']).min()
            m2 = df[column].groupby(df['Gender']).max()
            m3 = df[column].groupby(df['Gender']).mean()
            m4 = df[column].groupby(df['Gender']).median()
            s = df[column].groupby(df['Gender']).std()
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Female',m1.iloc[0],m2.iloc[0],m3.iloc[0],m4.iloc[0],s.iloc[0]))
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Male',m1.iloc[1],m2.iloc[1],m3.iloc[1],m4.iloc[1],s.iloc[1]))

    if choice == 4:
        X = ['Min','Max','Mean','Median','STD']
        features = ['Salary','Age','Experience_Years']
        for var in features:
            YFemale = []
            YMale = []

            df1 = df[var].groupby(df['Gender']).min()
            YFemale.append(df1.iloc[0])
            YMale.append(df1.iloc[1])

            df1 = df[var].groupby(df['Gender']).max()
            YFemale.append(df1.iloc[0])
            YMale.append(df1.iloc[1])

            df1 = df[var].groupby(df['Gender']).mean()
            YFemale.append(df1.iloc[0])
            YMale.append(df1.iloc[1])

            df1 = df[var].groupby(df['Gender']).median()
            YFemale.append(df1.iloc[0])
            YMale.append(df1.iloc[1])

            df1 = df[var].groupby(df['Gender']).std()
            YFemale.append(df1.iloc[0])
            YMale.append(df1.iloc[1])

            X_axis = np.arange(len(X))
            plt.bar(X_axis-0.2, YFemale, 0.4, label = 'Female')
            plt.bar(X_axis+0.2, YMale, 0.4, label = 'Male')
            plt.xticks(X_axis,X)
            plt.xlabel('Statistical Information')
            plt.ylabel(var)
            plt.title('Groupwise statistical information of employee salary dataset')
            plt.legend()
            plt.show()

    if choice == 5:
        df2 = pd.read_csv('IRIS.csv')

        print('Information of dataset:\n', df2.info)
        print('Shape of Dataset (row x column): ', df2.shape)
        print('Columns Name: ', df2.columns)
        print('Total elements in dataset: ', df2.size)
        print('Datatypes of attributes (columns): ', df2.dtypes)
        print('First 5 rows:\n', df2.head().T)
        print('Last 5 rows:\n', df2.tail().T)
        print('Any 5 rows:\n', df2.sample(5).T)

    if choice == 6:
        print('Statistical information of numerical columns: \n')
        columns = ['sepal_length','sepal_width','petal_length','petal_width']
        print("{:<25}{:<15}{:<15}{:<25}{:<15}{:<25}".format('Columns','Min','Max','Mean','Median','STD'))
        for column in columns:
            m1,m2,m3 = df2[column].min(),df2[column].max(),df2[column].mean()
            m4,s = df2[column].median(),df2[column].std() 
            print("{:<25}{:<15}{:<15}{:<25}{:<15}{:<25}".format(column,m1,m2,m3,m4,s))

    if choice == 7:
        print('Groupwise statistical summary..')
        columns = ['sepal_length','sepal_width','petal_length','petal_width']
        for column in columns:
            print('\n---------------',column,'--------------\n')
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Columns','Min','Max','Mean','Median','STD'))
            m1 = df2[column].groupby(df2['species']).min()
            m2 = df2[column].groupby(df2['species']).max()
            m3 = df2[column].groupby(df2['species']).mean()
            m4 = df2[column].groupby(df2['species']).median()
            s = df2[column].groupby(df2['species']).std()
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris-setosa',m1.iloc[0],m2.iloc[0],m3.iloc[0],m4.iloc[0],s.iloc[0]))
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris-versicolor',m1.iloc[1],m2.iloc[1],m3.iloc[1],m4.iloc[1],s.iloc[1]))
            print("{:<20}{:<10}{:<10}{:<20}{:<10}{:<20}".format('Iris-virginica',m1.iloc[0],m2.iloc[0],m3.iloc[0],m4.iloc[0],s.iloc[0]))

    if choice == 8:
        X = ['Min','Max','Mean','Median','STD']
        features = ['sepal_length','sepal_width','petal_length','petal_width']
        for var in features:
            YIris_setosa = []
            YIris_versicolor = []
            YIris_virginica = []

            df3 = df2[var].groupby(df2['species']).min()
            print(df3)
            print(df3.iloc[0],df3.iloc[1],df3.iloc[2])
            a = input()

            YIris_setosa.append(df3.iloc[0])
            YIris_versicolor.append(df3.iloc[1])
            YIris_virginica.append(df3.iloc[2])

            df3 = df2[var].groupby(df2['species']).max()
            YIris_setosa.append(df3.iloc[0])
            YIris_versicolor.append(df3.iloc[1])
            YIris_virginica.append(df3.iloc[2])

            df3 = df2[var].groupby(df2['species']).min()
            YIris_setosa.append(df3.iloc[0])
            YIris_versicolor.append(df3.iloc[1])
            YIris_virginica.append(df3.iloc[2])

            df3 = df2[var].groupby(df2['species']).median()
            YIris_setosa.append(df3.iloc[0])
            YIris_versicolor.append(df3.iloc[1])
            YIris_virginica.append(df3.iloc[2])

            df3 = df2[var].groupby(df2['species']).std()
            YIris_setosa.append(df3.iloc[0])
            YIris_versicolor.append(df3.iloc[1])
            YIris_virginica.append(df3.iloc[2])

            X_axis = np.arange(len(X))
            plt.bar(X_axis-0.2,YIris_setosa, 0.3, label = 'Iris-setosa')
            plt.bar(X_axis+0.1,YIris_versicolor, 0.3, label = 'Iris-versicolor')
            plt.bar(X_axis+0.4,YIris_virginica, 0.3, label = 'Iris-virginica')
            plt.xticks(X_axis,X)
            plt.xlabel('Statistical Information')
            plt.ylabel(var)
            plt.title('Groupwise statistical information of employee salary dataset')
            plt.legend()
            plt.show()

    if choice == 9:
        break