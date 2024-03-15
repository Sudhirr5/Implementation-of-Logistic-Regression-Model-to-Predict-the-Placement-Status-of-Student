# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm:
1.Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices. 

## Program:
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: SUDHIR KUMAR. R
RegisterNumber:212223230221
```

import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])


```


## Output:
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/299165f7-badf-43a6-9bac-2e28a49ba09c)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/74198cf9-0ada-4ef9-87b9-66860113b6e0)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/3ae44786-f167-441c-8323-19319db2945b)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/7f87834c-f62d-48d2-aa76-e71c76e611c7)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/56aaa71b-5270-4840-ab9a-0b6911c02c88)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/3aa52bfd-c381-4ed0-b2a8-7db738bbfeca)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/110c5f2f-db78-4461-95a5-5cb80b248297)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/62cd1cc9-80d0-4244-9392-33c27553d5c8)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/f74b74a7-a576-4485-8e34-b7279e71216d)
![image](https://github.com/Murali-Krishna0/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/149054535/0930c274-62a9-445d-b8f5-69891f72150f)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
