# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Detect the encoding of the `spam.csv` file and load it using the detected encoding.
2. Check basic data information and identify any null values.
3. Define the features (`X`) and target (`Y`), using `v2` as the feature (message text) and `v1` as the target (spam/ham label).
4. Split the data into training and testing sets (80-20 split).
5. Use `CountVectorizer` to convert the text data in `X` to a matrix of token counts, fitting on the training set and transforming both training and test sets.
6. Initialize and train an SVM classifier on the transformed training data.
7. Predict the target labels for the test set.
8. Calculate and display the model's accuracy.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: THARUN D
RegisterNumber:  212223240167
*/

import chardet
file='spam.csv'
with open(file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd 
data=pd.read_csv("spam.csv",encoding="Windows-1252")
data.head()
data.info()
data.isnull().sum()
x=data["v2"].values
y=data["v1"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
print("Y-Predicted:",y_pred)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
print("Accuracy:",accuracy)
```

## Output:

![Screenshot 2024-11-15 212508](https://github.com/user-attachments/assets/63d2ca64-25d6-4fa2-a32b-49a8688931e3)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
