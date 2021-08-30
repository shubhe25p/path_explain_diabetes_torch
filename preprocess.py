import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def diabetes_data(dir='diabetes.csv'):
  data = pd.read_csv(dir)
  data = data.replace(to_replace=['Yes', 'No', 'Positive', 'Negative', 'Male', 'Female'], value=[1,0,1,0,1,0])
  features = ['Age','Gender','Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia','Genital thrush','visual blurring','Itching','Irritability','delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']
  y = data['class']
  data = data.drop(['class'], axis=1)
  x_train_un, x_test_un, y_train, y_test = train_test_split(data, y, test_size=0.2, random_state=2, stratify=y)

  sc = StandardScaler()
  x_train = sc.fit_transform(x_train_un)
  x_test  = sc.transform(x_test_un)

  x_train = x_train.astype(np.float32)
  x_test  = x_test.astype(np.float32)
  y_train = np.asarray(y_train)
  y_test = np.asarray(y_test)

  return x_train, y_train, x_test, y_test, features, x_train_un, x_test_un