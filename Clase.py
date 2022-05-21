#!/usr/bin/env python
# coding: utf-8

#In[0]
#  Copyright (c) 2022.
#  Realizado por: Christian Palacios
#  All rights reserved.
#Flores

#In[1]
from sklearn.datasets import load_iris
import pandas as pd
iris = load_iris()
print(iris)

#In[2]
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image

#In[3]
print(iris.data)
clf = DecisionTreeClassifier().fit(iris.data,iris.target)

#In[4]
dot_data = export_graphviz(clf, out_file=None, filled=True, rounded=True, feature_names=iris.feature_names, 
class_names=['Versicolor', 'Setona', 'Virginica'])
graph = pydotplus.graph_from_dot_data(dot_data)
image = graph.write_png("imagen.png")
Image(filename="imagen.png")