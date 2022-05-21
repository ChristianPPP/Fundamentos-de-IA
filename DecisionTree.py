#!/usr/bin/env python
# coding: utf-8

#In[0]
#  Copyright (c) 2022.
#  Realizado por: Christian Palacios
#  All rights reserved.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
from IPython.display import Image
calidad_vino = pd.read_csv('./winequality-red.csv')
columnas_x = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']
calidad_vino_x = calidad_vino[columnas_x]
calidad_vino_y = calidad_vino.quality

#In[1]
x_train, x_test, y_train, y_test = train_test_split(calidad_vino_x, calidad_vino_y, test_size=0.3 ,random_state=1)
classifier = DecisionTreeClassifier(max_depth=5).fit(x_train, y_train)

#In[2]
datos_dot = export_graphviz(classifier, out_file=None, filled=True, rounded=True, feature_names=columnas_x, class_names=['MBC', 'BC', 'MC', 'AC', 'MAC', 'EC'])
grafica = pydotplus.graph_from_dot_data(datos_dot)
imagen= grafica.write_png("Arbol.png")
Image(filename="Arbol.png")
# %%
