# %% [markdown]
# # Detección de objetos con SVM mediante las características HOG

# %% [markdown]
# Un Histograma de Gradientes Orientados (HOG - Histogram Oriented Gradients) es un descriptor de características utilizado en una variedad de aplicaciones de procesamiento de imágenes y visión por computadora con el fin de detectar objetos.  
#   
# En esta sección, demostraremos cómo se pueden usar las funciones de la biblioteca `python-opencv` para detectar objetos en una imagen usando **HOG-SVM**. El siguiente código muestra cómo calcular los descriptores HOG a partir de una imagen y utilizar los descriptores para alimentar un clasificador SVM previamente entrenado.  
#    
# Para el ejemplo, haremos uso de un dataset que contiene los siguientes 6 escenarios:  
#   
# - **Buildings** - 0
# - **Forest**    - 1
# - **Glacier**   - 2
# - **Mountain**  - 3
# - **Sea**       - 4
# - **Street**    - 5  
#   
# El dataset contiene alrededor de 25k imágenes de 150x150. 

# %% [markdown]
# **1. Importamos las librerías necesarias y las imágenes a usar:**

# %%
seed = 0

# %%
from elasticsearch import Elasticsearch
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import cv2

from tqdm import tqdm
import random as rn
from random import shuffle  
from zipfile import ZipFile
from PIL import Image

from skimage import feature, color, data
from sklearn.preprocessing import LabelEncoder
from io import BytesIO
import base64

# %%
es = Elasticsearch([{"scheme": "http", 'host': '192.168.56.103', 'port': 9200}])

# %%
#Extraemos nuestros datos de train y test
x = []
y = []


for foto in range(1,24):
    res = es.get(index = 'index_datos', id = foto)
    img = res['_source']['codigo_foto'].encode()
    im = Image.open(BytesIO(base64.b64decode(img)))
    x.append(im)
    persona = res['_source']['Nombre']
    y.append(persona)

# %%
(x)

# %%
for i in x:
    imgplot = plt.imshow(i)
    plt.show()

# %%
len(x)

# %%
len(y)

# %%
from mtcnn.mtcnn import MTCNN

# %%
# extract a single face from a given photograph
def extract_face(image, required_size=(160, 160)):
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = np.asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # deal with negative pixel index
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    return face_array


# %%
# # load the photo and extract the face
# pixels = extract_face('../input/5-celebrity-faces-dataset/data/train/ben_afflek/httpcsvkmeuaeccjpg.jpg')
# plt.imshow(pixels)
# plt.show()
# print(pixels.shape)

# %%
X = []
contador = 0
indice = []
for foto in x:
    contador += 1
    try:
        X.append(extract_face(foto))
    except:
        indice.append(contador -1)
print(indice)


# %%
indice

# %%
len(X)

# %%
len(y)

# %%
#extract_face(x[0])

# %%
y.pop(1)
len(y)

# %%
y

# %%
for i in y:
    print(i)

# %%
for i in X:
    imgplot = plt.imshow(i)
    plt.show()

# %%
X

# %%
y.pop(0)
len(y)

# %%
X.pop(0)
len(X)

# %%
from sklearn.model_selection import train_test_split

xtr,xte,ytr,yte=train_test_split(X,y,train_size=0.7, stratify= y)

# %%
def pasar_array(foto):
    fotoo=[]
    for i in foto:
        fotoo.append(np.array(i))
    
    fotoo=np.array(fotoo)
    fotoo = fotoo.reshape(len(fotoo),-1)
    fotoo = fotoo.flatten().reshape(len(fotoo),-1)
    return(fotoo)

# %%
xtr=pasar_array(xtr)

# %%
xtest=xte
xte=pasar_array(xte)

# %% [markdown]
# **2. Creamos un modelo lineal SVM y lo entrenamos:**

# %%
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.svm import LinearSVC

# %%
xtr

# %%
lsvc = LinearSVC(random_state=0,tol=1e-5)
lsvc.fit(xtr,ytr)

# %%
from sklearn.model_selection import GridSearchCV
lsvc = GridSearchCV(LinearSVC(random_state = 0),{'max_iter':[2000,10000,9000],'C': [1.5, 2.0, 3.0]}, cv = KFold(n_splits = 3, random_state = 0, shuffle = True), verbose = 3)
lsvc.fit(xtr, ytr)

# %%
lsvc.best_score_

# %%
lsvc = lsvc.best_estimator_
lsvc.fit(xtr, ytr)

# %% [markdown]
# Aplicamos una validación cruzada:

# %%
import warnings # filter all the warnings
warnings.filterwarnings('ignore')

# 10-fold cross validation
lsvc_score = lsvc.score(xte,yte)
print('Score', lsvc_score)
kfold = KFold(n_splits=4, random_state=0, shuffle = True)
cv_results = cross_val_score(lsvc , xtr, ytr, cv=kfold, scoring="accuracy")
print(cv_results)

# %% [markdown]
# **3. Finalmente, predecimos la clasificación de escenarios de algunas imágenes aleatorias:**

# %%
predicciones = lsvc.predict(xte)
predicciones

# %%
yte

# %%
fig, ax = plt.subplots(1, 7, figsize=(25, 15),
                       subplot_kw=dict(xticks=[], yticks=[]))

for i, foto in enumerate(xtest):
    pred=predicciones[i]
    ax[i].imshow(foto)
    ax[i].set_title(str(pred))

# %% [markdown]
# A continuación nos podemos scar una foto directamente (mediante la cámara del ordenador), para posteriormente poder predecir el nombre de quien se la ha hecho.

# %%
#pip install opencv-python

# %%
#Codigo para sacar fotos
import cv2

# abrir la cámara
cap = cv2.VideoCapture(0)

# establecer dimensiones
cap.set(cv2.CAP_PROP_FRAME_WIDTH,2560) # ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1440) # alto

# Tomar una imagen
ret, frame = cap.read()
# Guardamos la imagen en un archivo
cv2.imwrite('C:/Users/colom/Desktop/uni/Data Science/RETO 7/TRABAJO DS Y BIG DATA/foto_modelo/rostro.jpg',frame)
#Liberamos la cámara
cap.release()

cv2.imshow('Imagen',frame)

# %%
import matplotlib.pyplot as plt
imgplot=plt.imshow(frame)
plt.show()

# %%
image = Image.fromarray(frame)

# %%
image = image.convert("RGB")

# %%
a = extract_face(image)

# %%
a

# %%
imgplot = plt.imshow(a)
plt.show()

# %%
def pasar_array2(foto):
    fotoo=[]
    for i in foto:
        fotoo.append(np.array(i))
    
    fotoo=np.array(fotoo)
    fotoo = fotoo.reshape(1,-1)
    fotoo = fotoo.flatten().reshape(len(fotoo),-1)
    return(fotoo)

# %%
xtest=xte
xte=pasar_array2(a)

# %%
len(xte)

# %%
predicciones = lsvc.predict(xte)
predicciones

# %% [markdown]
# Se guardarán las diferentes versiones de los modelos de
# reconocimiento junto a sus hiperparamétricos

# %%
indice2 = pd.DataFrame()
indice2['max_iter'] = ['2000', '10000', '9000','2000', '10000', '9000','2000', '10000', '9000']
indice2['c']= ['1.5', '1.5', '1.5', '2.0', '2.0','2.0', '3.0', '3.0','3.0']
indice2['score'] = ['0.85', '0.85','0.85','0.85','0.85','0.85','0.85','0.85','0.85']
indice2['modelo'] = ['LinearSVC', 'LinearSVC','LinearSVC','LinearSVC','LinearSVC','LinearSVC','LinearSVC','LinearSVC','LinearSVC']
indice2

# %%
index=list(indice2.index)

# %%
idd = 1
for i in index:

    doc = {
        'max_iter': indice2.loc[i, 'max_iter'],
        'c': indice2.loc[i, 'c'],
        'score': indice2.loc[i, 'score'],
        'Tipo_modelo': indice2.loc[i, 'modelo']
    }
   
    res = es.index(index = 'index_parametros', id = idd, document = doc)
    #print(res['result'], idd)
    idd += 1

# %%
res = es.get(index="index_parametros", id=4)
res['_source']

# %% [markdown]
# Vamos a crear un indice con las predicciones realizadas haciendo referencia a la
# versión del modelo empleado

# %%
modelo = ['LinearSVC(C=1.5, max_iter=2000, random_state=0)'] * 7

# %%
indice3 = pd.DataFrame()
indice3['prediccion'] = list(predicciones)
indice3['real']= list(yte)
indice3['modelo'] = modelo
indice3

# %%
index=list(indice3.index)

# %%
idd = 1
for i in index:

    doc = {
        'Valor_predicho': indice3.loc[i, 'prediccion'],
        'Valor_real': indice3.loc[i, 'real'],
        'Tipo_modelo': indice3.loc[i, 'modelo']
    }
   
    res = es.index(index = 'index_predicciones', id = idd, document = doc)
    #print(res['result'], idd)
    idd += 1

# %%
res = es.get(index="index_predicciones", id=4)
res['_source']

# %%



