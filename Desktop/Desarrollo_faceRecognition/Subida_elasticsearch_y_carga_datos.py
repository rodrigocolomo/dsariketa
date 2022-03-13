# %%
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from datetime import datetime
from elasticsearch import Elasticsearch
from PIL import Image
from io import BytesIO

# %% [markdown]
# ### Crear cliente elasticsearch

# %%
es = Elasticsearch([{"scheme": "http", 'host': '192.168.56.103', 'port': 9200}])

# %% [markdown]
# ### Lectura del dataset

# %%
df=pd.read_csv('nombres.csv')
df.head()

# %% [markdown]
# ### Codificar imagen en BASE64

# %%
import os
import glob
# traverse whole directory
for root, dirs, files in os.walk(r'C:\Users\colom\Desktop\uni\Data Science\RETO 7\TRABAJO DS Y BIG DATA\fotos_caras'):
	# select file name
	for file in files:
		# check the extension of files
		if file.endswith('.jpeg'):
			# print whole path of files
			print(os.path.join(root, file))

# %%
files = pd.DataFrame(files)
files = files.astype('string')
files=files.reset_index()
files.columns= ['index', 'name']
files=files['name']
files

# %%
# como no puedo carcar todas a la vez voy a probar solo con una foto (siguiente celda)
lista=[]
for i in files:
    with open('fotos_caras/'+i, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    encoded_string=encoded_string.decode("utf-8")
    lista.append(encoded_string)
#print(encoded_string)

# %%
len(lista)

# %%
bien_puesta=files[files.str.contains('1')==True].index

# %%
(files.index[0] in bien_puesta)

# %%
tipo=[]

bien_puesta=files[files.str.contains('1')==True].index
un_poco_bajada=files[files.str.contains('2')==True].index
muy_bajada=files[files.str.contains('3')==True].index


for i in files.index:
    if (i in bien_puesta)==True:
        tipo.append('Bien puesta')
    elif  (i in un_poco_bajada)==True:
        tipo.append('Un poco bajada')
    elif  (i in muy_bajada)==True:
        tipo.append('Muy bajada')
    else:
        tipo.append('Sin mascarilla')
    


# %%
files

# %%
tipo

# %%
#with open("fotos_caras/Ro_1.jpeg", "rb") as image_file:
    #encoded_string = base64.b64encode(image_file.read())
#encoded_string=encoded_string.decode("utf-8")
#print(encoded_string)                                   # sacamos el código de una foto

# %%
df['tipo_masc']=tipo
df['nombre_foto']= files
df['codigo_foto']=lista
df

# %%
index= list(df.index)
index

# %% [markdown]
# df[]

# %% [markdown]
# ### Indexar documento

# %%
idd = 1
for i in index:

    doc = {
        'Nombre': df.loc[i,'Nombre'],
        'tipo_masc': df.loc[i,'tipo_masc'],
        'nombre_foto': df.loc[i,'nombre_foto'],
        'codigo_foto' : df.loc[i,'codigo_foto'],
    }
   
    res = es.index(index = 'index_datos', id = idd, document = doc)
    #print(res['result'], idd)
    idd += 1

# %%
# sumatorio=1
# for i in lista:
#     doc = {
#         'image' : i
#     }
    
#     res = es.index(index = 'index_fotos_caras', id = sumatorio, document = doc)
#     sumatorio=sumatorio+1

# %% [markdown]
# ### Recuperar documento

# %%
res = es.get(index="index_datos", id=1)

# %% [markdown]
# ### Decodificar imagen recuperada del documento

# %%
img=res['_source']['codigo_foto'].encode()
im = Image.open(BytesIO(base64.b64decode(img)))

# %%
imgplot = plt.imshow(im)
plt.show()

# %%



