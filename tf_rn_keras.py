# %%
from tensorflow import keras

# %% [markdown]
# # PROJETO DAS FLORES

# %%
from sklearn import datasets

# %%
iris = datasets.load_iris(return_X_y= True)
x = iris[0]
y = iris[1]

# %%
datasets.load_iris()['feature_names']

# %%
datasets.load_iris()['target_names']

# %%
import matplotlib.pyplot as plt
import seaborn as sns

# %%
sns.scatterplot(x=x[:,2], y=x[:,3], hue=y, palette='tab10')
plt.xlabel('comprimento (cm)', fontsize=16)
plt.ylabel('largura (cm)', fontsize=16)
plt.title('Distribuição das pétalas', fontsize=18)
plt.show()

# %%
sns.scatterplot(x=x[:,0], y=x[:,1], hue=y, palette='tab10')
plt.xlabel('comprimento (cm)', fontsize=16)
plt.ylabel('largura (cm)', fontsize=16)
plt.title('Distribuição das sépalas', fontsize=18)
plt.show()

# %%
# Categorização
y = keras.utils.to_categorical(y)


# %%
# Normalização
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x = scaler.fit_transform(x)


# %%
# Separação do conjunto
from sklearn.model_selection import train_test_split

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.2, stratify=y, random_state=0)

# %%
# Criação do modelo
modelo = keras.Sequential([keras.layers.InputLayer(input_shape=[4,], name='entrada'),
                           keras.layers.Dense(512, activation='relu', name='oculta',
                                              kernel_initializer=keras.initializers.RandomNormal(seed=142)),
                                              
                                              keras.layers.Dense(3, activation='softmax', name='saida')])

# %%
modelo.summary()

# %%
# Compilando o modelo
modelo.compile(loss='categorical_crossentropy',
               optimizer='rmsprop', 
               metrics = ['categorical_accuracy'])

# %%
# Treinamento
historico = modelo.fit(x_treino, y_treino, epochs=100, validation_split=0.3)

# %%
import pandas as pd

pd.DataFrame(historico.history).plot()
plt.grid()
plt.show()

# %%
modelo.evaluate(x_teste, y_teste)

# %%
import numpy as np
input_data = np.array([[0.61, 0.5, 0.69, 0.79]])
modelo.predict(input_data)

# %%



