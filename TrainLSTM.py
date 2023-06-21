import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import random as rn
import os

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
import pickle
import matplotlib.pyplot as plt
import re

# Set random seed values
np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)

# Set environment variables for reproducibility
os.environ['PYTHONHASHSEED'] = '0'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

data = pd.read_csv('dataset_komentar_instagram_cyberbullying.csv')
# Mengambil column yang dibutuhkan
data = data[['Instagram Comment Text','Sentiment']]

data['Instagram Comment Text'] = data['Instagram Comment Text'].apply(lambda x: x.lower())
data['Instagram Comment Text'] = data['Instagram Comment Text'].apply((lambda x: re.sub('[^a-zA-z0-9\s]', '', x)))

negative = data[data['Sentiment'] == 'positive']
positive = data[data['Sentiment'] == 'negative']
print("= = = Dataset = = =")
print("data positive (200 x 2)\t : ", positive.size)
print("data negative (200 x 2)\t : ", negative.size)

for idx, row in data.iterrows():
    row[0] = row[0].replace('rt', ' ')

max_fatures = [1000, 2000, 3000]
# embed_dim = [64,128]
# batch_size = [32,64,128]

mf = 1
ed = 1
bs = 2

# baseline
mf = 1; ed =1; bs = 0

# max_fatures = [1000, 2000, 3000]
# # embed_dim = [64,128]
# # batch_size = [32,64,128]

# mf = 0; ed =0; bs = 0
# mf = 0; ed =0; bs = 1
mf = 1; ed =1; bs = 0
# mf = 2; ed =0; bs = 2
# mf = 1; ed =0; bs = 0
# mf = 1; ed =1; bs = 1
# mf = 1; ed =0; bs = 2
# mf = 2; ed =0; bs = 0
# mf = 2; ed =1; bs = 1
# mf = 2; ed =0; bs = 2

# mf = 1; ed = 1; bs = 2
# best
# mf = 2; ed = 1; bs = 2

tokenizer = Tokenizer(num_words=max_fatures[mf], split=' ')
tokenizer.fit_on_texts(data['Instagram Comment Text'].values)
X = tokenizer.texts_to_sequences(data['Instagram Comment Text'].values)
X = pad_sequences(X)
# Menyimpan file tokenizer
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

embed_dim = [64,128]
lstm_out = 196

model = Sequential()
model.add(Embedding(max_fatures[mf], embed_dim[ed],input_length = X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2,activation='softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer='adam',metrics = ['accuracy'])
print(model.summary())

Y = pd.get_dummies(data['Sentiment']).values
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=42, shuffle=True)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

batch_size = [32,64,128]
history = model.fit(X_train, Y_train, epochs = 10, batch_size=batch_size[bs], verbose = 2)


validation_size = 20

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]
score,acc = model.evaluate(X_test, Y_test, verbose = 2, batch_size = batch_size[bs])
print("score: %.2f" % (score))
print("acc: %.2f" % (acc))

plt.plot(history.history['accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.save("modelInstagram.h5")