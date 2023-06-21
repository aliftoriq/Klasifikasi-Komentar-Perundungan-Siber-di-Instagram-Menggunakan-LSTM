import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
import pickle
import streamlit as st

# load model
model = load_model("modelInstagram.h5")

# load tokenizer
tokenizer = Tokenizer(num_words=2000, split=' ')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict(txt):
    tempTxt = txt
    # preprocess txt
    txt = txt.lower()
    txt = re.sub('[^a-zA-z0-9\s]', '', txt)
    # vectorize the txt using the pre-fitted tokenizer
    txt = tokenizer.texts_to_sequences([txt])
    # pad the txt to have the same shape as the model input
    txt = pad_sequences(txt, maxlen=114, dtype='int32', value=0)
    # predict the sentiment using the loaded model
    sentiment = model.predict(txt, batch_size=1, verbose=2)[0]
    if np.argmax(sentiment) == 0:
        text = "Komentar Cyberbulying"
        return text
    else:
        text = "Komentar Biasa"
        return text
    
# # txt = 'jangan tolol banget deh jadi orang, sadar diri'
# txt = 'Kamu baik banget deh, boleh ga aku temenan sama kamu'
# # txt = 'sebenrnya kamutuh baik, tapi kenapasih kamu sering nyusahin orang'

# print(predict(txt))

def main():


    st.title("Klasifikasi Komentar Perundungan Siber di Instagram Menggunakan LSTM")
    text = st.text_input('Komentar Instagram')

    sentiment = "Masukan Komentar yang ingin diprediksi"

    if st.button('Analisis') :
        sentiment = predict(text)
    
    st.success(sentiment)

main()





