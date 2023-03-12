import numpy as np
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import re
import pickle

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
        text = "\nText: "+ tempTxt + "\nSentiment: Komentar Cyberbulying\n"
        return text
    else:
        text = "\nText: "+ tempTxt + "\nSentiment: Komentar Biasa\n"
        return text
    

# txt = 'jangan tolol banget deh jadi orang, sadar diri'
txt = 'Kamu baik banget deh, boleh ga aku temenan sama kamu'
# txt = 'sebenrnya kamutuh baik, tapi kenapasih kamu sering nyusahin orang'

print(predict(txt))



# # Load model
# model = load_model('modelInstagram.h5')

# def predict_Sentiment(txt):
#     max_fatures = 100
#     tokenizer = Tokenizer(num_words=max_fatures, split=' ')
#     tokenizer.fit_on_texts(txt)
#     X = tokenizer.texts_to_sequences(txt)
#     X = pad_sequences(X, maxlen=114)

#     Y_pred = model.predict(X)

#     for i in range(len(Y_pred)):
#         if np.argmax(Y_pred[i]) == 0:
#             print('Text:', txt[i], '\nSentiment: Komentar Cyberbulying')
#         else:
#             print('Text:', txt[i], '\nSentiment: Komentar Biasa')

# txt = [
#     'kebiasaan balajaer nyampah d ig para artis..suka2 yg punya ig lah mau bikin caption apa,kok balajaer yg heboh dan asik ceramahin yg punya ig.tar lama2 d bikinin lagu sm teh melly loh balajaer yg berjudul',
#     'Kamu baik banget deh, boleh ga aku temenan sama kamu',
#     'jangan tolol banget deh jadi orang, sadar diri'
#        ]

# predict_Sentiment(txt)