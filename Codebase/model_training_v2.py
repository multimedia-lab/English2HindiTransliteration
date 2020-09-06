#reading dataset
from input_output import read_data
import os
from more_itertools import flatten
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential,Model
from keras.layers import LSTM,Embedding,Dense,TimeDistributed,Dropout,Bidirectional,Flatten,SpatialDropout1D,Conv1D,MaxPool1D
from keras.layers import Input
from keras.layers.merge import add
from sklearn.model_selection import train_test_split
import numpy as np
import pickle


model_name="EnglishToHindi_v11_ed"

current_filepath=os.path.dirname(os.path.realpath(__file__))
base_dir=os.path.dirname(current_filepath)
dataset_filepath=os.path.join(base_dir,"dataset")
filename="English2Hindi_v2.xlsx"
df=read_data(filename,dataset_filepath).sample(frac=1)


#preprocessing 
df["English"]=df["English"].apply(lambda x: str(x).lower().strip())
df["Hindi"]=df["Hindi"].apply(lambda x: str(x).strip())

#serializing words
english_seqlist=[list(word) for word in df["English"]]
hindi_seqlist=[list(word) for word in df["Hindi"]]

#creating encoder of charachers
english_charset=set(flatten(english_seqlist))
english_charset.add(" ") if " " not in english_charset else None
eng2index=dict((char,i) for i,char in enumerate(english_charset))

hindi_charset=set(flatten(hindi_seqlist))
hindi_charset.add(" ") if " " not in hindi_charset else None
hindi2index=dict((char,i) for i,char in enumerate(hindi_charset))


#sequence encoding
encoded_eng_seqlist=[[eng2index[char]  for char in seq ] for seq in english_seqlist]
encoded_hindi_seqlist=[[hindi2index[char]  for char in seq ] for seq in hindi_seqlist]

with open(os.path.join(base_dir,"model",model_name+"_hindi2index.pickle"),mode="wb") as file:
    pickle.dump(hindi2index,file)

with open(os.path.join(base_dir,"model",model_name+"_eng2index.pickle"),mode="wb") as file:
    pickle.dump(eng2index,file)


#maxsequence length
max_len=40

# sequence padding
eng_padded_seq=pad_sequences(maxlen=max_len,sequences=encoded_eng_seqlist,padding="post",value=eng2index[" "])
hindi_padded_seq=pad_sequences(maxlen=max_len,sequences=encoded_hindi_seqlist,padding="post",value=hindi2index[" "])

#one hot encoding of hindi sequence
y=[to_categorical(seq,num_classes=len(hindi2index)) for seq in hindi_padded_seq]

eng_train,eng_test,y_train,y_test=train_test_split(eng_padded_seq,y,test_size=0.1)

#defining architecture of network

input_layer = Input(shape=(max_len,), dtype=np.int32)
embd=Embedding(input_dim=len(eng2index),output_dim=22, input_length= max_len)(input_layer)
embd=Dropout(0.1)(embd)
x=Bidirectional(LSTM(units=140,return_sequences=True,recurrent_dropout=0.2,dropout=0.2))(embd)
x_rnn=Bidirectional(LSTM(units=140,return_sequences=True,recurrent_dropout=0.2,dropout=0.2))(x)
x=add([x,x_rnn])
out=TimeDistributed(Dense(len(hindi2index),activation="softmax"))(x)
model = Model(input_layer, out)
model.summary()

model.compile(optimizer="rmsprop",loss="categorical_crossentropy",metrics=["accuracy"])
history=model.fit(eng_train,np.array(y_train),validation_data=(eng_test,np.array(y_test)),batch_size=32,epochs=10,verbose=1)
model_json=model.to_json()
with open(os.path.join(base_dir,"model",model_name+"_json.json"),"w") as json_file:
    json_file.write(model_json)
model.save_weights(os.path.join(base_dir,"model",model_name+"_weights.h5"))

from matplotlib import pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

from matplotlib import pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()











