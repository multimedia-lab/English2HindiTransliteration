from keras.models import model_from_json
from keras.preprocessing.sequence import pad_sequences
import pickle
import os
import pandas as pd
import numpy as np


current_filepath=os.path.dirname(os.path.realpath(__file__))
base_dir=os.path.dirname(current_filepath)
model_name="EnglishToHindi_v13"
max_len=40


#char encoder
with open(os.path.join(base_dir,"model",model_name+"_eng2index.pickle"),"rb") as file:
    eng2index=pickle.load(file)

with open(os.path.join(base_dir,"model",model_name+"_hindi2index.pickle"),"rb") as file:
    hindi2index=pickle.load(file)

#char decoder
index2hindi=dict((i,char) for char,i in hindi2index.items())
index2eng=dict((i,char) for char,i in eng2index.items())


#model loading
with open(os.path.join(base_dir,"model",model_name+"_json.json"),"r") as file:
    json_str=file.read()
model=model_from_json(json_str)

weight_path=os.path.join(base_dir,"model",model_name+"_weights.h5")
model.load_weights(weight_path)


# read test
text_case_filename="test_case2.xlsx"
df=pd.read_excel(os.path.join(base_dir,"test",text_case_filename),encoding="utf-8")
print(df)

texts=df["English"]
char_seq_list=[list(text.lower().strip()) for text in texts]

encoded_seq_list=[[eng2index[char]  for char in char_seq] for char_seq in char_seq_list]
padded_seq=pad_sequences(encoded_seq_list,maxlen=max_len,padding="post",value=eng2index[" "])
pred=model.predict(padded_seq)
dic={}
dic["English"]=[]
dic["Hindi"]=[]
dic["Predicted"]=[]
for wordSeq,prob in zip(encoded_seq_list,pred):
    hi=""
    eng=""
    prob = np.argmax(prob, axis=-1)
    for w, pred in zip(wordSeq, prob):
        eng = eng + index2eng[w]
        hi=hi+index2hindi[pred]
    dic["English"].append(eng)
    dic["Predicted"].append(hi)
dic["Hindi"]=df["Hindi"]
pd.DataFrame(dic).to_excel(os.path.join(base_dir,"test","output_13_"+text_case_filename))
    
    
    

    










