import pandas as pd
import os
from measure import cer,bleu


#reading file
filename="output_4_test_case2.xlsx"
filepath="E:/GithubProjects/HindiTransliteration/test"
df=pd.read_excel(os.path.join(filepath,filename))
resultdic={}
resultdic["English"]=[]
resultdic["Hindi"]=[]
resultdic["Predicted"]=[]
resultdic["CER"]=[]
resultdic["BLEU"]=[]

for i,row in df.iterrows():
    pred=row["Predicted"]
    act=row["Hindi"]
    _cer=cer(pred.strip(),act.strip())
    _bleu=bleu(pred,act)
    resultdic["English"].append(row["English"])
    resultdic["Predicted"].append(pred)
    resultdic["Hindi"].append(act)
    resultdic["CER"].append(_cer)
    resultdic["BLEU"].append(_bleu)

pd.DataFrame(resultdic).to_excel(os.path.join(filepath,"score_"+filename))


