from Levenshtein import distance
from nltk import ngrams
from collections import Counter

def cer(predicted,actual):
    return distance(predicted,actual)

def create_ngram_map(text,n=4):
    ngram_list=[]
    for i in range(n):
        ngram_list=ngram_list+[ng for ng in ngrams(text,i)]
    return Counter(ngram_list)
    

def bleu(predicted,actual,order=4):
    bleu_prod=1
    for i in range(order):
        pred_ngram_map=create_ngram_map(predicted,i+1)
        act_ngram_map=create_ngram_map(actual,i+1)
        bleu_sum=0
        ngsize=len(act_ngram_map)
        for word,count in act_ngram_map.items():
            if pred_ngram_map.get(word):
                bleu_sum=bleu_sum+min(pred_ngram_map[word],count)
        bleu_prod=bleu_prod*(bleu_sum/ngsize)
    return bleu_prod**(1/order)



        
            






    

    
