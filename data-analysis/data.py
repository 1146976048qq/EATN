import pandas as pd
import numpy as np
import gensim


model = gensim.models.KeyedVectors.load_word2vec_format('glove_model.txt', binary=False)
print("Test Embedding:\n", model['test'], "\n ********", "***"*10)




lap = open('/Users/kaizhang/Desktop/data-analysis/Aspect_word/laptop_aspect_words.txt')
lines = lap.readlines()
lap_emb = []
for line in lines:
    #print(line)
    line = ''.join(line)
    line = line.replace('\n', '')
    try:
        lap_emb.append(model[line])
    except:
        lap_emb.append(0)

#### 是list类型
print(type(lap_emb))

#### lab_emb 是所有的laptop aspect的300维向量表征！  # 4400个word！！！
#print(len(lap_emb))



res = open('/Users/kaizhang/Desktop/data-analysis/Aspect_word/restaurant_aspect_words.txt')
lines = res.readlines()
res_emb = []
for line in lines:
    #print(line)
    line = ''.join(line)
    line = line.replace('\n', '')
    try:
        res_emb.append(model[line])
    except:
        res_emb.append(0)

#### 是list类型
print("res:", type(res_emb))

#### res_emb 是所有的restaurant aspect的300维向量表征！  # 6586个word！！！
print("res:", len(res_emb))




twi = open('/Users/kaizhang/Desktop/data-analysis/Aspect_word/twitter_aspect_words.txt')
lines = twi.readlines()
twi_emb = []
for line in lines:
    #print(line)
    line = ''.join(line)
    line = line.replace('\n', '')
    try:
        twi_emb.append(model[line])
    except:
        twi_emb.append(0)

#### 是list类型
print("twi:", type(twi_emb))

#### twi_emb 是所有的laptop aspect的300维向量表征！  # 10789个word！！！
print("twi:", len(twi_emb))





