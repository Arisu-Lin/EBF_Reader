import os
import json
import random
# from tools.sdp_extra import data_sdp
random.seed(0)

def filt(train):
    data = []
    max_sent = 0
    for d in train:
        if len(d['context'][0][1])>max_sent:
            max_sent = len(d['context'][0][1])
        if d['answer'] not in ['yes','no','unknown']:
            question = d['question']
            context = ''.join(c for c in d['context'][0][1])
            if d['answer'] in context[0:512-3-len(question)]:
                data.append(d)
        else:
            data.append(d)
    print(len(data),max_sent)
    return data

with open("cail2020/train.json","r",encoding="utf-8") as fp:
    data_big = json.load(fp)


data = filt(data_big)
random.shuffle(data)

train_data = data[0:3000]
dev_data = data[3000:4000]
test_data = data[4000:]
print('train:%d, dev:%d, test:%d'%(len(train_data), len(dev_data), len(test_data)))
with open("data_2020/train.json","w",encoding="utf-8") as f:
    json.dump(train_data,f,ensure_ascii=False)
    print("载入文件完成...")
with open("data_2020/dev.json","w",encoding="utf-8") as f:
    json.dump(dev_data,f,ensure_ascii=False)
    print("载入文件完成...")
with open("data_2020/test.json","w",encoding="utf-8") as f:
    json.dump(test_data,f,ensure_ascii=False)
    print("载入文件完成...")



