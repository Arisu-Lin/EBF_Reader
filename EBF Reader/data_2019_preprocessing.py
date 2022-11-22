import json
import re
# from tools.sdp_extra import data_sdp


ANSWER_MARGIN=50
MAX_SEQ_LEN=512

def process_context(line):
    line = line.replace("&middot;", "", 100)
    spans = re.split('([,。，；;])', line)
    if len(spans) <= 2:
        spans = re.split('([，。])', line)
    if len(spans) <= 2:
        spans = re.split('([;；，。,])', line)
    assert len(spans) > 2, spans
    spans_sep = []
    for i in range(len(spans)//2):
        spans_sep.append(spans[2*i]+spans[2*i+1])
    assert len(spans_sep) > 0, spans
    return [[spans_sep[0],spans_sep]]

def supporting_facts(answers, context_lines):
    res = []
    ans_st = answers['answer_start']
    ans_ed = answers['answer_start'] + len(answers['text'])-1
    start=0
    a = 0
    b = 0
    for i in range(len(context_lines)):
        line_span = list(range(start,start+len(context_lines[i])))
        start=start + len(context_lines[i])
        if ans_st in line_span:
            a = i
        if ans_ed in line_span:
            b = i
            break
        

    if answers['text'] not in ['YES','NO','unknown']:
        for u in range(a,b+1):
            res.append([context_lines[0], u])
           
    return res

ori = ['big_train_data','dev_ground_truth','test_ground_truth']
tgt = ['train','dev','test']
for i in range(3):
    with open('data_2019/'+tgt[i]+'.json', 'w', encoding='utf8') as fw:
        fin = open('cail2019/'+ori[i]+'.json', 'r', encoding='utf8')
        line = fin.readline()
        dic = json.loads(line)
        results = []
        _id = 0
        for item in dic['data']:
            id = item['caseid']
            domain = item['domain']
            para = item['paragraphs'][0]
            context = para['context']
            casename = para['casename']
            qas = para['qas']
            for qa in qas:
                question = qa['question']
                qid = qa['id']
                is_unknown = qa['is_impossible']
                answers = qa['answers']
                ans_end = 0
                if answers:
                    ans_starts = answers[0]['answer_start'] 
                    ans_end = ans_starts + len(answers[0]['text'])-1
                if ans_end+3+len(question)>512:
                    continue
                if answers and context.count(answers[0]['text'])>1 :
                    continue

                conv_dic = {}
                conv_dic['_id'] = _id
                conv_dic['context'] = process_context(context[0:512-3-len(question)])
                conv_dic['question'] = question
                conv_dic['supporting_facts'] = []
                if is_unknown == "true":
                    conv_dic['answer'] = "unknown"
                else:
                    if answers[0]['text'] in ["YES","NO"]:
                        conv_dic['answer'] = answers[0]['text'].lower()
                    else:
                        conv_dic['answer'] = answers[0]['text']
                    ans = answers[0]
                    conv_dic['supporting_facts'] = supporting_facts(ans, conv_dic['context'][0][1])
                results.append(conv_dic)
                _id+= 1
        fin.close()
        # results = data_sdp(results)
        print(tgt[i],len(results))
        fw.write(json.dumps(results, ensure_ascii=False, indent=4))
print('FIN')