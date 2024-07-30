import os
import json
import pandas as pd
import numpy as np
from collections import Counter

def most_common_string(strings):
    # 使用Counter统计每个字符串的出现次数
    string_counter = Counter(strings)
    
    # 获取出现次数最多的字符串和其出现次数
    most_common, count = string_counter.most_common(1)[0]
    
    return most_common


root, dirs, _ = next(os.walk("/root/Agents/llama2_inference/scales/moe_gate_trained_alternate_with_system_prompt_v2"))
for d in dirs:
    files = next(os.walk(f"{root}/{d}/dialog"))[-1]
    if len(files)%20!=0 or len(files)==100:
        continue
    # if len(files)!=148:
    #     continue
    print(f"{root}/{d}/dialog")
    with open(f"{root}/{d}/config.json","r") as f:
        config = json.load(f)
    print("base_model",config['base_model'])
    print("lora_weights",config['lora_weights'])
    print(config['active_factor_polar'], config['active_prof'])
    factor_dict = {}
    for file in files:
        item_path = f"{root}/{d}/dialog/{file}"
        with open(item_path,'r') as f:
            item = json.load(f) 

        judge_list = item['judge_list']
        ans_list = [judge['option'] for judge in judge_list]
        ans = most_common_string(ans_list)
        assert ans in ['A', 'B', 'C', 'D', 'E']

        if item['polarity']=='pos':
            score = 6- (ord("F")-ord(ans))
        elif item['polarity']=='neg':
            score = ord("F")-ord(ans)

        if item['factor'] not in factor_dict:
            factor_dict[item['factor']] = [score]
        else:
            factor_dict[item['factor']] += [score]       

    factor_dict = dict(sorted(factor_dict.items()))
    factor_scores = []
    for key, value in factor_dict.items():
        factor_scores.append(
            {
                "factor":key, "sum":np.sum(value),"mean":np.mean(value)
            }
        )
        print(key, np.sum(value), np.mean(value))
        print(dict(sorted(Counter(value).items())))

    import pandas as pd

    pd.DataFrame(factor_scores).T.to_csv(f"{root}/{d}/results.csv")