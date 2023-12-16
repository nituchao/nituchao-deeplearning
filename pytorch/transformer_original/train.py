import pandas as pd
from nltk import word_tokenize

def load_data(path):
    en = []
    cn = []

    with open(path, 'r') as f:
        for line in f:
            line = line.strip().split('\t')

            en.append(["BOS"] + word_tokenize(line[0].lower()) + ["EOS"])
            cn.append(["BOS"] + word_tokenize(" ".join([w for w in line[1]])) + ["EOS"])

    return en, cn

def build_dict(self, sentences, max_words = 50000):
    word_count = Counter()

    for sentence in sentences:
        for s in sentence:
            word_count[s] += 1
    
    ls = word_count.most_common(max_words)
    # 这是为了筛选出频率最高的词汇
    # 返回的对象应该是一个元组列表，像这样[("a", 10), ...]
    total_words = len(ls) + 2

    word_dict = {w[0]: index + 2 for index, w in enumerate(ls)}
    word_dict['UNK'] = args.UNK
    word_dict['PAD'] = args.PAD

    index_dict = {v: k for k, v in word_dict.items()}

path = "/Users/bytedance/Documents/Workspace/dance_codes/nituchao_deeplearning/pytorch/transformer_original/data_train.csv"
en, cn = load_data(path)
print(en)
print(cn)