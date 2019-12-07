from Model.Dee_albert import Dee
import albert.tokenization as tokenization
from create_record import process
from create_record import get_path

class Config(object):
    """CNN配置参数"""

    def __init__(self):
        self.seq_length = 256
        self.batch_size = 1
        self.bert_config_path = './albert/config/albert_config_tiny.json'
        self.lr = 1e-3
        self.lamdba = 0.2
        self.hidden_size = 312
        self.sentence_size = 64
        self.fields_size = 35
        self.event_type_size = 5
        self.embedding_size = 128
        self.num_attention_heads = 6
        self.initializer_range = 0.02
        self.pos_size = 49
        self.vocab_size = 21128

        self.print_per_batch = 100  # 每多少轮输出一次结果
        self.dev_per_batch = 1000  # 多少轮验证一次

        self.train_path = './tfrecord/train.record'
        self.dev_path = './tfrecord/sample_train.record'
        self.num_epochs = 100

    def get_all_number(self):
        rt = {}
        for key, value in vars(self).items():
            rt[key] = value
        return rt

from tools import read_json
cache = read_json('./tfrecord/ner_tag.json')
tag_map = {}

for k in cache.keys():
    tag_map[cache[k]] = k

def get_ner_index_and_ner_list_index(ner_logits, content, sentence_mask):
    sentence_mask = np.array(sentence_mask, np.int32)
    sentence_mask = np.reshape(sentence_mask, [-1, 64])[0]
    content = np.array(content, np.int32)
    sentences_size = sentence_mask.argmin(axis=0)
    ner_logits = ner_logits[:sentences_size]
    content = content[:sentences_size]
    ner_map = {}
    for index, x in enumerate(ner_logits):
        cache = [index, -1, -1]
        x_c = [str(_) for _ in content[index]][:sentence_mask[index]]
        for index_y, y in enumerate(x[:sentence_mask[index]]):
            if '_B' in tag_map[y]:
                if cache[1] != -1:
                    cache[2] = index_y + 1
                    name = '-'.join(x_c[cache[1]:cache[2]])
                    if name not in ner_map.keys():
                        ner_map[name] = []
                    ner_map[name].append(cache)
                    cache = [index, index_y, -1]
                else:
                    cache[1] = index_y + 1
            if 'O' == tag_map[y] and cache[1] != -1:
                cache[2] = index_y + 1
                name = '-'.join(x_c[cache[1]:cache[2]])
                if name not in ner_map.keys():
                    ner_map[name] = []
                ner_map[name].append(cache)
                cache = [index, -1, -1]

        # 处理最后一个字符也是实体的情况
        if cache[1] != -1:
            cache[2] = index_y + 1
            name = '-'.join(x_c[cache[1]:cache[2]])
            if name not in ner_map.keys():
                ner_map[name] = []
            ner_map[name].append(cache)

    ner_list_index = [0]
    ner_index = []
    for k in ner_map.keys():
        ner_index.extend(ner_map[k])
        ner_list_index.append(len(ner_index))

    return ner_logits, np.array(ner_index, np.int32), np.array(ner_list_index, np.int32)

if __name__ == '__main__':
    import os
    import numpy as np
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = Config()
    oj = Dee(config)
    cache = ["证券代码：600641证券简称：万业企业公告编号：临2009-003",
             "上海万业企业股份有限公司关于公司高管增持公司股票的公告",
             "本公司及董事会全体成员保证公告内容的真实、准确和完整，对公告的虚假记载、误导性陈述或者重大遗漏负连带责任。",
             "2008年12月31日，公司董事会秘书获悉,公司监事长张峻根据原计划在当日从二级市场购入本公司股票50000股，截止目前公司高管持有公司股票共计431090股。",
             "公司董事会将根据中国证监会及上海证券交易所有关规定，将上述增持股票申请锁定，同时监督并提醒其按照相关法律法规买卖所持有的本公司股票。",
             "特此公告。",
             "上海万业企业股份有限公司", "董事会", "2009年1月5日"]

 #    cache = ['四川和邦生物科技股份有限公司（以下简称“公司”）控股股东四川和邦投资集团有限公司（以下简称“和邦集团”）持有公司的股份总数为2496945803股，占公司总股本比例28.27%',
 # '本次股份解质后，和邦集团剩余质押的股份数量为2339806603股，占其持股总数比例为93.71%，占公司总股本比例为 26.49%',
 # '和邦集团及其一致行动人贺正刚先生合并持有公司股份2909577803股，占公司总股本比例32.95%',
 # '本次股份解质后，和邦集团及其一致行动人贺正刚先生合并剩余质押的股份数量为2505806603股，占其持股总数比例为86.12%，占公司总股本比例为28.37%']

    max_length = [64, 256]
    vocab = tokenization.FullTokenizer('./albert/config/vocab.txt')
    data = process(cache, vocab, max_length)

    rt = oj.predict('./save/20191207095809/model.ckpt', data, cache)
    # get_ner_index_and_ner_list_index(rt, data[0], data[1])
    # print('')
    paths = get_path(rt)
    print(paths)
