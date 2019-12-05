from Model.Dee import Dee
import albert.tokenization as tokenization
from create_record import process

class Config(object):
    """CNN配置参数"""

    def __init__(self):
        self.seq_length = 256
        self.batch_size = 1
        self.bert_config_path = './model/albert_tiny/albert_config_tiny.json'
        self.lr = 1e-3
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


if __name__ == '__main__':
    import os
    import numpy as np
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = Config()
    oj = Dee(config)
    cache = [
        "证券代码：300042证券简称：朗科科技公告编号：2018-041",
        "深圳市朗科科技股份有限公司关于持股5%以上股东将已质押股票办理延期购回的公告",
        "本公司及董事会全体成员保证公告内容真实、准确和完整，没有虚假记载、误导性陈述或重大遗漏。",
        "深圳市朗科科技股份有限公司（以下简称“公司”、“本公司”）近日接到持股5%以上股东邓国顺先生通知，获悉：邓国顺先生于2017年5月4日将1800000股公司无限售条件流通股质押给了南京证券股份有限公司，2018年4月13日，其将上述股票质押办理了部分提前购回，提前购回股数为100股。",
        "2018年5月3日，邓国顺先生将上述股票质押的1799900股办理了延期购回，延期购回日期为2019年5月3日。",
        "具体事项如下：",
        "一、股东股份质押的基本情况",
        "1、股东股份被质押及延期购回的基本情况",
        "2、股东股份累计被质押的情况",
        "截至本公告披露之时，邓国顺先生持有本公司股份28900000股，占公司总股本的比例为21.63%。",
        "邓国顺先生持有的本公司股份累计被质押的数量为22339900股，累计被质押的股份占邓国顺先生所持有的本公司股份总数的77.30%，占公司总股本的比例为16.72%。",
        "二、备查文件",
        "1、《股票质押式回购交易协议书》；",
        "2、中国证券登记结算有限责任公司深圳分公司下发的《证券质押及司法冻结明细表》。",
        "深圳市朗科科技股份有限公司",
        "董事会",
        "2018年5月4日"
      ]

    max_length = [64, 256]
    vocab = tokenization.FullTokenizer('./albert/config/vocab.txt')
    data = process(cache, vocab, max_length)

    rt = oj.predict('./save/20191204204216/model.ckpt', data, cache)
    print('')
