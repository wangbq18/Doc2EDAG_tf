from Model.Dee_bert import Dee
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
        "证券代码：601168证券简称：西部矿业编号：临2009-001",
        "西部矿业股份有限公司关于发起人股东减持公司股份后持股比例低于5%的公告",
        "本公司董事会及全体董事保证本公告内容不存在任何虚假记载、误导性陈述或者重大遗漏，并对其内容的真实性、准确性和完整性承担个别及连带责任。",
        "2009年1月5日，本公司接到股东——新疆塔城国际资源有限公司（以下简称“塔城国际”）和上海海成物资有限公司（以下简称“海成物资”）递交减持公司股票的通知，具体内容如下：",
        "2008年11月12日至2008年12月10日，塔城国际通过上海证券交易所竞价交易系统共计卖出西部矿业3000000股，所减持股份占西部矿业已发行股份总数的0.1259%；海成物资通过上海证券交易所竞价交易系统共计卖出西部矿业2330000股，所减持股份占西部矿业已发行股份总数的0.0978%。",
        "按照《上市公司收购管理办法》的规定，作为西部矿业的股东，塔城国际与海成物资构成一致行动关系。",
        "至此，塔城国际和海成物资合计持有西部矿业117820000股，占西部矿业已发行股份总数的4.9442%。",
        "截至2009年1月5日收盘清算后，股东塔城国际和海成物资仍合计持有本公司股票117820000股，占已发行股份总数的4.9442%。",
        "特此公告。",
        "西部矿业股份有限公司",
        "董事会",
        "二○○九年一月七日"
      ]

 #    cache = ['四川和邦生物科技股份有限公司（以下简称“公司”）控股股东四川和邦投资集团有限公司（以下简称“和邦集团”）持有公司的股份总数为2496945803股，占公司总股本比例28.27%',
 # '本次股份解质后，和邦集团剩余质押的股份数量为2339806603股，占其持股总数比例为93.71%，占公司总股本比例为 26.49%',
 # '和邦集团及其一致行动人贺正刚先生合并持有公司股份2909577803股，占公司总股本比例32.95%',
 # '本次股份解质后，和邦集团及其一致行动人贺正刚先生合并剩余质押的股份数量为2505806603股，占其持股总数比例为86.12%，占公司总股本比例为28.37%']

    max_length = [64, 256]
    vocab = tokenization.FullTokenizer('./albert/config/vocab.txt')
    data = process(cache, vocab, max_length)

    rt = oj.predict('./save/20191205205408/model.ckpt', data, cache)
    paths = get_path(rt)
    print(paths)
