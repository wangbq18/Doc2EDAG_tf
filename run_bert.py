from Model.Dee_bert import Dee
from tools import get_time
import os

class Config(object):
    """CNN配置参数"""

    def __init__(self):
        self.seq_length = 256
        self.batch_size = 1
        self.bert_config_path = './albert/config/albert_config_tiny.json'
        self.lr = 1e-5
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
        self.dev_per_batch = 5000  # 多少轮验证一次

        self.train_path = './tfrecord/train.record'
        self.dev_path = './tfrecord/dev.record'
        self.num_epochs = 100

    def get_all_number(self):
        rt = {}
        for key, value in vars(self).items():
            rt[key] = value
        return rt


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    now_tim = get_time()

    l_path = './save/albert_tiny/albert_model.ckpt'
    # l_path = None
    # l_path = './model/albert_ft/20191123144355/model.ckpt'

    path = './save/%s/model.ckpt' % (now_tim)

    config = Config()
    oj = Dee(config)
    if not os.path.exists('./save/%s/' % (now_tim)):
        os.makedirs('./save/%s/' % (now_tim))

    # 开始训练
    oj.train(l_path, path, "./data/log/", True)
