from Model.Dee import Dee


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
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    config = Config()
    oj = Dee(config)
    oj.print()