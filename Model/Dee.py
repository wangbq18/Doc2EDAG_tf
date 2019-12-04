import tensorflow as tf
import logging
import time
import numpy as np
from datetime import timedelta
from tqdm import tqdm
from Model.Transformer import transformer_model, create_attention_mask_from_input_mask
from tools import read_json, save_json

logging.basicConfig(level=logging.INFO)

events = ['EquityFreeze', 'EquityRepurchase', 'EquityUnderweight', 'EquityOverweight', 'EquityPledge']

cache = read_json('./tfrecord/ner_tag.json')
tag_map = {}

for k in cache.keys():
    tag_map[cache[k]] = k

events_fields = {'EquityFreeze': ['EquityHolder',
                                  'FrozeShares',
                                  'LegalInstitution',
                                  'TotalHoldingShares',
                                  'TotalHoldingRatio',
                                  'StartDate',
                                  'EndDate',
                                  'UnfrozeDate'],
                 'EquityRepurchase': ['CompanyName',
                                      'HighestTradingPrice',
                                      'LowestTradingPrice',
                                      'RepurchasedShares',
                                      'ClosingDate',
                                      'RepurchaseAmount'],
                 'EquityUnderweight': ['EquityHolder',
                                       'TradedShares',
                                       'StartDate',
                                       'EndDate',
                                       'LaterHoldingShares',
                                       'AveragePrice'],
                 'EquityOverweight': ['EquityHolder',
                                      'TradedShares',
                                      'StartDate',
                                      'EndDate',
                                      'LaterHoldingShares',
                                      'AveragePrice'],
                 'EquityPledge': ['Pledger',
                                  'PledgedShares',
                                  'Pledgee',
                                  'TotalHoldingShares',
                                  'TotalHoldingRatio',
                                  'TotalPledgedShares',
                                  'StartDate',
                                  'EndDate',
                                  'ReleasedDate']}

# fields = []
# for k in events_fields.keys():
#     fields.extend(events_fields[k])

# fields = list(set(fields))
fields = read_json('./tfrecord/fields.json')


class Dee(object):
    """
    文档级事件抽取模型
    """

    def __init__(self, config):
        self.config = config
        self.__create_model()

    def __get_data(self, path, is_training):
        """
        一篇文本就是一组数据
        :param path:
        :param is_training:
        :return:
        """

        def parser(record):
            features = tf.parse_single_example(record,
                                               features={
                                                   'sentences': tf.FixedLenFeature([], tf.string),  # 文本内容，多个句子的组合
                                                   'sentences_mask': tf.FixedLenFeature([], tf.string),  # 命名实体的tag
                                                   'event_tag': tf.FixedLenFeature([], tf.string),  # 事件类型的tag
                                                   'ner_tag': tf.FixedLenFeature([], tf.string),
                                                   # sent_index,start_index,end_index
                                                   'path_tag': tf.FixedLenFeature([], tf.string),
                                                   'ner_list_index': tf.FixedLenFeature([], tf.string),
                                                   'ner_index': tf.FixedLenFeature([], tf.string),
                                                   'path_num': tf.FixedLenFeature([], tf.int64),
                                                   'path_event_type': tf.FixedLenFeature([40], tf.int64),
                                                   'path_entity_list': tf.FixedLenFeature([], tf.string),
                                               }
                                               )
            return features['sentences'], features['sentences_mask'], features['event_tag'], features['ner_tag'] \
                , features['path_tag'], features['ner_list_index'], features['ner_index'], features['path_num'], \
                   features['path_event_type'], features['path_entity_list']

        dataset = tf.data.TFRecordDataset(path)
        dataset = dataset.map(parser, num_parallel_calls=4)
        if is_training:
            dataset = dataset.batch(self.config.batch_size)
            dataset = dataset.shuffle(self.config.batch_size * 10)
            dataset = dataset.prefetch(self.config.batch_size * 10)
        else:
            dataset = dataset.batch(self.config.batch_size)
        iter = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
        sentences, sentences_mask, event_tag, ner_tag, path_tag, ner_list_index, ner_index, path_num, path_event_type, \
        path_entity_list = iter.get_next()

        sentences = tf.decode_raw(sentences, tf.int32)
        sentences_mask = tf.decode_raw(sentences_mask, tf.int32)
        event_tag = tf.decode_raw(event_tag, tf.float32)
        ner_tag = tf.decode_raw(ner_tag, tf.int32)
        path_tag = tf.decode_raw(path_tag, tf.int32)
        ner_list_index = tf.decode_raw(ner_list_index, tf.int32)
        ner_index = tf.decode_raw(ner_index, tf.int32)
        path_entity_list = tf.decode_raw(path_entity_list, tf.int32)
        path_num = tf.cast(path_num, tf.int32)
        path_event_type = tf.cast(path_event_type, tf.int32)

        sentences = tf.reshape(sentences, [-1, 64, 256])[0]
        sentences_mask = tf.reshape(sentences_mask, [-1, 64])[0]
        event_tag = tf.reshape(event_tag, [-1, 5])[0]
        ner_tag = tf.reshape(ner_tag, [-1, 64, 256])[0]
        path_tag = tf.reshape(path_tag, [-1, 40, 20, 100])[0]
        ner_list_index = tf.reshape(ner_list_index, [-1])
        ner_index = tf.reshape(ner_index, [-1, 3])
        path_num = tf.reshape(path_num, [-1, 1])[0]
        path_event_type = tf.reshape(path_event_type, [-1])
        path_entity_list = tf.reshape(path_entity_list, [-1, 40, 20])[0]

        def select_path(path_tag, path_num, path_event_type, path_entity_list):
            path_index = np.random.randint(0, path_num[0], size=1, dtype=np.int32)[0]
            return path_tag[path_index], path_index, path_event_type[path_index], path_entity_list[path_index]

        # 随机选择路径
        path_tag, path_index, path_event_type, path_entity_list = tf.py_func(select_path,
                                                                             [path_tag, path_num, path_event_type,
                                                                              path_entity_list],
                                                                             [tf.int32, tf.int32, tf.int32, tf.int32])
        path_tag = tf.reshape(path_tag, [20, 100])

        # 去除padding的ner_index
        def select_nert_index(ner_index, ner_list_index, path_entity_list):
            size = ner_list_index.argmin(axis=0)
            size1 = ner_index[:, 0].argmin(axis=0)
            size2 = path_entity_list.argmin(axis=0)
            return ner_index[:size1, :], ner_list_index[:size], path_entity_list[:size2]

        ner_index, ner_list_index, path_entity_list = tf.py_func(select_nert_index,
                                                                 [ner_index, ner_list_index, path_entity_list],
                                                                 [tf.int32, tf.int32, tf.int32])

        ner_list_index = tf.reshape(ner_list_index, [-1])
        ner_index = tf.reshape(ner_index, [-1, 3])
        path_entity_list = tf.reshape(path_entity_list, [-1])

        return sentences, sentences_mask, event_tag, ner_tag[:,
                                                     1:-1], path_tag, ner_list_index, ner_index, path_num, path_event_type, path_entity_list, iter.make_initializer(
            dataset)

    def __get_ner_loss(self, input, tag, mask, pos_size, dtype=tf.float32):
        with tf.name_scope("ner-decode"):
            with tf.variable_scope("var-decode", reuse=tf.AUTO_REUSE):
                g_v = tf.get_variable('g_v', [self.config.pos_size * self.config.pos_size, 1], dtype=dtype)
                g_v = tf.reshape(g_v, [pos_size, pos_size])
                loss, _ = tf.contrib.crf.crf_log_likelihood(input, tag, mask, g_v)

        return tf.reduce_mean(-loss), g_v

    def __get_entity_and_sentences_embedding(self, input, ner_list_index, ner_index, sentences_mask, dtype=tf.float32):
        """
        获取近包含自身信息的实体编码以及句子编码
        1. 求每个entity的embedding
        2. 加pos_embedding
        3. trainsformer
        4. 合并相同的entity
        :param input:
        :param ner_list_index:
        :param ner_index:
        :param sentences_mask:
        :return: 去除填充部分的实体编码和句子编码
        """

        # 句子位置向量
        with tf.variable_scope('sentence_pos', reuse=tf.AUTO_REUSE):
            sentence_embedding_table = tf.get_variable(  # [vocab_size, embedding_size]
                name="embedding_table",
                shape=[self.config.sentence_size, self.config.hidden_size],
                initializer=tf.truncated_normal_initializer(stddev=self.config.initializer_range, dtype=dtype),
                dtype=dtype)

        sentence_pos_embedding = tf.slice(sentence_embedding_table, [0, 0],
                                          [64, -1])
        sentences_embedding = tf.reduce_max(input, axis=1) + sentence_pos_embedding

        def select_entity(input, ner_index, sentences_embedding, sentences_mask):
            entity = []
            # 连接
            for x in ner_index:
                entity.append(input[x[0]][x[1]:x[2]].max(0))
            sentences_embedding = sentences_embedding[:sentences_mask.argmin(axis=0)]
            return entity, ner_index, ner_index, sentences_embedding, sentences_mask

        # 计算实体embedding和去除padding部分的句子embedding
        entity_embedding, _, _, sentences_embedding, _ = tf.py_func(select_entity, [input, ner_index,
                                                                                    sentences_embedding,
                                                                                    sentences_mask],
                                                                    [tf.float32, tf.int32, tf.int32, tf.float32,
                                                                     tf.int32])

        entity_embedding = tf.reshape(entity_embedding, [-1, self.config.hidden_size])
        sentences_embedding = tf.reshape(sentences_embedding, [-1, self.config.hidden_size])

        sentence_pos_embedding = tf.nn.embedding_lookup(sentence_embedding_table, ner_index[:, 0])
        entity_embedding = sentence_pos_embedding + entity_embedding

        all_embedding = tf.reshape(tf.concat([sentences_embedding, entity_embedding], axis=0),
                                   [1, -1, self.config.hidden_size])

        # 第二层transformer编码
        all_embedding = \
        self.__get_transformer_model(all_embedding, None, self.config, name='transformer-2', use_embedding=False)[0]

        # 分离，并合并同个实体的embedding
        def split(input, ner_list_index, sentence_size):
            sentence_size = sentence_size[0]
            sentences_embedding = input[:sentence_size]
            entity_embedding = input[sentence_size:]

            entity = []
            for i in range(1, len(ner_list_index)):
                c = entity_embedding[ner_list_index[i - 1]:ner_list_index[i]].max(axis=0)
                entity.append(c)
            return entity, ner_list_index, sentences_embedding

        entity_embedding, _, sentences_embedding = tf.py_func(split, [all_embedding, ner_list_index,
                                                                      tf.shape(sentences_embedding)],
                                                              [tf.float32, tf.int32, tf.float32])

        entity_embedding = tf.reshape(entity_embedding, [-1, self.config.hidden_size])
        sentences_embedding = tf.reshape(sentences_embedding, [-1, self.config.hidden_size])

        # 计算文档embedding
        document_embedding = tf.reduce_max(sentences_embedding, axis=0, keep_dims=True)

        return entity_embedding, sentences_embedding, document_embedding

    def __get_next_node(self, m, entity, tag=None):
        """
        返回当历史路径下，不同属性填入该字段的概率
        :param m: 历史信息
        :param entity:  候选属性,传入前已经融合当前字段embedding
        :return:
        """
        input = tf.reshape(tf.concat([m, entity], axis=0), [1, -1, self.config.hidden_size])
        input_encode = \
        self.__get_transformer_model(input, None, self.config, name='transformer-3', use_embedding=False)[0]

        def select_entity(input, shape):
            return input[shape[0]:], shape

        input_encode, _ = tf.py_func(select_entity, [input_encode, tf.shape(m)], [tf.float32, tf.int32])
        input_encode = tf.reshape(input_encode, [-1, self.config.hidden_size])

        with tf.variable_scope('path_dense', reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(input_encode, 2, name='path_logits_dense', reuse=tf.AUTO_REUSE)

        if tag != None:
            loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.one_hot(tag, 2))
            return logits, loss
        else:
            return tf.nn.sigmoid(logits)

    def __get_next_node2(self, m, entity, field_id):
        with tf.variable_scope('path', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('fields', reuse=tf.AUTO_REUSE):
                # 词向量
                fields_embedding_table = tf.get_variable(
                    name="embedding_table",
                    shape=[self.config.fields_size, self.config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=self.config.initializer_range, dtype=tf.float32),
                    dtype=tf.float32)

            fields_embedding = tf.nn.embedding_lookup(fields_embedding_table, field_id)

        field_embedding = tf.reshape(fields_embedding, [1, self.config.hidden_size])
        n_entity_embedding = entity + field_embedding

        input = tf.reshape(tf.concat([m, n_entity_embedding], axis=0), [1, -1, self.config.hidden_size])
        input_encode = \
            self.__get_transformer_model(input, None, self.config, name='transformer-3', use_embedding=False)[0]

        def select_entity(input, shape):
            return input[shape[0]:], shape

        input_encode, _ = tf.py_func(select_entity, [input_encode, tf.shape(m)], [tf.float32, tf.int32])
        input_encode = tf.reshape(input_encode, [-1, self.config.hidden_size])

        with tf.variable_scope('path_dense', reuse=tf.AUTO_REUSE):
            logits = tf.layers.dense(input_encode, 2, name='path_logits_dense', reuse=tf.AUTO_REUSE)

        # 更新m

        return tf.nn.sigmoid(logits), n_entity_embedding

    def __get_path_loss(self, fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list):
        """
        计算路径loss
        :param fields_embedding: [fields_size, hidden_size]
        :param sentences_embedding:
        :param entity_embedding:
        :return:
        """
        fields_embedding = tf.reshape(fields_embedding, [-1, self.config.hidden_size])

        def cond(fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list, index, loss_list):
            rt = tf.less(index + 1, tf.shape(fields_embedding)[0])[0]
            return rt

        def body(fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list, index, loss_list):
            field_embedding = tf.reshape(fields_embedding[index[0], :], [1, self.config.hidden_size])
            n_entity_embedding = entity_embedding + field_embedding

            # 去除padding的path_tag
            def get_path_tag(path_tag, shape):
                return path_tag[:shape[0]], shape

            path_tag_o, _ = tf.py_func(get_path_tag, [path_tag[index[0]], tf.shape(n_entity_embedding)],
                                       [tf.int32, tf.int32])
            path_tag_o = tf.reshape(path_tag_o, [-1])
            logits, loss = self.__get_next_node(sentences_embedding, n_entity_embedding, path_tag_o)
            loss_list += tf.reduce_mean(loss)

            # 更新m
            n_entity_embedding = tf.concat([field_embedding, n_entity_embedding], axis=0)
            n_entity_embedding = tf.reshape(n_entity_embedding[path_entity_list[index[0]]],
                                            [-1, self.config.hidden_size])
            sentences_embedding = tf.concat([sentences_embedding, n_entity_embedding], axis=0)

            return fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list, index + 1, loss_list

        index = tf.zeros(shape=1, dtype=tf.int32)
        loss_list = tf.zeros(shape=1, dtype=tf.float32)
        fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list, index, loss_list = \
            tf.while_loop(cond, body,
                          [fields_embedding, sentences_embedding, entity_embedding, path_tag, path_entity_list, index,
                           loss_list])
        return loss_list / tf.cast(index, tf.float32)

    def __get_input_embedding(self, sentences, config, use_factorization=True, dtype=tf.float32):
        """
        初始输入的embedding
        :param sentences:
        :param config: 相关参数
        :param use_factorization: 是否使用因式分解
        :return:
        """
        with tf.variable_scope('input_embedding', reuse=tf.AUTO_REUSE):
            # 词向量
            embedding_table = tf.get_variable(  # [vocab_size, embedding_size]
                name="embedding_table",
                shape=[config.vocab_size, config.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=config.initializer_range, dtype=dtype), dtype=dtype)

            full_position_embeddings = tf.get_variable(
                name="position_embedding_name",
                shape=[config.seq_length * 4, config.embedding_size],
                initializer=tf.truncated_normal_initializer(stddev=config.initializer_range, dtype=dtype), dtype=dtype)

            position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                           [config.seq_length, -1])

            embedding = tf.nn.embedding_lookup(embedding_table, sentences, name='embedding')
            embedding = embedding + tf.expand_dims(position_embeddings, axis=0)

            if use_factorization:
                embedding = tf.layers.dense(embedding, config.hidden_size, name='embedding_dense',
                                            reuse=tf.AUTO_REUSE)
        return embedding

    def __get_transformer_model(self, input, mask, config, name, use_embedding=True):
        """
        对输入使用transformer编码
        :param input:
        :param mask:
        :param config:
        :param name:
        :param use_embedding:
        :return:
        """
        if mask != None:
            attention_mask = create_attention_mask_from_input_mask(
                input, tf.expand_dims(mask, axis=1))
        else:
            attention_mask = None

        if use_embedding:
            with tf.variable_scope('%s_embedding' % (name), reuse=tf.AUTO_REUSE):
                input_embedding = self.__get_input_embedding(input, config)
        else:
            input_embedding = input

        encode = transformer_model(input_embedding, attention_mask, hidden_size=config.hidden_size,
                                   num_attention_heads=config.num_attention_heads,
                                   intermediate_size=config.hidden_size * 4, name=name)
        return encode

    def __graph(self, input, is_training):
        """

        :param input:
        :param cells:
        :param is_training:
        :return:
        """
        sentences, sentences_mask, event_tag, ner_tag, path_tag, ner_list_index, ner_index, path_num, path_event_type, path_entity_list = input

        # 第一层transformer编码，用于实体识别以及后续输入
        output1 = self.__get_transformer_model(sentences, sentences_mask, self.config, name='transformer-1')

        """
        计算ner的loss,和实际输出值
        需要注意的是：输入的sentence和sentence_mask是包含了[CLS],[SEP]，做ner时，要去掉首尾，sentence_mask也要做相应处理
        """
        with tf.variable_scope('ner', reuse=tf.AUTO_REUSE):
            # # 没有使用激活函数
            ner_ft = tf.layers.dense(output1, self.config.pos_size, name='ner_dense', reuse=tf.AUTO_REUSE)
            ner_loss, g_v = self.__get_ner_loss(ner_ft[:, 1:-1, :], ner_tag, sentences_mask - 2, self.config.pos_size)

        """
        计算融合上下文信息的实体和句子embedding
        """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            # 计算融合了上下文信息的实体和句子embedding
            entity_embedding, sentences_embedding, document_embedding = self.__get_entity_and_sentences_embedding(
                output1, ner_list_index,
                ner_index, sentences_mask)

        with tf.variable_scope('event_type', reuse=tf.AUTO_REUSE):
            # 事件类型分类
            event_type_logit = tf.layers.dense(document_embedding, self.config.event_type_size, activation=tf.tanh,
                                               name='event_type_dense',
                                               reuse=tf.AUTO_REUSE)
            # TODO 需要排查
            event_type_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=event_type_logit,
                                                                      labels=tf.reshape(event_tag, [-1, 5]))

        """
        计算事件填充loss
        1. 根据事件类型，计算连接概率，输入：当前路径m:句子，加前置路径(字段embedding+属性embedding)
        """
        with tf.variable_scope('path', reuse=tf.AUTO_REUSE):
            def get_field_embedding(event_type):
                fields_name = events_fields[events[event_type]]
                rt = [fields.index(_) for _ in fields_name]
                return np.array(rt, dtype=np.int32)

            event_fields_ids = tf.py_func(get_field_embedding, [path_event_type], [tf.int32])

            event_fields_ids = tf.reshape(event_fields_ids, [1, -1])

            with tf.variable_scope('fields', reuse=tf.AUTO_REUSE):
                # 词向量
                fields_embedding_table = tf.get_variable(
                    name="embedding_table",
                    shape=[self.config.fields_size, self.config.hidden_size],
                    initializer=tf.truncated_normal_initializer(stddev=self.config.initializer_range, dtype=tf.float32),
                    dtype=tf.float32)

            fields_embedding = tf.nn.embedding_lookup(fields_embedding_table, event_fields_ids)
            path_loss = self.__get_path_loss(fields_embedding, sentences_embedding, entity_embedding, path_tag,
                                             path_entity_list)

        return ner_loss, tf.reduce_mean(event_type_loss), tf.reduce_mean(path_loss), g_v

    def __get_ner_and_event_type_predict(self, sentences, sentences_mask):

        # 第一层transformer编码，用于实体识别以及后续输入
        output1 = self.__get_transformer_model(sentences, sentences_mask, self.config, name='transformer-1')

        """
        ner，需要注意解码部分
        """
        with tf.variable_scope('ner', reuse=tf.AUTO_REUSE):
            # # 没有使用激活函数
            ner_ft = tf.layers.dense(output1, self.config.pos_size, name='ner_dense', reuse=tf.AUTO_REUSE)[:, 1:-1]

        def viterbi_decode(ner_ft, g_v):
            p = [tf.contrib.crf.viterbi_decode(x, g_v)[0] for x in ner_ft]
            return p, g_v

        ner_tf_max, _ = tf.py_func(viterbi_decode, [ner_ft, self.g_v], [tf.int32, tf.int32])

        """
        根据预测结果生成ner_list_index和ner_index, 写入index时需要+1
        """

        def get_ner_index_and_ner_list_index(ner_logits, content, sentence_mask):
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

            return ner_logits, ner_index, ner_list_index

        _, ner_index, ner_list_index = tf.py_func(get_ner_index_and_ner_list_index,
                                                  [ner_tf_max, sentences,
                                                   sentences_mask], [tf.int32, tf.int32, tf.int32])

        ner_index = tf.reshape(ner_index, [-1, 3])
        ner_list_index = tf.reshape(ner_list_index, [-1])

        """
        计算融合上下文信息的实体和句子embedding
        """
        with tf.variable_scope('inference', reuse=tf.AUTO_REUSE):
            # 计算融合了上下文信息的实体和句子embedding
            entity_embedding, sentences_embedding, document_embedding = self.__get_entity_and_sentences_embedding(
                output1, ner_list_index,
                ner_index, sentences_mask)

        with tf.variable_scope('event_type', reuse=tf.AUTO_REUSE):

            # 事件类型分类
            event_type_logit = tf.layers.dense(document_embedding, self.config.event_type_size, activation=tf.tanh,
                                               name='event_type_dense',
                                               reuse=tf.AUTO_REUSE)

        event_type_logit = tf.nn.sigmoid(event_type_logit)

        return ner_tf_max, event_type_logit, entity_embedding, sentences_embedding, ner_list_index, ner_index

    def __create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            train = self.__get_data(self.config.train_path, True)
            dev = self.__get_data(self.config.dev_path, False)
            self.train_op = train[-1]
            self.dev_op = dev[-1]

            self.data = train[:-1]

            self.ner_loss, self.event_type_loss, self.path_loss, self.g_v = self.__graph(train[:-1], True)
            dev_ner_loss, dev_event_type_loss, dev_path_loss, _ = self.__graph(dev[:-1], False)

            self.dev_loss = 0.2 * dev_ner_loss + 0.2 * dev_event_type_loss + 0.6 * dev_path_loss
            self.loss = 0.2 * self.ner_loss + 0.2 * self.event_type_loss + 0.6 * self.path_loss

            """
            创建用于预测的tensor
            1. 预测实体
            2. 预测事件类型
            3. 填表
            """

            self.sentences_input = tf.placeholder(tf.int32, [None, self.config.seq_length], name='sentence_input')
            self.sentences_mask_input = tf.placeholder(tf.int32, [None, 1], name='sentence_mask')

            self.sentences_mask_input_r = tf.reshape(self.sentences_mask_input, [-1, 64])[0]

            self.ner_ft, self.event_type_logit, self.entity_embedding, self.sentences_embedding, self.ner_list_index, self.ner_index = \
                self.__get_ner_and_event_type_predict(self.sentences_input, self.sentences_mask_input_r)

            self.m = tf.placeholder(tf.float32, [None, self.config.hidden_size], name='m_input')
            self.entity_embedding_input = tf.placeholder(tf.float32, [None, self.config.hidden_size],
                                                         name='entity_embedding_input')
            self.field_id_input = tf.placeholder(tf.int32, [1],
                                                 name='field_id_input')

            self.next_node_p, self.encode_field_entity_embedding = self.__get_next_node2(self.m,
                                                                                         self.entity_embedding_input,
                                                                                         self.field_id_input)

            self.optimization = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss)
            # optimization = tf.train.MomentumOptimizer(self.config.lr, momentum=0.9).minimize(train_loss)
            # optimization = self.__get_optimization(train_loss, self.config.lr)

            self.saver = tf.train.Saver()
            # self.saver1 = tf.train.Saver(tf.trainable_variables())
            # self.saver_v = tf.train.Saver(tf.trainable_variables()[:1])

            self.summary_train_loss = tf.summary.scalar('train_loss', self.loss)
            self.summary_dev_loss = tf.summary.scalar('dev_loss', self.dev_loss)

            for index, x in enumerate(tf.trainable_variables()):
                logging.info('%d:%s' % (index, x))

    def evaluate(self, sess, count, test_data_count):
        sess.run(self.dev_op)
        all_loss = []
        batch_size = self.config.batch_size
        size = test_data_count // batch_size if test_data_count % batch_size == 0 else test_data_count // batch_size + 1
        for step in tqdm(range(size)):
            loss, summary = sess.run([self.dev_loss, self.summary_dev_loss])
            all_loss.append(loss)

        return sum(all_loss) / len(all_loss)

    def train(self, load_path, save_path, log_path, is_reload=False):
        log_writer = tf.summary.FileWriter(log_path, self.graph)
        logging.info('Training and evaluating...')
        start_time = time.time()
        total_batch = 0  # 总批次
        min_loss = -1
        last_improved = 0  # 记录上一次提升批次
        test_data_count = get_data_count(self.config.dev_path)
        train_data_count = get_data_count(self.config.train_path)
        size = train_data_count // self.config.batch_size if train_data_count % self.config.batch_size == 0 else train_data_count // self.config.batch_size + 1
        require_improvement = 3500  # 如果超过指定轮未提升，提前结束训练
        all_loss = 0

        flag = False
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:

            if is_reload:
                sess.run(tf.global_variables_initializer())
                self.saver.restore(sess, load_path)
            else:
                sess.run(tf.global_variables_initializer())
                # self.saver_v.restore(sess, load_path)

            for epoch in range(self.config.num_epochs):
                if flag:
                    break
                logging.info('Epoch:%d' % (epoch + 1))
                sess.run(self.train_op)
                for step in range(size):
                    if total_batch % self.config.print_per_batch == 0:
                        if total_batch % self.config.dev_per_batch == 0:
                            dev_loss = self.evaluate(sess, total_batch // self.config.dev_per_batch - 1,
                                                     test_data_count)
                            if min_loss == -1 or dev_loss <= min_loss:
                                self.saver.save(sess=sess, save_path=save_path)
                                improved_str = '*'
                                last_improved = total_batch
                                min_loss = dev_loss
                            else:
                                improved_str = ''

                            time_dif = get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Val loss: {2:>6.5}, Time: {3} {4}'
                            logging.info(
                                msg.format(total_batch, all_loss / self.config.print_per_batch, dev_loss,
                                           time_dif, improved_str))
                        else:
                            time_dif = get_time_dif(start_time)
                            msg = 'Iter: {0:>6}, Train Loss: {1:>6.5}, Time: {2}'
                            logging.info(msg.format(total_batch, all_loss / self.config.print_per_batch, time_dif))
                        all_loss = 0

                    loss_train, summary, _ = sess.run([self.loss, self.summary_train_loss, self.optimization])  # 运行优化
                    log_writer.add_summary(summary, total_batch)
                    all_loss += loss_train
                    total_batch += 1

                    if total_batch - last_improved > require_improvement:
                        logging.info("No optimization for a long time, auto-stopping...")
                        flag = True
                        break
                if flag:
                    break

    def predict(self, model_path, data, raw_data):
        """
        预测数据
        :param model_path:
        :param data:
        :return:
        """

        def create_tree(self, sess, field_ids, m, entity, entity_name):
            """
            递归生成路径树
            :param sess:
            :param field_id: 剩余field
            :param m: 历史路径
            :param entity: 实体
            :return:
            """
            rt = {}
            for field_id in field_ids:
                next_node_p, encode_field_entity_embedding = sess.run(
                    [self.next_node_p, self.encode_field_entity_embedding],
                    feed_dict={self.m: m, self.entity_embedding_input: entity, self.field_id_input: [field_id]})
                for index, x in enumerate(next_node_p):
                    if x >= 0.5:
                        # 历史路径的更新
                        m = np.concatenate([m, np.reshape(encode_field_entity_embedding[index], [1, -1])])
                        rt[entity_name] = create_tree(sess, field_ids[1:], m, entity, event_name)
            return rt

        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            self.saver.restore(sess, model_path)
            # 先预测实体
            entity_embedding, event_type_logit, sentences_embedding, ner_list_index, ner_index = sess.run(
                [self.entity_embedding, self.event_type_logit, self.sentences_embedding, self.ner_list_index,
                 self.ner_index],
                feed_dict={self.sentences_input: data[0], self.sentences_mask_input: data[1]})

            # 生成实体
            entity = []
            for x in ner_index:
                cache = raw_data[x[0]][x[1]-1:x[2]-1]
                if cache not in entity:
                    entity.append(cache)

            # 构建路径树
            rt = {}
            for index, x in enumerate(event_type_logit[0]):
                if x < 0.5:
                    continue
                # 依次处理
                event_name = events[index]
                ids = [fields.index(field) for field in events_fields[event_name]]
                # 递归生成树
                # TODO 加上实体名称
                rt[event_name] = create_tree(self, sess, ids, sentences_embedding, entity_embedding, entity_name=entity)
            return rt

    def print(self):
        with tf.Session(graph=self.graph, config=tf.ConfigProto(allow_soft_placement=True,
                                                                gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(self.dev_op)
            ner_index = sess.run(self.ner_index)
            print(ner_index)


def get_data_count(path):
    c = 0
    if isinstance(path, list):
        for x in path:
            for record in tf.python_io.tf_record_iterator(x):
                c += 1
    else:
        for record in tf.python_io.tf_record_iterator(path):
            c += 1
    return c


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_f1(data):
    F = data[2] / data[0]
    if data[1] == 0:
        G = 0
    else:
        G = data[2] / data[1]
    if F + G == 0:
        return 0
    return F * G * 2 / (F + G)
