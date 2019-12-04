import json
import tensorflow as tf
import random
from tqdm import tqdm
import numpy as np
import tensorflow.contrib.keras as kr
from tools import *
import albert.tokenization as tokenization



def get_path(tree):
    """
    深度优先遍历
    :param tree: 树
    :return: 全部子路径
    """
    if len(tree.keys())==0:
        return []
    cache = [k for k in tree.keys()]

    rt = []
    for k in cache:
        child = get_path(tree[k])
        if len(child)==0:
            rt.append([k])
        else:
            for c in child:
                rt.append([k]+c)
    return rt



def process(data, tokenization, max_length):
    content = []
    masks = []
    for x in data:
        cache = tokenization.convert_tokens_to_ids(x)
        if len(cache) == 0:
            cache.append(tokenization.vocab['[PAD]'])
        padding_value = tokenization.vocab['[PAD]']

        seq = [tokenization.vocab['[CLS]']]
        seq.extend(cache[-max_length[1] + 2:])
        seq.append(tokenization.vocab['[SEP]'])

        mask = [len(seq) - 3]

        cache = kr.preprocessing.sequence.pad_sequences([seq], max_length[1], value=padding_value, padding='post',
                                                        truncating='post')
        content.extend(cache)
        masks.append(mask)
    content = content[:max_length[0]]
    masks = masks[:max_length[0]]
    # padding
    for index in range(max_length[0] - len(content)):
        content.extend(kr.preprocessing.sequence.pad_sequences(
            [[tokenization.vocab['[CLS]'], tokenization.vocab['[PAD]'], tokenization.vocab['[SEP]']]], max_length[1],
            value=padding_value, padding='post',
            truncating='post'))
        masks.append([0])
    return content, masks


def get_byte_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[np.array(data, np.int32).tobytes()]))

def get_int_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=np.array(data, np.int64)))


def save_as_record(path, data, vocab_path, max_length, fields):
    path_tag_size = [40,20,100]
    max_ner_size = 128
    events = {'EquityFreeze': 0, 'EquityRepurchase': 1, 'EquityUnderweight': 2, 'EquityOverweight': 3,
              'EquityPledge': 4}
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

    train_writer = tf.python_io.TFRecordWriter(path)
    vocab = tokenization.FullTokenizer(vocab_path)
    for x in tqdm(data):
        # 处理原文
        sentences, sentences_mask = process(x['sentences'], vocab, max_length)
        # 处理ner 标签
        # 空白的ner_tag
        ner_tag = np.zeros(max_length, dtype=np.int32)
        flag = 0
        for w in x['ann_mspan2dranges'].keys():
            field = w
            tag = x['ann_mspan2guess_field'][field]
            indexs = x['ann_mspan2dranges'][field]
            for index in indexs:
                if index[2]> max_length[1]-1:
                    flag=1
                    break
                ner_tag[index[0]][index[1]+1] = fields['%s_B' % str(tag).upper()]
                for i in range(index[1] + 1, index[2]):
                    ner_tag[index[0]][i+1] = fields['%s_I' % str(tag).upper()]
        if flag==1:
            continue

        # 生成路径tag
        # 维度分别是路径数，字段数，候选值
        path_tag = np.zeros(path_tag_size, dtype=np.int32)+(-1)

        # 存储完整路径
        path_entity_list = np.zeros(path_tag_size[:2], dtype=np.int32) + (-1)
        path_event_type= np.zeros([path_tag_size[0]], dtype=np.int32)+(-1)
        event_tag = [0 for i in range(5)]

        # 辅助数据：实体index
        ners = [k for k in x['ann_mspan2dranges'].keys()]
        ner_index = []
        ner_list_index = [0]
        for k in ners:
            ner_index.extend(x['ann_mspan2dranges'][k])
            ner_index[-1][1] += 1
            ner_index[-1][2] += 1
            ner_list_index.append(ner_list_index[-1]+len(x['ann_mspan2dranges'][k]))

        for i in range(max_ner_size-len(ner_index)):
            ner_index.append([-1,-1,-1])

        for i in range(max_ner_size - len(ner_list_index)):
            ner_list_index.append(-1)

        event_tree = {}
        for e in x['recguid_eventname_eventdict_list']:
            # 处理事件tag
            event_tag[events[e[1]]] = 1
            if events[e[1]] not in event_tree.keys():
                event_tree[events[e[1]]]={}
            et = event_tree[events[e[1]]]

            # 合并相同前缀
            for f in events_fields[e[1]]:
                value = e[2][f]
                if value== None:
                    value = 'NA'
                if value not in et.keys():
                    et[value] = {}
                et = et[value]

        # 创建路径tag
        path_type = np.zeros([10], dtype=np.int32)+(-1)
        paths =[]
        for index, k in enumerate(event_tree.keys()):
            start = len(paths)
            paths.extend(get_path(event_tree[k]))
            path_type[start:len(paths)] = k
            for index2,path in enumerate(paths[start:]):
                cache = event_tree[k][path[0]]
                # 跳过第一个节点
                if ners.index(path[0]) != 'NA':
                    path_entity_list[start + index2, 0] = ners.index(path[0])+1
                else:
                    path_entity_list[start + index2, 0] = 0
                for i, p in enumerate(path[1:]):
                    cache = cache[p]
                    tag = np.array([0 if f not in cache.keys() else 1 for f in ners], dtype=np.int32)
                    tag = np.concatenate([tag, np.zeros([path_tag_size[-1] - tag.size], dtype=np.int32)], axis=0)
                    path_tag[start+index2,i+1,:] = tag
                    if p!='NA':
                        path_entity_list[start+index2,i+1] = ners.index(p)+1
                    else:
                        path_entity_list[start + index2, i + 1] = 0
                path_event_type[start + index2] = k
            tag = np.array([0 if f not in [c[0] for c in paths] else 1 for f in ners], dtype=np.int32)
            tag = np.concatenate([tag, np.zeros([path_tag_size[-1] - tag.size], dtype=np.int32)], axis=0)
            path_tag[start:len(paths),0,:] = tag

        if len(ner_list_index) != max_ner_size:
            continue

        if len(ner_index) != max_ner_size:
            continue

        # # test
        # def select_path(path_tag, path_num, path_event_type, path_entity_list):
        #     path_index = np.random.randint(0, path_num[0], size=1, dtype=np.int32)[0]
        #     return path_tag[path_index], path_index, path_event_type[path_index], path_entity_list[path_index]
        #
        # path_tag, path_index, path_event_type, path_entity_list = select_path(path_tag, [len(paths)], path_event_type,
        #                                                                       path_entity_list)
        #
        # # 去除padding的ner_index
        # def select_nert_index(path_entity_list):
        #     size2 = path_entity_list.argmin(axis=0)
        #     return path_entity_list[:size2]
        #
        # path_entity_list = select_nert_index(path_entity_list)

        features = tf.train.Features(feature={
            'sentences': get_byte_feature(sentences), # 原始文本
            'sentences_mask': get_byte_feature(sentences_mask), # 原始文本长度
            'event_tag': get_byte_feature(event_tag), # 事件标签
            'ner_tag': get_byte_feature(ner_tag), # 实体标签
            'path_tag':get_byte_feature(path_tag), # 路径标签
            'ner_list_index': get_byte_feature(ner_list_index), #
            'ner_index': get_byte_feature(ner_index), #
            'path_event_type': get_int_feature(path_event_type),
            'path_num': get_int_feature([len(paths)]),
            'path_entity_list': get_byte_feature(path_entity_list)
            # 'mask1': tf.train.Feature(int64_list=tf.train.Int64List(value=mask1)),
        })

        example = tf.train.Example(features=features)
        train_writer.write(example.SerializeToString())
    train_writer.close()


def extend_data(data):
    """
    进行数据增强，
    :param data:
    :return:
    """
    rt = []
    for x in data:
        rt.append(x)
        # 乱序
        # 消失字符
        # 改为多音字


if __name__ == '__main__':
    # train_data = read_json('./Data/train.json')
    # save_json(train_data[:10], './Data/sample.json')

    train_data = read_json('./Data/sample.json')
    max_length = [64, 256]
    ner_tag = ['O']
    fields = []

    save_data = []
    for x in train_data:
        record = {}
        # 原文
        record['sentences'] = x[1]['sentences']
        # 实体tag
        record['ann_mspan2dranges'] = x[1]['ann_mspan2dranges']
        record['ann_mspan2guess_field'] = x[1]['ann_mspan2guess_field']
        for w in record['ann_mspan2guess_field'].keys():
            field = record['ann_mspan2guess_field'][w]
            if field not in fields:
                fields.append(field)
                ner_tag.extend([s % str(field).upper() for s in ['%s_B', '%s_I']])
        # 事件tag以及无环图tag
        record['recguid_eventname_eventdict_list'] = x[1]['recguid_eventname_eventdict_list']
        save_data.append(record)
    random.shuffle(save_data)
    cache = ner_tag
    ner_tag = {}
    for index, x in enumerate(cache):
        ner_tag[x] = index

    # save_json(ner_tag, './tfrecord/ner_tag.json')
    ner_tag = read_json('./tfrecord/ner_tag.json')
    save_as_record('./tfrecord/sample_train.record', save_data, './albert/config/vocab.txt', max_length, ner_tag)
