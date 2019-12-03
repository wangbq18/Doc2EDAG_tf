import json
import datetime
import csv
import xml.dom.minidom as xmldom
from tqdm import tqdm


def read_text(path):
    rt = []
    with open(path, 'r', encoding='utf-8') as f:
        for x in f:
            rt.append(x)
        return rt


def save_text(data,path):
    with open(path, 'w', encoding='utf-8') as f:
        for x in data:
            f.write(x+'\n')


def get_time():
    now_time = datetime.datetime.now()
    return now_time.strftime('%Y%m%d%H%M%S')


def read_json(path):
    with open(path,'r',encoding='utf-8') as f:
        data = json.loads(f.read())
    return data


def save_json(data, path):
    with open(path,'w',encoding='utf-8') as f:
        f.write(json.dumps(data,ensure_ascii=False))


def read_csv(path, delimiter =None):
    rt = []
    with open(path, 'r', encoding='utf-8') as file:
        if delimiter !=None:
            data = csv.reader(file,delimiter=delimiter)
        else:
            data = csv.reader(file)
        birth_header = next(data)  # 读取第一行每一列的标题
        for row in data:  # 将csv 文件中的数据保存到birth_data中
            rt.append(row)
    return rt

def write_to_csv(data, path, delimiter=None):
    with open(path, 'w', encoding='utf-8', newline='') as csvfile:
        if delimiter!=None:
            writer = csv.writer(csvfile, delimiter = delimiter)
        else:
            writer = csv.writer(csvfile)
        for x in data:
            writer.writerow(x)

def read_xml(path):
    rt = []
    # 得到文档对象
    domobj = xmldom.parse(path)
    # 得到元素对象
    elementobj = domobj.documentElement
    # 获得子标签
    subElementObj = elementobj.getElementsByTagName("Questions")
    for x in tqdm(subElementObj):
        rt.append([])
        # 正例
        value1 = x.getElementsByTagName("EquivalenceQuestions")[0].getElementsByTagName("question")
        value1 = [w.firstChild.data for w in value1 if w.firstChild!=None]
        value2 = x.getElementsByTagName("NotEquivalenceQuestions")[0].getElementsByTagName("question")
        value2 = [w.firstChild.data for w in value2 if w.firstChild!=None]
        # 构造正例
        for x in range(len(value1)-1):
            for y in range(x+1, len(value1)):
                rt[-1].append([0, value1[x], value1[y]])

        # 构造反例
        for x in range(len(value1)):
            for y in range(len(value2)):
                rt[-1].append([1, value1[x], value2[y]])
    return rt


def read_xml2(path):
    """
    构建三元组，分别是1与2相似，1与3不相似
    :param path:
    :return:
    """
    rt = []
    # 得到文档对象
    domobj = xmldom.parse(path)
    # 得到元素对象
    elementobj = domobj.documentElement
    # 获得子标签
    subElementObj = elementobj.getElementsByTagName("Questions")
    for x in tqdm(subElementObj):
        # 正例
        value1 = x.getElementsByTagName("EquivalenceQuestions")[0].getElementsByTagName("question")
        value1 = [w.firstChild.data for w in value1 if w.firstChild!=None]
        value2 = x.getElementsByTagName("NotEquivalenceQuestions")[0].getElementsByTagName("question")
        value2 = [w.firstChild.data for w in value2 if w.firstChild!=None]
        # 构造正例
        cache = []
        for x1 in range(len(value1)-1):
            for y in range(x1+1, len(value1)):
                cache.append([0, value1[x1], value1[y]])

        # 构造反例
        cache1 = []
        for c in cache:
            for y in range(len(value2)):
                cache1.append([c[0], c[1], value2[y]])

        for x in range(len(value1)):
            for y in range(len(value2)):
                rt.append([1, value1[x], value2[y]])
    return rt

if __name__=='__main__':
    import os
    path = os.path.abspath('./data/train_set.xml')
    read_xml(path)