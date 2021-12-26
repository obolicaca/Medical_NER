from dataload import DataLoad
import os
from tqdm import tqdm
import torch
import pickle
from transformers import XLNetTokenizer

class Transform():
    """建立词表，将words和tags转换为ids"""
    # 定义填充词与未知词的词语
    UNK_TAG= 'UNK'      # UNK_TAG代表未知词语
    PAD_TAG = 'PAD'     # PAD_TAG代表未知词语
    CLS_TAG = '[CLS]'   # CLS_TAG代表句首单词
    SEP_TAG = '[SEP]'   # SEP_TAG代表句尾单词
    UNK = 0             # 未知词语代表数字0
    PAD = 1             # 填充词语代表数字1
    CLS = 2         # 句首单词代表数字2
    SEP = 3         # 句尾单词代表数字3

    def __init__(self):
        self.word_dict = {self.UNK_TAG: self.UNK, self.PAD_TAG : self.PAD, self.CLS_TAG: self.CLS, self.SEP_TAG:self.SEP}  # 创建由words映射到数字的字典
        self.tags_dict = {}                                                 # 创建由tags映射到数字的字典
        self.len_tag = None                                                 # 用于记录tags_dict的长度

    def __len__(self):
        return len(self.word_dict)

    def build_dict(self,sentences,targets):
        for sentence in sentences:
            # 建立由词语映射到数字的字典
            for word in sentence:
                if word not in self.word_dict:
                    self.word_dict[word] = len(self.word_dict)

        for tags in targets:
            for tag in tags:
                if tag not in self.tags_dict:
                    self.tags_dict[tag] = len(self.tags_dict)
        self.len_tag = len(self.tags_dict)                     # 保存预测类别的数量

    def words_to_ids(self,sentence,max_len = None):
        # 将一条语句转换为数字序列
        if max_len != None:
            if len(sentence) < max_len:         # 若限制长度超过语句长度，则进行填充
                sentence = sentence + [self.PAD_TAG] * (max_len - len(sentence))
            else:
                sentence = sentence[:max_len]
        words_inputs =  [self.word_dict.get(word,self.UNK) for word in sentence]             # 将词语转换为数字，若词语不存在则将未知词语映射为0
        return words_inputs

    # 将一条语句中的标签转换为数字序列
    def tags_to_ids(self, tags, max_len = None):
        if max_len != None:
            if len(tags) < max_len:         # 若限制长度超过语句长度，则进行填充
                tags = tags + ['O'] * (max_len - len(tags))
            else:
                tags = tags[:max_len]
        tags_inputs = [self.tags_dict.get(tag,self.tags_dict['O']) for tag in tags]
        return tags_inputs

    def reverse(self,ids):
        # 建立从数字映射到词语的字典
        self.id_to_word_dict = dict(zip(self.word_dict.values(), self.word_dict.keys()))
        self.id_to_tag_dict = dict(zip(self.tags_dict.values(), self.tags_dict.keys()))
        # 进行转化
        id_to_word = [self.id_to_word_dict.get(id,self.UNK_TAG) for id in ids]
        id_to_tag = [self.id_to_tag_dict[id] for id in ids]
        return id_to_word,id_to_tag

class Encode():
    def __init__(self,file_name,train = True):
        self.file_name = file_name                                              # 确定选择训练的数据集
        self.train = train
        self.transform = Transform()
        self.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
        self.max_len = 300
        self.words_inputs = []
        self.tags_inputs = []
        self.save_name = os.path.join('./model', self.file_name + '_dict.pkl')      # 定义保存字典的名称

        # 若词表存在则加载词表，不存在则重新创建词表
        if os.path.exists(self.save_name):
            self.transform = pickle.load(open(self.save_name, "rb"))
        else:
            self.build_vocab()
        self.num_labels = self.transform.len_tag                                    # self.num_labels为预测类别的数量

    def build_vocab(self) :
        data = DataLoad(self.file_name, self.train)             # 根据file_name选择进行训练的数据集
        bar = tqdm(enumerate(data), desc="创建词表", total=len(data))
        for index, (sentences, targets) in bar:
            self.transform.build_dict(sentences, targets)
        pickle.dump(self.transform, open(self.save_name, 'wb'))           # 语料字典的保存

    def embedding_sentences(self,sentences):
        # words_temps = [self.tokenizer.encode(sentence, add_special_tokens=False) for sentence in sentences]         # 将词转化为id

        self.words_inputs = []
        for sentence in sentences:
            words_temp = self.transform.words_to_ids(sentence)
            self.words_inputs.append([self.transform.word_dict[self.transform.CLS_TAG]] + self.save_max_len(words_temp) + [self.transform.word_dict[self.transform.SEP_TAG]])
        # self.words_inputs = [[self.tokenizer.cls_token_id] + self.save_max_len(words_temp) + [self.tokenizer.sep_token_id] for words_temp in words_temps]
        return self.output([self.pad_seq(words_input,"sentences") for words_input in self.words_inputs])

    def embedding_targets(self,targets):
        # 将tags转换为ids
        self.tags_inputs = []
        for target in targets:
            tags_temp = self.transform.tags_to_ids(target)
            self.tags_inputs.append([self.transform.tags_dict['O']] + self.save_max_len(tags_temp) + [self.transform.tags_dict['O']])
        return self.output([self.pad_seq(tags_input,"targets") for tags_input in self.tags_inputs])

    def attention_mask(self,input_ids):
        mask_temps = [[1 for i in range(self.max_len if len(temp)>self.max_len else len(temp))] for temp in input_ids ]
        return self.output([self.pad_seq(mask,"mask") for mask in mask_temps])

    def save_max_len(self,x):
        # 调整额外的[CLS] 和 [SEP] 标签, 因此需要调整长度为max_len - 2
        if self.max_len >= len(x):
            return x[:len(x) - 2]
        else:
            return x[:self.max_len - 2]

    def pad_seq(self, x,types):
        # 将sentences,targets,mask分别填充至最大长度
        if types == "sentences":
            return x + [self.transform.PAD for i in range(self.max_len - len(x))]
        elif types == "targets":
            return x + [self.transform.tags_dict['O'] for i in range(self.max_len - len(x))]
        else:
            return x + [0 for i in range(self.max_len - len(x))]

    def output(self, x: list) -> torch.tensor:
        return torch.tensor(x, dtype=torch.long)

if __name__ == '__main__':
    encoder = Encode('CCKS2017')
    encoder.build_vocab()