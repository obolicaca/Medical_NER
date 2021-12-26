""" 为模型准备输入的数据集类 """
import os
from torch.utils.data import Dataset, DataLoader


class DataSet(Dataset):
    def __init__(self, file_name, train=True):
        file_path = os.path.join('./dataset',file_name )      # 添加数据集路径
        if train == True:
            with open(os.path.join(file_path, 'train.tsv'), 'r',encoding='utf-8') as f:
                data = f.readlines()
        else:
            with open(os.path.join(file_path, 'test.tsv'), 'r',encoding='utf-8') as f:
                data = f.readlines()

        # 将数据集中每一行的内容分成word和tag
        self.content = []
        words = []
        tags = []
        for line in data:
            line_content = [i.strip().upper() for i in line.split()]
            if line_content == []:
                temp = list(zip(words, tags))
                self.content.append(temp)
                words = []
                tags = []
            else:
                words.append(line_content[0])
                tags.append(line_content[-1])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, item):
        contents, target = zip(*self.content[item])
        return list(contents), list(target)


def collate_fn(batch):
    sentences, tags = zip(*batch)
    return sentences, tags


def DataLoad(file_name,train=True):
    if train == True:
        batch_size = 32
    else:
        batch_size = 64
    dataset = DataSet(file_name,train)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return dataloader

if __name__ == '__main__':
    dataset = DataLoad('CCKS2017',True)
    for index, (sentences, tags) in enumerate(dataset):
        print(index)
        print(sentences)
        print(tags)
        break


