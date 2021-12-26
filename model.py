import math
import copy             # 用于深度拷贝的copy工具包
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF                        # 导入TorchCRF工具包使用CRF模型
from transformers import XLNetModel,XLNetConfig
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')

class NER(nn.Module):
    """ NER model for tagging """
    def __init__(self,type_num,drop = 0.1):
        super(NER, self).__init__()

        self.drop = drop                          # Dropout的比例为0.1
        self.head = 8                             # head代表多头注意力头数为8
        self.hidden_size = 128                    # BiLSTM隐藏层大小为128
        self.num_layers = 1                       # BiLSTM网络层数为1
        self.type_num = type_num                  # self.type_num 为预测类别数目

        self.dropout = nn.Dropout(self.drop)
        self.config = XLNetConfig.from_pretrained('xlnet-base-cased',is_encoder_decoder= True)         # 导入配置文件
        self.xlnet = XLNetModel.from_pretrained('xlnet-base-cased',config= self.config)                # 使用XlNet基础模型
        self.bilstm = nn.LSTM(input_size = self.xlnet.config.hidden_size,hidden_size= self.hidden_size,num_layers= self.num_layers,batch_first=True,bidirectional= True)
        self.MHA = MultiHeadedAttention(head= self.head ,embedding_dim=self.hidden_size,drop = self.drop)      # 引入多头注意力机制
        self.linear = nn.Linear(self.hidden_size,self.type_num)
        self.crf = CRF(num_tags=self.type_num, batch_first= True)

    def forward(self,input_ids,attention_mask,labels):
        outputs = self.xlnet(input_ids = input_ids, attention_mask = attention_mask)              # XLNetModel模型输出最后一维为self.xlnet.config.hidden_size
        output = self.dropout(outputs.last_hidden_state)                                          # output形状:[batch_size,seq_len,hidden_size]
        batch_size = output.size(0)                                                               # 获取每次batch的batch_size
        h0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(device)             # 初始化状态，如果不初始化，torch默认h0,c0初始值为全0
        c0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(device)

        """outout:[batch_size,seq_len,hiddden_size * 2]
           hn=cn:[self.num_layers * 2,batch_size,hidden_size]"""
        output, (hn, cn) = self.bilstm(output, (h0, c0))                        # hn为最后一个时刻隐藏层
        output = output[:,:,:self.hidden_size] + output[:,:,self.hidden_size:]  # 将两个方向的输出进行拼接由[batch_size,seq_len,hiddden_size * 2]变为[batch_size,seq_len,hiddden_size]
        output = self.MHA(output,output,output)                                 # 使用多头注意力机制
        logits = self.linear(output).to(device)
        attention_mask = attention_mask == 1
        loss = self.crf(emissions = logits,tags = labels,mask = attention_mask,reduction= 'mean')         # crf模型的输出为交叉熵损失,reduction='mean'时输出为每个批次上的平均损失
        return -loss

    def predict(self,input_ids,attention_mask):
        """使用测试集进行预测"""
        outputs = self.xlnet(input_ids=input_ids , attention_mask=attention_mask)
        output = self.dropout(outputs.last_hidden_state)
        batch_size = output.size(0)
        h0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        c0 = torch.rand(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        output, (hn, cn) = self.bilstm(output, (h0, c0))
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        output = self.MHA(output, output, output)
        logits = self.linear(output).to(device)
        attention_mask = attention_mask == 1
        return self.crf.decode(logits,attention_mask)      # 输出预测值

# 实现多头注意力机制的处理
class MultiHeadedAttention(nn.Module):
    def __init__(self, head, embedding_dim, drop=0.1):
        """在类的初始化时, 会传入三个参数，head代表头数，embedding_dim代表词嵌入的维度，
           dropout代表进行dropout操作时置0比率，默认是0.1."""
        super(MultiHeadedAttention, self).__init__()

        # 在函数中，首先使用了一个测试中常用的assert语句，判断h是否能被d_model整除，
        # 这是因为我们之后要给每个头分配等量的词特征.也就是embedding_dim/head个.
        assert embedding_dim % head == 0

        # 得到每个头获得的分割词向量维度d_k
        self.d_k = embedding_dim // head

        # 传入头数h
        self.head = head

        # 然后获得线性层对象，通过nn的Linear实例化，它的内部变换矩阵是embedding_dim x embedding_dim，然后使用clones函数克隆四个，
        # 为什么是四个呢，这是因为在多头注意力中，Q，K，V各需要一个，最后拼接的矩阵还需要一个，因此一共是四个.
        self.linears = self.clones(nn.Linear(embedding_dim, embedding_dim), 4)

        # self.attn为None，它代表最后得到的注意力张量，现在还没有结果所以为None.
        self.attn = None

        # 最后就是一个self.dropout对象，它通过nn中的Dropout实例化而来，置0比率为传进来的参数dropout.
        self.drop = drop
        self.dropout = nn.Dropout(p=self.drop)

    def forward(self, query, key, value, mask=None):
        """前向逻辑函数, 它的输入参数有四个，前三个就是注意力机制需要的Q, K, V，
           最后一个是注意力机制中可能需要的mask掩码张量，默认是None. """

        # 如果存在掩码张量mask
        if mask is not None:
            # 使用unsqueeze拓展维度
            mask = mask.unsqueeze(0)

        # 接着，我们获得一个batch_size的变量，他是query尺寸的第1个数字，代表有多少条样本.
        batch_size = query.size(0)

        # 之后就进入多头处理环节
        # 首先利用zip将输入QKV与三个线性层组到一起，然后使用for循环，将输入QKV分别传到线性层中，
        # 做完线性变换后，开始为每个头分割输入，这里使用view方法对线性变换的结果进行维度重塑，多加了一个维度h，代表头数，
        # 这样就意味着每个头可以获得一部分词特征组成的句子，其中的-1代表自适应维度，
        # 计算机会根据这种变换自动计算这里的值.然后对第二维和第三维进行转置操作，
        # 为了让代表句子长度维度和词向量维度能够相邻，这样注意力机制才能找到词义与句子位置的关系，
        # 从attention函数中可以看到，利用的是原始输入的倒数第一和第二维.这样我们就得到了每个头的输入.
        query, key, value = \
           [model(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2)
            for model, x in zip(self.linears, (query, key, value))]

        # 得到每个头的输入后，接下来就是将他们传入到attention中，
        # 这里直接调用我们之前实现的attention函数.同时也将mask和dropout传入其中.
        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 通过多头注意力计算后，我们就得到了每个头计算结果组成的4维张量，我们需要将其转换为输入的形状以方便后续的计算，
        # 因此这里开始进行第一步处理环节的逆操作，先对第二和第三维进行转置，然后使用contiguous方法，
        # 这个方法的作用就是能够让转置后的张量应用view方法，否则将无法直接使用，
        # 所以，下一步就是使用view重塑形状，变成和输入形状相同.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)

        # 最后使用线性层列表中的最后一个线性层对输入进行线性变换得到最终的多头注意力结构的输出.
        return self.linears[-1](x)

    # 首先需要定义克隆函数, 因为在多头注意力机制的实现中, 用到多个结构相同的线性层.
    # 我们将使用clone函数将他们一同初始化在一个网络层列表对象中. 之后的结构中也会用到该函数.
    def clones(self,module, N):
        """用于生成相同网络层的克隆函数, 它的参数module表示要克隆的目标网络层, N代表需要克隆的数量"""
        # 在函数中, 我们通过for循环对module进行N次深度拷贝, 使其每个module成为独立的层,
        # 然后将其放在nn.ModuleList类型的列表中存放.
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

    def attention(self,query, key, value, mask=None, dropout=None):
        """注意力机制的实现, 输入分别是query, key, value, mask: 掩码张量,
           dropout是nn.Dropout层的实例化对象, 默认为None"""
        # 在函数中, 首先取query的最后一维的大小, 一般情况下就等同于我们的词嵌入维度, 命名为d_k
        d_k = query.size(-1)
        # 按照注意力公式, 将query与key的转置相乘, 这里面key是将最后两个维度进行转置, 再除以缩放系数根号下d_k, 这种计算方法也称为缩放点积注意力计算.
        # 得到注意力得分张量scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        # 接着判断是否使用掩码张量
        if mask is not None:
            # 使用tensor的masked_fill方法, 将掩码张量和scores张量每个位置一一比较, 如果掩码张量处为0
            # 则对应的scores张量用-1e9这个值来替换, 如下演示
            scores = scores.masked_fill(mask == 0, -1e9)

        # 对scores的最后一维进行softmax操作, 使用F.softmax方法, 第一个参数是softmax对象, 第二个是目标维度.
        # 这样获得最终的注意力张量
        p_attn = F.softmax(scores, dim=-1)

        # 之后判断是否使用dropout进行随机置0
        if dropout is not None:
            # 将p_attn传入dropout对象中进行'丢弃'处理
            p_attn = dropout(p_attn)

        # 最后, 根据公式将p_attn与value张量相乘获得最终的query注意力表示, 同时返回注意力张量
        return torch.matmul(p_attn, value), p_attn

if __name__ == "__main__":
    model = NER(4)
    print(model)

