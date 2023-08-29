import math
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

# 自制数据集
sentences = [['我 是 学 生 P', 'S I am a student', 'I am a student E'],         # S: 开始字符
             ['我 喜 欢 学 习', 'S I like learning P', 'I like learning P E'],  # E: 结束字符 
             ['我 是 男 生 P',  'S I am a boy',        'I am a boy E']]         # P: 占位符号，如果当前句子不足固定长度用P占位 pad补0

src_vocab = {'P': 0, '我': 1, '是': 2, '学': 3, '生': 4, '喜': 5, '欢': 6, '习': 7, '男': 8} # 词源字典，字：索引
src_idx2word = {src_vocab[key]: key for key in src_vocab}
src_vocab_size = len(src_vocab)

tgt_vocab = {'S': 0, 'E': 1, 'P': 2, 'I': 3, 'am': 4, 'a': 5, 'student': 6, 'like': 7, 'learning': 8, 'boy': 9}
tgt_idx2word = {tgt_vocab[key]: key for key in tgt_vocab}
tgt_vocab_size = len(tgt_vocab)

src_len = len(sentences[0][0].split(" "))
tgt_len = len(sentences[0][1].split(" "))

print("输入句子长度：", src_len)
print("输出句子长度：", tgt_len)

print(len(sentences))

# 把sentences转换成字典索引
def gen_sentences_idx_dict(sentences):
    enc_inputs, dec_inputs, dec_outputs = [], [], []
    for i in range(len(sentences)):
        enc_input = [[src_vocab[n] for n in sentences[i][0].split()]]
        dec_input = [[tgt_vocab[n] for n in sentences[i][1].split()]]
        dec_output = [[tgt_vocab[n] for n in sentences[i][2].split()]]
        enc_inputs.extend(enc_input)
        dec_inputs.extend(dec_input)
        dec_outputs.extend(dec_output)
    return torch.LongTensor(enc_inputs), torch.LongTensor(dec_inputs), torch.LongTensor(dec_outputs)

enc_inputs, dec_intputs, dec_outputs = gen_sentences_idx_dict(sentences)
print(enc_inputs)
print(dec_intputs)
print(dec_outputs)

# 自定义数据集类
class MyDataSet(Data.Dataset):
    def __init__(self, enc_inputs, dec_inputs, dec_outputs):
        super(MyDataSet, self).__init__()
        self.enc_inputs = enc_inputs
        self.dec_inputs = dec_inputs
        self.dec_outputs = dec_outputs

    def __len__(self):
        return self.enc_inputs.shape[0]
    
    def __getitem__(self, idx):
        return self.enc_inputs[idx], self.dec_inputs[idx], self.dec_outputs[idx]
    
loader = Data.DataLoader(MyDataSet(enc_inputs, dec_intputs, dec_outputs), 2, True)

# 模型参数设置
d_model = 512           # 字Embedding的维度
d_ff = 2048             # 前向传播隐藏层维度
d_q = d_k = k_v = 64    # K(=Q), V的维度
n_layers = 6            # 有多少个encoder和decoder
n_heads = 8             # Multi-Head Attention设置为8

# 位置编码Embedding
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pos_table = np.array([
            [pos / np.power(10000, 2 * i / d_model) for i in range(d_model)]
            if pos != 0 else np.zeros(d_model) for pos in range(max_len)])
        pos_table[1:, 0::2] = np.sin(pos_table[1:, 0::2])    # 字嵌入维度为偶数时
        pos_table[1:, 1::2] = np.cos(pos_table[1:, 1::2])    # 字嵌入维度为奇数时
        self.pos_table = torch.FloatTensor(pos_table).cuda() # enc_inputs: [seq_len, d_model]
    def forward(self, enc_inputs):                           # enc_inputs: [batch_size, seq_len, d_model]
        enc_inputs += self.pos_table[:enc_inputs.size(1), :]
        return self.dropout(enc_inputs.cuda())

# mask掉停用词
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, len_q = seq_q.size()                                # seq_q用于升维，为了做attention，mask score矩阵用的
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)                   # 判断输入那些含有P(=0)，用1标记，[batch_size, 1, len_k]
    return pad_attn_mask.expand(batch_size, len_q, len_k)           # 扩展成多维度 [batch_size, len_q, len_k]

def get_attn_subsequence_mask(seq):                                 # seq: [batch_size, tgt_len]
    attn_shape = [seq.size(0), seq.size(1), seq.size(1)]            # 生成上三角矩阵，[batch_size, tgt_len, tgt_len]
    subsequence_mask = np.triu(np.ones(attn_shape), k=1)
    subsequence_mask = torch.from_numpy(subsequence_mask).byte()    # [batch_size, tgt_len, tgt_len]
    return subsequence_mask

# 点积注意力
class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    # Q: [batch_size, n_heads, len_q, d_k]
    # K: [batch_size, n_heads, len_k, d_k]
    # V: [batch_size, n_heads, len_v(=len_k), d_v]
    # attn_mask: [batch_size, n_heads, seq_len, seq_len]
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)# scores: [batch_size, n_heads, len_q, len_k]
        scores.masked_fill_(attn_mask, -1e9)                        # 如果是停用词P就等于0
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)                             # [batch_size, n_heads, len_q, d_v]
        return context, attn
    

# 多头注意力
class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_K = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.W_V = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)

    def forward(self, input_Q, input_K, input_V, attn_mask):
        '''
        input_Q: [batch_size, len_q, d_model]
        input_K: [batch_size, len_k, d_model]
        input_V: [batch_size, len_v(=len_k), d_model]
        attn_mask: [batch_size, seq_len, seq_len]
        '''
        residual, batch_size = input_Q, input_Q.size(0)
        Q = self.W_Q(input_Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # Q: [batch_size, n_heads, len_q, d_k]
        K = self.W_K(input_K).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # K: [batch_size, n_heads, len_k, d_k]
        V = self.W_V(input_V).view(batch_size, -1, n_heads, d_k).transpose(1, 2) # V: [batch_size, n_heads, len_k, d_k]
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)              # attn_mask: [batch_size, n_heads, seq_len, seq_len]
        context, attn = ScaleDotProductAttention()(Q, K, V, attn_mask)           # context: [batch_size, n_heads, len_q, d_v]
                                                                                 # attn: [batch_size, n_heads, len_q, len_k]
        context = context.transpose(1, 2).reshape(batch_size, -1, n_heads * d_v) # context: [batch_size, len_q, n_heads * d_v]
        output = self.fc(context)                                                # [batch_size, len_q, d_model]
        return nn.LayerNorm(d_model).cuda()(output + residual), attn
    
# 前馈神经网络
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False)
        )

    def forward(self, inputs):                                                  # inputs: [batch_size, seq_len, d_model]
        residual = inputs
        output = self.fc(inputs)
        return nn.LayerNorm(d_model).cuda()(output + residual)                  # [batch_size, seq_len, d_model]
    
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()                               # 多头注意力机制
        self.pos_ffn = PoswiseFeedForwardNet()                                  # 前馈神经网络

    def forward(self, enc_inputs, enc_self_attn_mask):                              # enc_inputs: [batch_size, src_len, d_model]
        # 输入3个enc_inputs分别于W_q, W_k, W_v相乘得到Q/K/V                            # enc_self_attn_mask: [batch_size, src_len, src_len]
        enc_outputs, attn = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs,  # enc_outputs: [batch_size, src_len, d_model]
                                               enc_self_attn_mask)                  # attn: [batch_size, n_heads, src_len, src_len]
        enc_outputs = self.pos_ffn(enc_outputs)                                     # enc_outputs: [batch_size, src_len, d_model]
        return enc_outputs, attn

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.src_emb = nn.Embedding(src_vocab_size, d_model)
        self.pos_emb = PositionalEmbedding(d_model)
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
    
    def forward(self, enc_inputs):
        '''
        enc_inputs: [batch_size, src_len]
        '''
        enc_outputs = self.src_emb(enc_inputs)                                      # [batch_size, src_len, d_model]
        enc_outputs = self.pos_emb(enc_outputs.transpose(0, 1)).transpose(0, 1)     # [batch_size, src_len, d_model]
        enc_self_attn_mask = get_attn_pad_mask(enc_inputs, enc_inputs)              # [batch_size, src_len, src_len]
        enc_self_attns = []
        for layer in self.layers:
            # enc_outputs: [batch_size, src_len, d_model], enc_self_attn: [batch_size, n_heads, src_len, src_len]
            enc_outputs, enc_self_attn = layer(enc_outputs, enc_self_attn_mask)
            enc_self_attns.append(enc_self_attn)
        return enc_outputs, enc_self_attns
    
