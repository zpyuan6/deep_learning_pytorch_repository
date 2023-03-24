import torch
import torch.nn as nn


class BiLSTM_Sentiment_Classifier(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, lstm_layers, bidirectional,batch_size, dropout, device):
        super(BiLSTM_Sentiment_Classifier,self).__init__()
        
        self.lstm_layers = lstm_layers
        self.num_directions = 2 if bidirectional else 1
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.batch_size = batch_size
        
        # 一个简单的查找表，用于存储固定字典和大小的嵌入。该模块通常用于存储词嵌入并使用索引检索它们。模块的输入是索引列表，输出是相应的词嵌入。
        # num_embeddings ( int ) – 嵌入字典的大小
        # embedding_dim ( int ) – 每个嵌入向量的大小
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim,
                            num_layers=lstm_layers,
                            dropout=dropout,
                            bidirectional=bidirectional,
                            batch_first=True)

        self.fc = nn.Linear(hidden_dim*self.num_directions, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.device = device
        
    def forward(self, x, hidden):
        self.batch_size = x.size(0)
        ##EMBEDDING LAYER
        embedded = self.embedding(x)
        print(f"embedded_shape: {embedded.shape}")
        #LSTM LAYERS
        out, hidden = self.lstm(embedded, hidden)
        print(f"out: {out.shape}")
        #Extract only the hidden state from the last LSTM cell
        out = out[:,-1,:]
        print(f"out: {out.shape}")
        #FULLY CONNECTED LAYERS
        out = self.fc(out)
        print(f"fc out: {out.shape}")
        out = self.softmax(out)
        print(f"softmax out: {out.shape}")

        return out, hidden

    def init_hidden(self, batch_size):
        #Initialization of the LSTM hidden and cell states
        h0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(self.device)
        c0 = torch.zeros((self.lstm_layers*self.num_directions, batch_size, self.hidden_dim)).detach().to(self.device)
        hidden = (h0, c0)
        return hidden