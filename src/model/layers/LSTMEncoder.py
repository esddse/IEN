import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LSTMEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, max_length):
        super(LSTMEncoder, self).__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.max_length = max_length

        # bilstmencoder
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=True)
        self.h0, self.c0 = self.init_hidden()

    def init_hidden(self):
        h0 = nn.Parameter(torch.zeros(2, self.batch_size, self.hidden_dim))
        c0 = nn.Parameter(torch.zeros(2, self.batch_size, self.hidden_dim))
        return h0, c0

    def forward(self, text_embeddings, text_lengths):
        text_embeddings = torch.nn.utils.rnn.pack_padded_sequence(text_embeddings, text_lengths, batch_first=True)
        lstm_out, _ = self.lstm(text_embeddings, (self.h0, self.c0))        # [seq_len, batch_size, 2 * hidden_dim]
        lstm_out = torch.nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True, padding_value=0.0, total_length=self.max_length)[0]
        return lstm_out


if __name__ == '__main__':

    encoder = LSTMEncoder(5, 3)
    text = torch.randn(1, 10, 5)
    length = torch.LongTensor([7])
    print("text")
    print(text)
    out = encoder(text, length)
    print("out")
    print(out)
