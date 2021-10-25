import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, input_dim, args):
        super().__init__()
        self.n_layers = args.n_layers

        self.rnn = nn.GRU(
            input_dim, args.hidden_dim, args.n_layers, dropout=args.dropout
        )
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inp_seq):
        inp_seq = inp_seq.permute(1, 0, 2)
        outputs, hidden = self.rnn(inp_seq)

        return outputs, hidden


class BahdanauAttention(nn.Module):
    def __init__(self, args):
        super(BahdanauAttention, self).__init__()
        self.dec_output_dim = args.hidden_dim
        self.units = args.hidden_dim
        self.W1 = nn.Linear(self.dec_output_dim, self.units)
        self.W2 = nn.Linear(self.dec_output_dim, self.units)
        self.V = nn.Linear(self.dec_output_dim, 1)

    def forward(self, hidden, enc_output):
        query_with_time_axis = hidden.unsqueeze(1)

        score = self.V(torch.tanh(self.W1(query_with_time_axis) + self.W2(enc_output)))

        attention_weights = torch.softmax(score, axis=1)

        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector, attention_weights


class Decoder(nn.Module):
    def __init__(self, attention, args):
        super().__init__()
        self.dec_feature_size = args.target_n
        self.encoder_hidden_dim = args.hidden_dim
        self.output_dim = args.target_n
        self.decoder_hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.attention = attention

        self.layer = nn.Linear(self.dec_feature_size, args.hidden_dim)
        self.rnn = nn.GRU(
            self.encoder_hidden_dim * 2,
            self.decoder_hidden_dim,
            self.n_layers,
            dropout=args.dropout,
        )
        self.fc_out = nn.Linear(args.hidden_dim, self.output_dim)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, enc_output, dec_input, hidden):
        dec_input = self.layer(dec_input)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        dec_input = torch.cat([torch.sum(context_vector, dim=0), dec_input], dim=1)
        dec_input = dec_input.unsqueeze(0)

        output, hidden = self.rnn(dec_input, hidden)

        prediction = self.fc_out(output.sum(0))

        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, attention):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, encoder_input, decoder_input, teacher_forcing=False):
        batch_size = decoder_input.size(0)
        trg_len = decoder_input.size(1)

        outputs = torch.zeros(batch_size, trg_len - 1, self.decoder.output_dim).to(
            torch.device("cuda:0")
        )
        enc_output, hidden = self.encoder(encoder_input)

        dec_input = decoder_input[:, 0]
        for t in range(1, trg_len):
            output, hidden = self.decoder(enc_output, dec_input, hidden)
            outputs[:, t - 1] = output
            if teacher_forcing == True:
                dec_input = decoder_input[:, t]
            else:
                dec_input = output

        return outputs
