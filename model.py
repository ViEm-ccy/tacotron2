from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from layers import ConvNorm, LinearNorm
from utils import to_gpu, get_mask_from_lengths


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        padding = int((attention_kernel_size - 1) / 2)
        self.location_conv = ConvNorm(2, attention_n_filters,
                                      kernel_size=attention_kernel_size,
                                      padding=padding, bias=False, stride=1,
                                      dilation=1)
        self.location_dense = LinearNorm(attention_n_filters, attention_dim,
                                         bias=False, w_init_gain='tanh')

    def forward(self, attention_weights_cat):
        # 卷积：(batch,2,1024)->(batch,32,1024)
        processed_attention = self.location_conv(attention_weights_cat)
        # (batch,32,1024)->(batch,1024,32)
        processed_attention = processed_attention.transpose(1, 2)
        # 线性：(batch,1024,32)->(batch,1024,128)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention  # (batch,1024,128)


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        # 1024->128
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        # 512->128 Vhj
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        # 128->1
        self.v = LinearNorm(attention_dim, 1, bias=False)

        self.location_layer = LocationLayer(attention_location_n_filters,  # 32
                                            attention_location_kernel_size,  # 31
                                            attention_dim)  # 128
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch,1024)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time=text_length)

        RETURNS
        -------
        alignment (batch, max_time=Text_length)
        """
        # (batch,1,1024)->(batch,1,128)  Wsi-1
        processed_query = self.query_layer(query.unsqueeze(1))
        # 卷积提取位置信息 U·fij # (batch,text_length,128)
        processed_attention_weights = self.location_layer(attention_weights_cat)
        # processed_memory = Vhj (batch,Text_length,128)
        # 注意力得分！！！！！！！
        # (batch, max_time=Text_length，128)->(batch, max_time=Text_length，1)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))
        # (batch, max_time=Text_length，1)->(batch, max_time=Text_length)
        energies = energies.squeeze(-1)
        return energies  # (batch, max_time=Text_length)

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output  (batch,1024)
        memory: encoder outputs  (batch,Text_length,512)
        processed_memory: processed encoder outputs  (batch,Text_length,128)
        attention_weights_cat: previous and cummulative attention weights  (batch,2,text_length)
        mask: binary mask for padded data  BYTE(batch,max_length)
        """
        # (batch, Text_length)
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)
            # 在mask值为1的位置处用value填充。
            # mask的元素个数需和本tensor相同，但尺寸可以不同

        attention_weights = F.softmax(alignment, dim=1)
        # 批矩阵乘法（权值*encode输出）
        # (batch,1,Text_length)*(batch,Text_length,512) = (batch,1,512)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        # (batch,1,512)->(batch,512)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights  # (batch,512), (batch, Text_length)


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes): # 80*n [256,256]
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        # 80*n ->256 ->256
        self.layers = nn.ModuleList(
            [LinearNorm(in_size, out_size, bias=False)
             for (in_size, out_size) in zip(in_sizes, sizes)])


    def forward(self, x):
        for linear in self.layers:
            x = F.dropout(F.relu(linear(x)), p=0.5, training=True)
        return x


class Postnet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, hparams):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        # (batch, n*mels, T_out)-> (batch，512, T_out) 但是这里没有n
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.n_mel_channels, hparams.postnet_embedding_dim,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(hparams.postnet_embedding_dim))
        )
        # 多层(batch，512, T_out)-> (batch，512, T_out)
        for i in range(1, hparams.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(hparams.postnet_embedding_dim,
                             hparams.postnet_embedding_dim,
                             kernel_size=hparams.postnet_kernel_size, stride=1,
                             padding=int((hparams.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(hparams.postnet_embedding_dim))
            )
        # (batch，512, T_out)-> (batch, n*mels, T_out)
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(hparams.postnet_embedding_dim, hparams.n_mel_channels,
                         kernel_size=hparams.postnet_kernel_size, stride=1,
                         padding=int((hparams.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(hparams.n_mel_channels))
            )

    def forward(self, x):
        # (B, n*mels, T_out)
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)
        # (B, n*mels, T_out)
        return x


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, hparams):
        super(Encoder, self).__init__()

        convolutions = []
        # 多层卷积 (batch,512,text_length)->(batch,512,text_length)
        for _ in range(hparams.encoder_n_convolutions):
            conv_layer = nn.Sequential(
                ConvNorm(hparams.encoder_embedding_dim,
                         hparams.encoder_embedding_dim,
                         kernel_size=hparams.encoder_kernel_size, stride=1,
                         padding=int((hparams.encoder_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hparams.encoder_embedding_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        # 512 -> 512*0.5*2(双向LSTM）
        self.lstm = nn.LSTM(hparams.encoder_embedding_dim,
                            int(hparams.encoder_embedding_dim / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        # (batch,512,text_length) (batch)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        # (batch,512,text_length) -> (batch,text_length,512)
        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        # 这里的pack，理解成压紧比较好。
        # 将一个 填充过的变长序列 压紧。
        # （填充时候，会有冗余，所以压紧一下）
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        # (batch,text_length,512)->(batch,text_length,512*0.5*2)
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)
        # (batch,text_length,512)
        return outputs

    def inference(self, x):
        # (batch,512,text_length)
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)
        # (batch,512,text_length) -> (batch,text_length,512)
        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        # (batch,text_length,512)->(batch,text_length,512*0.5*2)
        outputs, _ = self.lstm(x)
        # (batch,text_length,512)
        return outputs


class Decoder(nn.Module):
    def __init__(self, hparams):
        super(Decoder, self).__init__()
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        self.encoder_embedding_dim = hparams.encoder_embedding_dim
        self.attention_rnn_dim = hparams.attention_rnn_dim
        self.decoder_rnn_dim = hparams.decoder_rnn_dim
        self.prenet_dim = hparams.prenet_dim
        self.max_decoder_steps = hparams.max_decoder_steps
        self.gate_threshold = hparams.gate_threshold
        self.p_attention_dropout = hparams.p_attention_dropout
        self.p_decoder_dropout = hparams.p_decoder_dropout

        self.prenet = Prenet(
            hparams.n_mel_channels * hparams.n_frames_per_step,  # 80*n
            [hparams.prenet_dim, hparams.prenet_dim])  # [256,256]

        # 256+512->1024
        self.attention_rnn = nn.LSTMCell(
            hparams.prenet_dim + hparams.encoder_embedding_dim,  # 256+512
            hparams.attention_rnn_dim)  # 1024

        self.attention_layer = Attention(hparams.attention_rnn_dim,   # 1024
                                         hparams.encoder_embedding_dim,   # 512
                                         hparams.attention_dim,    # 128
                                         hparams.attention_location_n_filters,   # 32
                                         hparams.attention_location_kernel_size)   # 31
        # 1024+512->1024
        self.decoder_rnn = nn.LSTMCell(
            hparams.attention_rnn_dim + hparams.encoder_embedding_dim,  # 1024+512
            hparams.decoder_rnn_dim, 1)  # 1024 （单层）
        # 1024+512->n*mels
        self.linear_projection = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim,  # 1024+512
            hparams.n_mel_channels * hparams.n_frames_per_step)  # n*mels
        # 1024+512->1
        self.gate_layer = LinearNorm(
            hparams.decoder_rnn_dim + hparams.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs (batch,Text_length,512)

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)  # batch
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input  # (batch,n*mels)

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs (batch,Text_length,512)
        mask: Mask for padded data if training, expects None for inference :BYTE(batch,max_len)
        """
        B = memory.size(0)  # batch
        MAX_TIME = memory.size(1)  # Text_length
        
        # 初始化attention的LSTM的隐藏层和单元状态为0值
        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())  # (batch, 1024)
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())  # (batch, 1024)
        # 初始化decoder的LSTM的隐藏层和单元状态为0值
        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())  # (batch, 1024)
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())  # (batch, 1024)

        #
        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())  # (batch, text_length)
        # 楼上的累加
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())  # (batch, text_length)
        # 上下文向量
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())  # (batch, 512)

        self.memory = memory  # (batch,Text_length,512)
        # (batch,Text_length,512)->(batch,Text_length,128)
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask  # BYTE(batch,max_len)

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs
        (batch,mels,frames)
        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        #  把N帧mels压缩到一次输出
        decoder_inputs = decoder_inputs.view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out, n_mel_channels) -> (T_out, B, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs: list(batch, n*mels)
        gate_outputs: gate output energies list(batch)
        alignments: list(batch, Text_length)

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out, B , Text_length) -> (B, T_out,Text_length)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out, B) -> (B, T_out)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        gate_outputs = gate_outputs.contiguous()
        # (T_out, B, n_mel_channels) -> (B, T_out, n_mel_channels)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step 把连续输出的帧拆开
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)
        # (B, n_mel_channels, T_out),(B, T_out),(B, T_out,Text_length)
        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output # (batch，256)

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        # 将input和attention上下文向量拼接起来
        cell_input = torch.cat((decoder_input, self.attention_context), -1)  # (batch，256+512)
        # h=(batch，1024) c=(batch，1024)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))  # in=(batch，1024) h0=c0=(batch，1024)
        # 将attention 的 h 做dropout处理，概率p_attention_dropout
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        # 2 的来源
        # (batch,2,text_length)
        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        # (batch,512), (batch, Text_length)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        # 累加
        self.attention_weights_cum += self.attention_weights
        # 拼接第一层LSTM的输出和上下文向量 (batch，1024+512)
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        # 第二层LSTM
        # (batch, 1024) , (batch, 1024)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        # 拼接第二层LSTM的输出和上下文向量 (batch，1024+512)
        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        # (batch，n*mels)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)
        # (batch，1)
        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        # 返回这一步的输出、终止预测和attention权值
        # (batch，n*mels) (batch，1) (batch, Text_length)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs (batch,Text_length,512)
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs ( batch, n_mel_channels,T_out)
        memory_lengths: Encoder output lengths for attention masking.(batch,1)

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        # 初始化一个全零帧作为decoder中LSTM的第一个输入
        decoder_input = self.get_go_frame(memory).unsqueeze(0)  # (1,batch,n*mels)
        #  把N帧mels压缩到一次输出后的decode输入
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs)  # (T_out, batch, n_mel_channels)
        # 拼接初始帧和解码输入
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0)

        decoder_inputs = self.prenet(decoder_inputs)  # (T_out, batch, 256)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        # 循环条件：输出列表元素个数<输出次数（-1是因为之前拼接了初始帧）
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            # 取出第i个input
            decoder_input = decoder_inputs[len(mel_outputs)]  # (batch，256)
            # 过decode
            # (batch，n*mels) (batch，1) (batch, Text_length)
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            # 列表APPEND
            mel_outputs += [mel_output.squeeze(1)]  # list(batch，n*mels)
            gate_outputs += [gate_output.squeeze()]  # list（batch）
            alignments += [attention_weights]  # list(batch, Text_length)

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)
        # (B, n_mel_channels, T_out),(B, T_out),(B, T_out,Text_length)
        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs # (batch,text_length,512)

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)  # (frames,n*mels)

        self.initialize_decoder_states(memory, mask=None)

        mel_outputs, gate_outputs, alignments = [], [], []
        while True:
            decoder_input = self.prenet(decoder_input)
            mel_output, gate_output, alignment = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]
            alignments += [alignment]

            if torch.sigmoid(gate_output.data) > self.gate_threshold:
                break
            elif len(mel_outputs) == self.max_decoder_steps:
                print("Warning! Reached max decoder steps")
                break
            # 上一次的输出作为下一次的输入
            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


class Tacotron2(nn.Module):
    def __init__(self, hparams):
        super(Tacotron2, self).__init__()
        self.mask_padding = hparams.mask_padding  # ======
        self.fp16_run = hparams.fp16_run  # 是否16位运行
        self.n_mel_channels = hparams.n_mel_channels
        self.n_frames_per_step = hparams.n_frames_per_step
        # 字符编号->512维向量
        self.embedding = nn.Embedding(
            hparams.n_symbols, hparams.symbols_embedding_dim)
        std = sqrt(2.0 / (hparams.n_symbols + hparams.symbols_embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(hparams)
        self.decoder = Decoder(hparams)
        self.postnet = Postnet(hparams)

    def parse_batch(self, batch):
        text_padded, input_lengths, mel_padded, gate_padded, \
            output_lengths = batch
        text_padded = to_gpu(text_padded).long()
        input_lengths = to_gpu(input_lengths).long()
        max_len = torch.max(input_lengths.data).item()
        mel_padded = to_gpu(mel_padded).float()
        gate_padded = to_gpu(gate_padded).float()
        output_lengths = to_gpu(output_lengths).long()

        return (
            (text_padded, input_lengths, mel_padded, max_len, output_lengths),
            (mel_padded, gate_padded))

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths)
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            # (n*mels,batch,T_out)->(batch,n*mels,T_out)
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies (batch,1,T_out)作掩码

        return outputs

    def forward(self, inputs):
        text_inputs, text_lengths, mels, max_len, output_lengths = inputs
        text_lengths, output_lengths = text_lengths.data, output_lengths.data

        # (batch,text_length,512)->(batch,512,text_length)
        embedded_inputs = self.embedding(text_inputs).transpose(1, 2)

        # (batch,text_length,512)
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)

        # 这一步训练时候要输入encoder的输出和对应音频的mel
        # 但是预测的时候只需要encoder的输出
        # (B, n*mels, T_out),(B, T_out),(B, T_out,Text_length)
        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels, memory_lengths=text_lengths)
        # postnet 平滑处理？
        # (B, n*mels, T_out)
        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments],
            # [(batch，n*mels, T_out), (batch，n*mels, T_out),(B, T_out),(B, T_out,Text_length)]
            output_lengths)  # (batch)

    def inference(self, inputs):
        # (batch,text_length,512)->(batch,512,text_length)
        embedded_inputs = self.embedding(inputs).transpose(1, 2)
        # (batch,text_length,512)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        outputs = self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])

        return outputs
