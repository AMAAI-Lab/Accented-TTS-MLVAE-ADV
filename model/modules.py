import os
import json
import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.autograd import Variable

from utils.tools import get_mask_from_lengths, pad

from .blocks import (
    LinearNorm,
    ConvNorm,
)
from text.symbols import symbols
from mlvae import MLVAENet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """Encoder module:
        - Three 1-d convolution banks
        - Bidirectional LSTM
    """
    def __init__(self, config):
        super(Encoder, self).__init__()

        convolutions = []
        for _ in range(config["encoder"]["encoder_n_convolutions"]):
            conv_layer = nn.Sequential(
                ConvNorm(config["encoder"]["encoder_embedding_dim"],
                         config["encoder"]["encoder_embedding_dim"],
                         kernel_size=config["encoder"]["encoder_kernel_size"], stride=1,
                         padding=int((config["encoder"]["encoder_kernel_size"] - 1) / 2),
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(config["encoder"]["encoder_embedding_dim"]))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm = nn.LSTM(config["encoder"]["encoder_embedding_dim"],
                            int(config["encoder"]["encoder_embedding_dim"] / 2), 1,
                            batch_first=True, bidirectional=True)

    def forward(self, x, input_lengths):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        # pytorch tensor are not reversible, hence the conversion
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs, batch_first=True)

        return outputs

    def inference(self, x):
        for conv in self.convolutions:
            x = F.dropout(F.relu(conv(x)), 0.5, self.training)

        x = x.transpose(1, 2)

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)

        return outputs


class Decoder(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(Decoder, self).__init__()
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
        #Jan

        if model_config["lin_proj"]:
            self.encoder_embedding_dim = model_config["lin_proj"]["out_dim"]
        else:
            self.encoder_embedding_dim = model_config["encoder"]["encoder_embedding_dim"] + model_config["accent_encoder"]["y_dim"] \
            + model_config["accent_encoder"]["z_dim"] + model_config["speaker_encoder"]["y_dim"] +model_config["speaker_encoder"]["z_dim"]



        self.attention_rnn_dim = model_config["attention"]["attention_rnn_dim"]
        self.decoder_rnn_dim = model_config["decoder"]["decoder_rnn_dim"]
        self.prenet_dim = model_config["decoder"]["prenet_dim"]
        self.max_decoder_steps = model_config["decoder"]["max_decoder_steps"]
        self.gate_threshold = model_config["decoder"]["gate_threshold"]
        self.p_attention_dropout = model_config["decoder"]["p_attention_dropout"]
        self.p_decoder_dropout = model_config["decoder"]["p_decoder_dropout"]
        attention_dim = model_config["attention"]["attention_dim"]
        attention_location_n_filters = model_config["location_layer"]["attention_location_n_filters"]
        attention_location_kernel_size = model_config["location_layer"]["attention_location_kernel_size"]

        self.prenet = Prenet(
            self.n_mel_channels * self.n_frames_per_step,
            [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = nn.LSTMCell(
            self.prenet_dim + self.encoder_embedding_dim,
            self.attention_rnn_dim)

        self.attention_layer = Attention(
            self.attention_rnn_dim, self.encoder_embedding_dim,
            attention_dim, attention_location_n_filters,
            attention_location_kernel_size)

        self.decoder_rnn = nn.LSTMCell(
            self.attention_rnn_dim + self.encoder_embedding_dim,
            self.decoder_rnn_dim, 1)

        self.linear_projection = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim,
            self.n_mel_channels * self.n_frames_per_step)

        self.gate_layer = LinearNorm(
            self.decoder_rnn_dim + self.encoder_embedding_dim, 1,
            bias=True, w_init_gain='sigmoid')

    def get_go_frame(self, memory):
        """ Gets all zeros frames to use as first decoder input
        PARAMS
        ------
        memory: decoder outputs

        RETURNS
        -------
        decoder_input: all zeros frames
        """
        B = memory.size(0)
        decoder_input = Variable(memory.data.new(
            B, self.n_mel_channels * self.n_frames_per_step).zero_())
        return decoder_input

    def initialize_decoder_states(self, memory, mask):
        """ Initializes attention rnn states, decoder rnn states, attention
        weights, attention cumulative weights, attention context, stores memory
        and stores processed memory
        PARAMS
        ------
        memory: Encoder outputs
        mask: Mask for padded data if training, expects None for inference
        """
        B = memory.size(0)
        MAX_TIME = memory.size(1)

        self.attention_hidden = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(
            B, self.attention_rnn_dim).zero_())

        self.decoder_hidden = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(
            B, self.decoder_rnn_dim).zero_())

        self.attention_weights = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_weights_cum = Variable(memory.data.new(
            B, MAX_TIME).zero_())
        self.attention_context = Variable(memory.data.new(
            B, self.encoder_embedding_dim).zero_())

        self.memory = memory
        self.processed_memory = self.attention_layer.memory_layer(memory)
        self.mask = mask

    def parse_decoder_inputs(self, decoder_inputs):
        """ Prepares decoder inputs, i.e. mel outputs
        PARAMS
        ------
        decoder_inputs: inputs used for teacher-forced training, i.e. mel-specs

        RETURNS
        -------
        inputs: processed decoder inputs

        """
        # (B, n_mel_channels, T_out) -> (B, T_out, n_mel_channels)
        decoder_inputs = decoder_inputs.transpose(1, 2)
        # Grouping multiple frames if necessary: (B, n_mel_channels, T_out) -> (B, T_out/r, n_mel_channels*r)
        decoder_inputs = decoder_inputs.contiguous().view(
            decoder_inputs.size(0),
            int(decoder_inputs.size(1)/self.n_frames_per_step), -1)
        # (B, T_out/r, n_mel_channels*r) -> (T_out/r, B, n_mel_channels*r)
        decoder_inputs = decoder_inputs.transpose(0, 1)
        return decoder_inputs

    def parse_decoder_outputs(self, mel_outputs, gate_outputs, alignments):
        """ Prepares decoder outputs for output
        PARAMS
        ------
        mel_outputs:
        gate_outputs: gate output energies
        alignments:

        RETURNS
        -------
        mel_outputs:
        gate_outpust: gate output energies
        alignments:
        """
        # (T_out/r, B) -> (B, T_out/r)
        alignments = torch.stack(alignments).transpose(0, 1)
        # (T_out/r, B) -> (B, T_out/r)
        gate_outputs = torch.stack(gate_outputs).transpose(0, 1)
        # tile gate_outputs to make frames per step.
        B = gate_outputs.size(0)
        gate_outputs = gate_outputs.contiguous().view(-1, 1).repeat(1,self.n_frames_per_step).view(B, -1)

        # (T_out/r, B, n_mel_channels*r) -> (B, T_out/r, n_mel_channels*r)
        mel_outputs = torch.stack(mel_outputs).transpose(0, 1).contiguous()
        # decouple frames per step: (B, T_out/r, n_mel_channels*r) -> (B, T_out, n_mel_channels)
        mel_outputs = mel_outputs.view(
            mel_outputs.size(0), -1, self.n_mel_channels)
        # (B, T_out, n_mel_channels) -> (B, n_mel_channels, T_out)
        mel_outputs = mel_outputs.transpose(1, 2)

        return mel_outputs, gate_outputs, alignments

    def decode(self, decoder_input):
        """ Decoder step using stored states, attention and memory
        PARAMS
        ------
        decoder_input: previous mel output

        RETURNS
        -------
        mel_output:
        gate_output: gate output energies
        attention_weights:
        """
        cell_input = torch.cat((decoder_input, self.attention_context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)

        attention_weights_cat = torch.cat(
            (self.attention_weights.unsqueeze(1),
             self.attention_weights_cum.unsqueeze(1)), dim=1)
        self.attention_context, self.attention_weights = self.attention_layer(
            self.attention_hidden, self.memory, self.processed_memory,
            attention_weights_cat, self.mask)

        self.attention_weights_cum += self.attention_weights
        decoder_input = torch.cat(
            (self.attention_hidden, self.attention_context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            decoder_input, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(
            self.decoder_hidden, self.p_decoder_dropout, self.training)

        decoder_hidden_attention_context = torch.cat(
            (self.decoder_hidden, self.attention_context), dim=1)
        decoder_output = self.linear_projection(
            decoder_hidden_attention_context)

        gate_prediction = self.gate_layer(decoder_hidden_attention_context)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, memory, decoder_inputs, memory_lengths):
        """ Decoder forward pass for training
        PARAMS
        ------
        memory: Encoder outputs
        decoder_inputs: Decoder inputs for teacher forcing. i.e. mel-specs
        memory_lengths: Encoder output lengths for attention masking.

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """

        decoder_input = self.get_go_frame(memory).unsqueeze(0) # (1, B, n_mel_channels)
        decoder_inputs = self.parse_decoder_inputs(decoder_inputs) # (T_out/r, B, n_mel_channels*r)
        decoder_inputs = torch.cat((decoder_input, decoder_inputs), dim=0) # (1+(T_out/r), B, n_mel_channels*r)
        decoder_inputs = self.prenet(decoder_inputs) # (1+(T_out/r), B, prenet_dim)

        self.initialize_decoder_states(
            memory, mask=~get_mask_from_lengths(memory_lengths))

        mel_outputs, gate_outputs, alignments = [], [], []
        while len(mel_outputs) < decoder_inputs.size(0) - 1:
            decoder_input = decoder_inputs[len(mel_outputs)]
            mel_output, gate_output, attention_weights = self.decode(
                decoder_input)
            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output.squeeze(1)]
            alignments += [attention_weights]

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments

    def inference(self, memory):
        """ Decoder inference
        PARAMS
        ------
        memory: Encoder outputs

        RETURNS
        -------
        mel_outputs: mel outputs from the decoder
        gate_outputs: gate outputs from the decoder
        alignments: sequence of attention weights from the decoder
        """
        decoder_input = self.get_go_frame(memory)

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
            elif len(mel_outputs) == self.max_decoder_steps // self.n_frames_per_step:
                print("Warning! Reached max decoder steps")
                break

            decoder_input = mel_output

        mel_outputs, gate_outputs, alignments = self.parse_decoder_outputs(
            mel_outputs, gate_outputs, alignments)

        return mel_outputs, gate_outputs, alignments


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
        processed_attention = self.location_conv(attention_weights_cat)
        processed_attention = processed_attention.transpose(1, 2)
        processed_attention = self.location_dense(processed_attention)
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size):
        super(Attention, self).__init__()
        self.query_layer = LinearNorm(attention_rnn_dim, attention_dim,
                                      bias=False, w_init_gain='tanh')
        self.memory_layer = LinearNorm(embedding_dim, attention_dim, bias=False,
                                       w_init_gain='tanh')
        self.v = LinearNorm(attention_dim, 1, bias=False)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self.score_mask_value = -float("inf")

    def get_alignment_energies(self, query, processed_memory,
                               attention_weights_cat):
        """
        PARAMS
        ------
        query: decoder output (batch, n_mel_channels * n_frames_per_step)
        processed_memory: processed encoder outputs (B, T_in, attention_dim)
        attention_weights_cat: cumulative and prev. att weights (B, 2, max_time)

        RETURNS
        -------
        alignment (batch, max_time)
        """

        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_weights_cat)
        energies = self.v(torch.tanh(
            processed_query + processed_attention_weights + processed_memory))

        energies = energies.squeeze(-1)
        return energies

    def forward(self, attention_hidden_state, memory, processed_memory,
                attention_weights_cat, mask):
        """
        PARAMS
        ------
        attention_hidden_state: attention rnn last output
        memory: encoder outputs
        processed_memory: processed encoder outputs
        attention_weights_cat: previous and cummulative attention weights
        mask: binary mask for padded data
        """
        alignment = self.get_alignment_energies(
            attention_hidden_state, processed_memory, attention_weights_cat)

        if mask is not None:
            alignment.data.masked_fill_(mask, self.score_mask_value)

        attention_weights = F.softmax(alignment, dim=1)
        attention_context = torch.bmm(attention_weights.unsqueeze(1), memory)
        attention_context = attention_context.squeeze(1)

        return attention_context, attention_weights


class Prenet(nn.Module):
    def __init__(self, in_dim, sizes):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
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

    def __init__(self, preprocess_config, model_config):
        super(Postnet, self).__init__()
        n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        postnet_embedding_dim = model_config["postnet"]["postnet_embedding_dim"]
        postnet_kernel_size = model_config["postnet"]["postnet_kernel_size"]
        postnet_n_convolutions = model_config["postnet"]["postnet_n_convolutions"]

        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels, postnet_embedding_dim,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size, stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim, n_mel_channels,
                         kernel_size=postnet_kernel_size, stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x

class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self, model_config):

        super().__init__()
        K = len(model_config["reference_encoder"]["ref_enc_filters"])
        filters = [1] + model_config["reference_encoder"]["ref_enc_filters"]

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=model_config["reference_encoder"]["ref_enc_filters"][i])
             for i in range(K)])

        out_channels = self.calculate_channels(model_config["n_mel_channels"], 3, 2, 1, K)
        self.gru = nn.GRU(input_size=model_config["reference_encoder"]["ref_enc_filters"][-1] * out_channels,
                          hidden_size=model_config["reference_encoder"]["ref_enc_gru_size"],
                          batch_first=True)
        self.n_mel_channels = model_config["n_mel_channels"]
        self.ref_enc_gru_size = model_config["reference_encoder"]["ref_enc_gru_size"]

    def forward(self, inputs, input_lengths=None):
        assert inputs.size(-1) == self.n_mel_channels
        out = inputs.unsqueeze(1)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2)  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]


        if input_lengths is not None:
            # print(input_lengths.cpu().numpy(), 2, len(self.convs))
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = max(input_lengths.round().astype(int), [1])
            # print(input_lengths, 'input lengths')
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, l, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            l = (l - kernel_size + 2 * pad) // stride + 1
        return l


class MLVAEencoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        #x_dim, y_dim=10, z_dim, n_classes
        self.encoder = ReferenceEncoder(model_config)
        self.mlvae = MLVAENet(model_config)


    def forward(self, inputs, acc_labels=None, spk_labels=None, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        (z_acc, z_spk, z_acc_sg, (mu_acc, var_acc, mu_spk, var_spk, mu_acc_sg, var_acc_sg)) = self.mlvae(enc_out, acc_labels)

        return (z_acc, z_spk, z_acc_sg, (mu_acc, var_acc, mu_spk, var_spk, mu_acc_sg, var_acc_sg))

    def inference(self, inputs, acc_labels=None, spk_labels=None, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)

        (z_acc, z_spk, z_acc_sg, (mu_acc, var_acc, mu_spk, var_spk, mu_acc_sg, var_acc_sg)) = self.mlvae.inference(enc_out, acc_labels)
        
        return (z_acc, z_spk, z_acc_sg, (mu_acc, var_acc, mu_spk, var_spk, mu_acc_sg, var_acc_sg))


class ADVclassifiers(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.accent_classifier = nn.Sequential(
                LinearNorm(
                model_config["speaker_encoder"]["z_dim"], 
                32),
                nn.ReLU(),
                LinearNorm(32,
                model_config["accent_encoder"]["n_classes"])
        )

    def forward(self, inputs):
        spk_embs = inputs

        acc_prob = self.accent_classifier(spk_embs) #acc classifier on spk embs, for adv loss

        return (acc_prob)

    def inference(self, inputs):
        spk_embs = inputs

        acc_prob = self.accent_classifier(spk_embs) #acc classifier on spk embs, for adv loss

        return (acc_prob)