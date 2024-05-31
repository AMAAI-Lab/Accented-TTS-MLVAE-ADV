import os
import json
from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LinearNorm
from .modules import Encoder, Decoder, Postnet, MLVAEencoder, ADVclassifiers
from utils.tools import get_mask_from_lengths
from text.symbols import symbols


class Tacotron2(nn.Module):
    """ Tacotron2 """

    def __init__(self, preprocess_config, model_config, train_config):
        super(Tacotron2, self).__init__()
        self.model_config = model_config
        n_symbols = len(symbols) + 1

        self.mask_padding = train_config["optimizer"]["mask_padding"]
        self.fp16_run = train_config["optimizer"]["fp16_run"]
        self.n_mel_channels = preprocess_config["preprocessing"]["mel"]["n_mel_channels"]
        self.embedding = nn.Embedding(
            n_symbols, model_config["encoder"]["symbols_embedding_dim"])
        std = sqrt(2.0 / (n_symbols  + model_config["encoder"]["symbols_embedding_dim"]))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(model_config)
        self.decoder = Decoder(preprocess_config, model_config)
        self.postnet = Postnet(preprocess_config, model_config)
        self.lin_proj = LinearNorm(model_config["lin_proj"]["in_dim"], model_config["lin_proj"]["out_dim"])

        self.speaker_emb = None
        self.accent_emb = None

        self.accent_encoder_type = model_config["accent_encoder"]["encoder_type"]
        self.MLVAEencoder = MLVAEencoder(model_config)
        self.adv_classifiers = ADVclassifiers(model_config)

    def parse_output(self, outputs, output_lengths=None):
        if self.mask_padding and output_lengths is not None:
            mask = ~get_mask_from_lengths(output_lengths, outputs[0].size(2))
            mask = mask.expand(self.n_mel_channels, mask.size(0), mask.size(1))
            mask = mask.permute(1, 0, 2)

            outputs[0].data.masked_fill_(mask, 0.0)
            outputs[1].data.masked_fill_(mask, 0.0)
            outputs[2].data.masked_fill_(mask[:, 0, :], 1e3)  # gate energies

        outputs[0] = outputs[0].transpose(-2, -1)
        outputs[1] = outputs[1].transpose(-2, -1)
        return outputs

    def forward(
        self,
        speakers,
        texts,
        src_lens,
        max_src_len,
        mels,
        mel_lens,
        r_len_pad=None,
        max_mel_len=None,
        gates=None,
        spker_embeds=None,
        accents=None,
    ):

        # process text
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder(embedded_inputs, src_lens)


        if self.accent_encoder_type is not None:
            assert accents is not None, "Accent labels should not be None"
            # (z_acc, z_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
            (z_acc, z_spk, z_acc_sg, mlvae_stats) = self.MLVAEencoder(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)

            # encoder_outputs = encoder_outputs + accent_embedding.unsqueeze(1).expand(
            #     -1, max_src_len, -1
            # )

        #freeze every layer of classifiers
        #can freeze the whole group? or with torch.no_grad()?
        # freeze in train script
        # self.adv_classifiers.acc_classifier.parameters

        #run the classifiers yolo

        acc_prob = self.adv_classifiers(z_spk)



        # vae_outs=torch.cat([z_acc,z_spk,y_acc,y_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)
        vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)

        # encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,z_acc,z_spk,y_acc,y_spk],axis=1))
        encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,vae_outs],axis=2))


        mel_outputs, gate_outputs, alignments = self.decoder(
            encoder_outputs, mels.transpose(-2, -1), memory_lengths=src_lens)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet


        # #D step now
        # # maybe no need to run this again? just disable the gradient somehow? (would be freezing the encoder I guess)
        # # but we want the freezes to happen before loss is calculated and optimizers used?
        # with torch.no_grad():
        #     (z_acc, z_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
        

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mlvae_stats, acc_prob],
            mel_lens)

    def inference(
        self,
        speakers,
        texts,
        mels,
        max_src_len=None,
        spker_embeds=None,
        accents=None,
    ):
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        # if self.speaker_emb is not None:
        #     if self.embedder_type == "none":
        #         encoder_outputs = encoder_outputs + self.speaker_emb(speakers).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )
        #     else:
        #         assert spker_embeds is not None, "Speaker embedding should not be None"
        #         encoder_outputs = encoder_outputs + self.speaker_emb(spker_embeds).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )
        #     if self.accent_encoder_type is not None:
        #         assert accent_labels is not None, "Accent labels should not be None"
        #         (accent_embedding, a_prob) = self.accent_encoder.inference(mels,labels=accent_labels, input_lengths=None)
        #         encoder_outputs = encoder_outputs + accent_embedding.unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )

        # if self.accent_encoder_type is not None:
        #     assert accents is not None, "Accent labels should not be None"
        #     (z_acc, y_acc, z_spk, y_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder.inference(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
        #     encoder_outputs = encoder_outputs + accent_embedding.unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

        if self.accent_encoder_type is not None:
            assert accents is not None, "Accent labels should not be None"
            # (z_acc, z_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder.inference(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
            (z_acc, z_spk, z_acc_sg, mlvae_stats) = self.MLVAEencoder.inference(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)

        # if self.accent_encoder_type is not None:
        #     assert accents is not None, "Accent labels should not be None"
        #     # (z_acc, y_acc, z_spk, y_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
        #     (z_acc, z_spk, a_prob) = self.accent_encoder.inference(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
                        
        acc_prob = self.adv_classifiers(z_spk)

        # vae_outs=torch.cat([z_acc,z_spk,y_acc,y_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)
        vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)

        # encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,z_acc,z_spk,y_acc,y_spk],axis=1))
        encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,vae_outs],axis=2))


        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments, mlvae_stats, acc_prob])


    def inference_sampling(
        self,
        speakers,
        texts,
        mels,
        max_src_len=None,
        spker_embeds=None,
        accents=None,
        accents2=None,
        z_acc=None,
        z_spk=None,
        args=None,
    ):
        embedded_inputs = self.embedding(texts).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        # if self.speaker_emb is not None:
        #     if self.embedder_type == "none":
        #         encoder_outputs = encoder_outputs + self.speaker_emb(speakers).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )
        #     else:
        #         assert spker_embeds is not None, "Speaker embedding should not be None"
        #         encoder_outputs = encoder_outputs + self.speaker_emb(spker_embeds).unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )
        #     if self.accent_encoder_type is not None:
        #         assert accent_labels is not None, "Accent labels should not be None"
        #         (accent_embedding, a_prob) = self.accent_encoder.inference(mels,labels=accent_labels, input_lengths=None)
        #         encoder_outputs = encoder_outputs + accent_embedding.unsqueeze(1).expand(
        #             -1, max_src_len, -1
        #         )

        # if self.accent_encoder_type is not None:
        #     assert accents is not None, "Accent labels should not be None"
        #     (z_acc, y_acc, z_spk, y_spk, (mu_acc, var_acc, mu_spk, var_spk)) = self.accent_encoder.inference(mels, acc_labels=accents, spk_labels=speakers, input_lengths=None)
        #     encoder_outputs = encoder_outputs + accent_embedding.unsqueeze(1).expand(
        #         -1, max_src_len, -1
        #     )

        vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)
        # vae_outs=torch.cat([z_acc,z_spk],axis=1).unsqueeze(1).expand(-1,max_src_len,-1)

        # encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,z_acc,z_spk,y_acc,y_spk],axis=1))
        encoder_outputs = self.lin_proj(torch.cat([encoder_outputs,vae_outs],axis=2))


        mel_outputs, gate_outputs, alignments = self.decoder.inference(
            encoder_outputs)

        mel_outputs_postnet = self.postnet(mel_outputs)
        mel_outputs_postnet = mel_outputs + mel_outputs_postnet

        return self.parse_output(
            [mel_outputs, mel_outputs_postnet, gate_outputs, alignments])