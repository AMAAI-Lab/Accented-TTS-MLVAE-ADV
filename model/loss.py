import math
from torch import nn
import numpy as np
import torch
from torch.nn import functional as F

eps = 1e-8
def frange_cycle_linear(n_iter, start=0.0001, stop=1.0,  n_cycle=4, ratio=0.5):
    L = np.ones(n_iter)*stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule
    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L

def buildupfc(n_iter, Btype='expo', n_stop=25000, n_up=5000, start=-5.0, stop=0.0):
    if Btype=='expo':
        Llow = np.ones(n_up)*(10**start)
        Lhigh = np.ones(n_iter-n_stop)*(10**stop)
        Lramp = np.ones(n_stop-n_up)*(10**(np.linspace(start,stop,n_stop-n_up)))
    else:
        Llow = np.ones(n_up)*start
        Lhigh = np.ones(n_iter-n_stop)*stop
        Lramp = np.linspace(start,stop,n_stop-n_up)
    return np.concatenate((Llow,Lramp,Lhigh))

# def expo_buildup(n_iter, n_stop=25000, n_up=5000, start=-5.0, stop=0.0):
#     Llow = np.ones(n_up)*(10**start)
#     Lhigh = np.ones(n_iter-n_stop)*(10**stop)
#     Lramp = np.linspace(start,stop,n_stop-n_up)
#     return np.concatenate((Llow,Lramp,Lhigh))

class Tacotron2Loss(nn.Module):
    """ Tacotron2 Loss """

    def __init__(self, preprocess_config, model_config, train_config):
        super(Tacotron2Loss, self).__init__()
        self.n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
        self.use_guided_attn_loss = train_config["optimizer"]["guided_attn"]
        self.n_accent_classes = model_config["accent_encoder"]["n_classes"]
        self.encoder_type = model_config["accent_encoder"]["encoder_type"]
        self.n_speaker_classes = model_config["speaker_encoder"]["n_classes"]

        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        if self.use_guided_attn_loss:
            self.guided_attn_loss = GuidedAttentionLoss(
                sigma=train_config["optimizer"]["guided_sigma"],
                alpha=train_config["optimizer"]["guided_lambda"],
            )

        # self.start = train_config["cyclic_anneling"]["start"]
        # self.stop = train_config["cyclic_anneling"]["stop"]
        # self.n_cycle = train_config["cyclic_anneling"]["n_cycle"]
        # self.ratio = train_config["cyclic_anneling"]["ratio"]
        self.n_iter = train_config["step"]["total_step"]
        self.n_stopKL = train_config["linbuildkl"]["n_stop"]
        self.n_upKL = train_config["linbuildkl"]["n_up"]
        self.stopKL = train_config["linbuildkl"]["stop"]
        self.startKL = train_config["linbuildkl"]["start"]
        self.KLBtype = train_config["linbuildkl"]["type"]
        self.n_stopadv = train_config["linbuildadv"]["n_stop"]
        self.n_upadv = train_config["linbuildadv"]["n_up"]
        self.stopadv = train_config["linbuildadv"]["stop"]
        self.startadv = train_config["linbuildadv"]["start"]
        self.advBtype = train_config["linbuildadv"]["type"]

        self.acc_kl_coef = train_config["coeffs"]["acc_kl"]
        self.spk_kl_coef = train_config["coeffs"]["spk_kl"]
        self.acc_adv_coef = train_config["coeffs"]["acc_adv"]
        self.spk_adv_coef = train_config["coeffs"]["spk_adv"]
        self.reco_coef = train_config["coeffs"]["reco"]

        # self.L = frange_cycle_linear(self.n_iter,start=self.start,stop=self.stop, n_cycle=self.n_cycle, ratio=self.ratio)
        self.LKL = buildupfc(self.n_iter,self.KLBtype,self.n_stopKL,self.n_upKL,start=self.startKL,stop=self.stopKL)
        self.Ladv = buildupfc(self.n_iter,self.advBtype,self.n_stopadv,self.n_upadv,start=self.startadv,stop=self.stopadv)


    def forward(self, inputs, predictions,step):
        mel_target, input_lengths, output_lengths, r_len_pad, gate_target, encoder_labels \
                                = inputs[6], inputs[4], inputs[7], inputs[9], inputs[10], inputs[12]
        mel_out, mel_out_postnet, gate_out, alignments, mlvae_stats, acc_prob = predictions

        mel_target.requires_grad = False
        gate_target.requires_grad = False
        acc_kl_lambda = self.LKL[step] * self.acc_kl_coef
        spk_kl_lambda = self.LKL[step] * self.spk_kl_coef

        gate_target = gate_target.view(-1, 1)

        gate_out = gate_out.view(-1, 1)
        mel_loss = self.reco_coef * (self.mse_loss(mel_out, mel_target) + \
            self.mse_loss(mel_out_postnet, mel_target))
        gate_loss = self.bce_loss(gate_out, gate_target)
        # expressive_encoder_loss
        encoder_loss, acc_kl_loss, spk_kl_loss = self.get_encoder_loss(encoder_labels, 
            mlvae_stats, 
            self.n_accent_classes, 
            acc_kl_lambda,
            spk_kl_lambda,
            self.encoder_type,
            )

        acc_adv_lambda = self.Ladv[step] * self.acc_adv_coef
        spk_adv_lambda = self.Ladv[step] * self.spk_adv_coef

        acc_adv_loss = self.adversarial_loss(acc_prob, self.n_accent_classes) * acc_adv_lambda

        if self.use_guided_attn_loss:
            attn_loss = self.guided_attn_loss(alignments, input_lengths, \
                                (output_lengths + r_len_pad)//self.n_frames_per_step)
            total_loss = mel_loss + gate_loss + attn_loss + encoder_loss + acc_adv_loss
            return total_loss, mel_loss, gate_loss, attn_loss, acc_kl_loss, spk_kl_loss, acc_adv_loss
        else:
            total_loss = mel_loss + gate_loss + encoder_loss
            return total_loss, mel_loss, gate_loss, torch.tensor([0.], device=mel_target.device)

    def KL_loss(self, mu, var):
        return torch.mean(0.5 * torch.sum(torch.exp(var) + mu ** 2 - 1. - var, 1))

    def indices_to_one_hot(self, data, n_classes):
#        targets = np.array(data).reshape(-1)
#        return torch.from_numpy(np.eye(n_classes)[targets]).cuda()
        targets = data.contiguous().view(-1)
        return torch.nn.functional.one_hot(targets,num_classes=n_classes)
#        return torch.eye(targets, device=targets.device)[n_classes]


    def get_encoder_loss(self, id_, vae_stats, classes_, acc_kl_lambda, spk_kl_lambda, encoder_type):
        cat_target = self.indices_to_one_hot(id_, classes_)

        if (encoder_type == 'gst' or encoder_type == 'x-vector') and cat_lambda != 0.0:
            loss = cat_lambda * (-self.entropy(cat_target, vae_stats) - np.log(0.1))
        elif (encoder_type == 'vae' or encoder_type == 'gst_vae') and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = cat_lambda * (-self.entropy(cat_target, vae_stats[2]) - np.log(0.1)) + \
                   kl_lambda * self.KL_loss(vae_stats[0], vae_stats[1])
        elif (encoder_type == 'gmvae') and (cat_lambda != 0.0 or kl_lambda != 0.0):
            loss = self.gaussian_loss(vae_stats[0], vae_stats[1], vae_stats[2], vae_stats[3], vae_stats[4])*kl_lambda + (-self.entropy(vae_stats[5],cat_target) - np.log(0.1))*cat_lambda
        
        elif (encoder_type == 'mlvae') and (acc_kl_lambda != 0.0 or spk_kl_lambda != 0.0):
            loss_class =  acc_kl_lambda * self.KL_loss(vae_stats[0], vae_stats[1])
            loss_style =  spk_kl_lambda * self.KL_loss(vae_stats[2], vae_stats[3])

            loss = loss_style + loss_class
            return loss, loss_class, loss_style
#            print(f"KL Lambda:{kl_lambda} ---- Cat Lambda: {cat_lambda} ----  Style Loss: {loss_style} ---  Class Loss: {loss_class} ----  CE Loss: {loss_cat}")
        elif (encoder_type == 'cvae') and (cat_lambda != 0.0 or kl_lambda != 0.0):
            # loss = cat_lambda * (-self.entropy(cat_target, vae_stats[2]) - np.log(0.1)) + \
            #        kl_lambda * self.KL_loss(vae_stats[0], vae_stats[1])

            loss = acc_kl_lambda * self.KL_loss(vae_stats[0], vae_stats[1]) + spk_kl_lambda * self.KL_loss(vae_stats[2], vae_stats[3])
    
        else:
            loss = 0.0


        return loss

    def adversarial_loss(self, probs, n_class):
        uni=(torch.ones((probs.size(0),n_class),device=probs.device)/n_class)#.to('cuda')
        # adv_loss = torch.sum((torch.nn.functional.softmax(probs,dim=1)-uni)**2,1)
        adv_loss=self.mse_loss(uni,torch.nn.functional.softmax(probs,dim=1))
        # adv_loss = torch.mean((torch.nn.functional.softmax(probs)-torch.tensor.uniform_(0,n_class))**2)
        return adv_loss




class DLoss(nn.Module):
    """ Classifier CE loss in MLVAE-ADV Taco """

    def __init__(self, preprocess_config, model_config, train_config):
        super(DLoss, self).__init__()
        self.n_accent_classes = model_config["accent_encoder"]["n_classes"]
        self.n_speaker_classes = model_config["speaker_encoder"]["n_classes"]
        self.acc_ce_lambda = model_config["accent_classifier"]["lambda"]
        self.spk_ce_lambda = model_config["speaker_classifier"]["lambda"]

        self.n_iter = train_config["step"]["total_step"]
        self.n_stop = train_config["linbuildce"]["n_stop"]
        self.n_up = train_config["linbuildce"]["n_up"]
        self.stop = train_config["linbuildce"]["stop"]
        self.start = train_config["linbuildce"]["start"]
        self.CEBtype = train_config["linbuildce"]["type"]
        self.acc_ce_coef = train_config["coeffs"]["acc_ce"]
        self.spk_ce_coef = train_config["coeffs"]["spk_ce"]

        self.L = buildupfc(self.n_iter,self.CEBtype,self.n_stop,self.n_up,start=self.start,stop=self.stop)


    def forward(self, inputs, predictions, step):
        # MODIFY INPUTS MAYBE
        # mel_target, input_lengths, output_lengths, r_len_pad, gate_target, encoder_labels \
        #                         = inputs[6], inputs[4], inputs[7], inputs[9], inputs[10], inputs[12]

        mel_target, acc_id, spk_id = inputs[0], inputs[1], inputs[2]
        #predictions, make sure we're getting them
        acc_prob = predictions

        mel_target.requires_grad = False
        acc_ce_coef = self.L[step] * self.acc_ce_coef

        acc_ce_loss = self.cross_entropy(acc_prob, acc_id.view(-1)) * acc_ce_coef

        total_loss = acc_ce_loss
        return total_loss, acc_ce_loss

    def cross_entropy(self,inputs,targets):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(inputs,targets)
        return loss

    def indices_to_one_hot(self, data, n_classes):
        targets = data.contiguous().view(-1)
        return torch.nn.functional.one_hot(targets,num_classes=n_classes)

class GuidedAttentionLoss(nn.Module):
    """Guided attention loss function module.
    See https://github.com/espnet/espnet/blob/e962a3c609ad535cd7fb9649f9f9e9e0a2a27291/espnet/nets/pytorch_backend/e2e_tts_tacotron2.py#L25
    This module calculates the guided attention loss described
    in `Efficiently Trainable Text-to-Speech System Based
    on Deep Convolutional Networks with Guided Attention`_,
    which forces the attention to be diagonal.
    .. _`Efficiently Trainable Text-to-Speech System
        Based on Deep Convolutional Networks with Guided Attention`:
        https://arxiv.org/abs/1710.08969
    """

    def __init__(self, sigma=0.4, alpha=1.0, reset_always=True):
        """Initialize guided attention loss module.
        Args:
            sigma (float, optional): Standard deviation to control
                how close attention to a diagonal.
            alpha (float, optional): Scaling coefficient (lambda).
            reset_always (bool, optional): Whether to always reset masks.
        """
        super(GuidedAttentionLoss, self).__init__()
        self.sigma = sigma
        self.alpha = alpha
        self.reset_always = reset_always
        self.guided_attn_masks = None
        self.masks = None

    def _reset_masks(self):
        self.guided_attn_masks = None
        self.masks = None

    def forward(self, att_ws, ilens, olens):
        """Calculate forward propagation.
        Args:
            att_ws (Tensor): Batch of attention weights (B, T_max_out, T_max_in).
            ilens (LongTensor): Batch of input lenghts (B,).
            olens (LongTensor): Batch of output lenghts (B,).
        Returns:
            Tensor: Guided attention loss value.
        """
        if self.guided_attn_masks is None:
            self.guided_attn_masks = self._make_guided_attention_masks(ilens, olens).to(
                att_ws.device
            )
        if self.masks is None:
            self.masks = self._make_masks(ilens, olens).to(att_ws.device)
        losses = self.guided_attn_masks * att_ws
        loss = torch.mean(losses.masked_select(self.masks))
        if self.reset_always:
            self._reset_masks()
        return self.alpha * loss

    def _make_guided_attention_masks(self, ilens, olens):
        n_batches = len(ilens)
        max_ilen = max(ilens)
        max_olen = max(olens)
        guided_attn_masks = torch.zeros((n_batches, max_olen, max_ilen))
        for idx, (ilen, olen) in enumerate(zip(ilens, olens)):
            guided_attn_masks[idx, :olen, :ilen] = self._make_guided_attention_mask(
                ilen, olen, self.sigma
            )
        return guided_attn_masks

    @staticmethod
    def _make_guided_attention_mask(ilen, olen, sigma):
        """Make guided attention mask.
        Examples:
            >>> guided_attn_mask =_make_guided_attention(5, 5, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([5, 5])
            >>> guided_attn_mask
            tensor([[0.0000, 0.1175, 0.3935, 0.6753, 0.8647],
                    [0.1175, 0.0000, 0.1175, 0.3935, 0.6753],
                    [0.3935, 0.1175, 0.0000, 0.1175, 0.3935],
                    [0.6753, 0.3935, 0.1175, 0.0000, 0.1175],
                    [0.8647, 0.6753, 0.3935, 0.1175, 0.0000]])
            >>> guided_attn_mask =_make_guided_attention(3, 6, 0.4)
            >>> guided_attn_mask.shape
            torch.Size([6, 3])
            >>> guided_attn_mask
            tensor([[0.0000, 0.2934, 0.7506],
                    [0.0831, 0.0831, 0.5422],
                    [0.2934, 0.0000, 0.2934],
                    [0.5422, 0.0831, 0.0831],
                    [0.7506, 0.2934, 0.0000],
                    [0.8858, 0.5422, 0.0831]])
        """
        grid_x, grid_y = torch.meshgrid(torch.arange(olen), torch.arange(ilen))
        grid_x, grid_y = grid_x.float().to(olen.device), grid_y.float().to(ilen.device)
        return 1.0 - torch.exp(
            -((grid_y / ilen - grid_x / olen) ** 2) / (2 * (sigma ** 2))
        )

    def _make_masks(self, ilens, olens):
        """Make masks indicating non-padded part.
        Args:
            ilens (LongTensor or List): Batch of lengths (B,).
            olens (LongTensor or List): Batch of lengths (B,).
        Returns:
            Tensor: Mask tensor indicating non-padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
        Examples:
            >>> ilens, olens = [5, 2], [8, 5]
            >>> _make_mask(ilens, olens)
            tensor([[[1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1],
                     [1, 1, 1, 1, 1]],
                    [[1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [1, 1, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0],
                     [0, 0, 0, 0, 0]]], dtype=torch.uint8)
        """
        in_masks = self.make_non_pad_mask(ilens)  # (B, T_in)
        out_masks = self.make_non_pad_mask(olens)  # (B, T_out)
        return out_masks.unsqueeze(-1) & in_masks.unsqueeze(-2)  # (B, T_out, T_in)


    def make_non_pad_mask(self, lengths, xs=None, length_dim=-1):
        return ~self.make_pad_mask(lengths, xs, length_dim)


    def make_pad_mask(self, lengths, xs=None, length_dim=-1):
        if length_dim == 0:
            raise ValueError("length_dim cannot be 0: {}".format(length_dim))

        if not isinstance(lengths, list):
            lengths = lengths.tolist()
        bs = int(len(lengths))
        if xs is None:
            maxlen = int(max(lengths))
        else:
            maxlen = xs.size(length_dim)

        seq_range = torch.arange(0, maxlen, dtype=torch.int64)
        seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
        seq_length_expand = seq_range_expand.new(lengths).unsqueeze(-1)
        mask = seq_range_expand >= seq_length_expand

        if xs is not None:
            assert xs.size(0) == bs, (xs.size(0), bs)

            if length_dim < 0:
                length_dim = xs.dim() + length_dim
            # ind = (:, None, ..., None, :, , None, ..., None)
            ind = tuple(
                slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
            )
            mask = mask[ind].expand_as(xs).to(xs.device)
        return mask


