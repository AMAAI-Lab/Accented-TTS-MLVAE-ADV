import argparse
import os

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.model import get_model, get_vocoder
from utils.tools import to_device, log, synth_one_sample, plot_embedding
from model import Tacotron2Loss, DLoss
from dataset import Dataset
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate(model, step, configs, mel_stats, logger=None, vocoder=None, len_losses=3):
    preprocess_config, model_config, train_config = configs
    out_dir = train_config["path"]["plot_path"]
    array_path = train_config["path"]["array_path"]

    colors = 'k','r','b','g','y','c','m'
    labels = preprocess_config["accents"]
    # colors2 = 'g','r','r','c','g','b','b','m','b','c','m','b','y','g','m','c','g','r','y','m','c','r','y','y'
    colors2 = 'r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y','k','c','r','b','g','y'
    # labels2 = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "PNV", "LXC", "HKK", "ASI", "THV", "MBMPS", "SVBI", "ZHAA", "HJK", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]
    labels2 = ["RRBI", "ABA", "SKA", "EBVS", "TNI", "NCC", "BWC", "HQTV", "TXHC", "ERMS", "CLB", "PNV", "BDL", "LXC", "HKK", "ASI", "THV", "MBMPS", "SLT", "SVBI", "ZHAA", "HJK", "RMS", "TLV", "NJS", "YBAA", "YDCK", "YKWK"]

    # Get dataset
    dataset = Dataset(
        "val.txt", preprocess_config, model_config, train_config, sort=True, drop_last=False
    )
    batch_size = train_config["optimizer"]["batch_size"]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
    )
    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]

    # Get loss function
    LossG = Tacotron2Loss(preprocess_config, model_config, train_config).to(device) # new
    LossD = DLoss(preprocess_config, model_config, train_config).to(device) # NEW
    # Evaluation
    loss_sums = [0 for _ in range(len_losses)]
    # inf_loss_sums = [0 for _ in range(len_losses)]
    accent_id_list=[]
    spk_id_list=[]
    acc_emb_list=[]
    acc_emb_sg_list=[]
    spk_emb_list=[]
    for batchs in loader:
        for batch in batchs:
            batch = to_device(batch, device, mel_stats if normalize else None)
            with torch.no_grad():
                # Forward
                output = model(*(batch[2:]))

                lossesG = LossG(batch, output, step)
                smallbatch=(batch[6],batch[12],batch[2])

                (z_acc, z_spk, z_acc_sg, mlvae_stats) = model.module.MLVAEencoder(smallbatch[0], acc_labels=smallbatch[1], spk_labels=smallbatch[2], input_lengths=None)
                acc_prob = model.module.adv_classifiers(z_spk)

                # D loss
                total_lossD, acc_ce_loss = LossD(smallbatch, acc_prob, step)

                for i in range(len(lossesG)):
                    loss_sums[i] += lossesG[i].item() * len(batch[0])
                loss_sums[7] += acc_ce_loss.item() * len(batch[0])

                acc_emb = output[4][0]
                spk_emb = output[4][2]
                acc_emb_sg = output[4][4]
                acc_emb_list.append(acc_emb.cpu().detach())
                acc_emb_sg_list.append(acc_emb_sg.cpu().detach())
                spk_emb_list.append(spk_emb.cpu().detach())

                accent_id_list.append(batch[12].cpu().detach())#check
                spk_id_list.append(batch[2].cpu().detach())
                # Inference

                # output_inference = model.inference(*batch[2:5], *batch[6:9])





                # Cal Loss
                # inf_loss = Loss(batch, output_inference, step)



                # for i in range(len(inf_losses)):
                #     inf_loss_sums[i] += inf_losses[i].item() * len(batch[0])

    embedding_acc=np.zeros((560,acc_emb_list[0].size()[1]))
    embedding_acc_sg=np.zeros((560,acc_emb_sg_list[0].size()[1]))
    embedding_spk=np.zeros((560,spk_emb_list[0].size()[1]))

    embedding_accent_id=np.zeros((560))
    embedding_spk_id=np.zeros((560))

    xi=0
    for bat in acc_emb_list:
        for ii in bat:
            embedding_acc[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in acc_emb_sg_list:
        for ii in bat:
            embedding_acc_sg[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in spk_emb_list:
        for ii in bat:
            embedding_spk[xi,:] = np.array(ii)
            xi+=1

    xi=0
    for bat in accent_id_list:
        for ii in bat:
            embedding_accent_id[xi] = np.array(ii)
            xi+=1

    xi=0
    for bat in spk_id_list:
        for ii in bat:
            embedding_spk_id[xi] = np.array(ii)
            xi+=1    

    np.save(os.path.join(array_path,'acc_mu.npy'),embedding_acc)
    np.save(os.path.join(array_path,'acc_mu_sg.npy'),embedding_acc_sg)
    np.save(os.path.join(array_path,'spk_mu.npy'),embedding_spk)
    np.save(os.path.join(array_path,'acc_id.npy'),embedding_accent_id)
    np.save(os.path.join(array_path,'spk_id.npy'),embedding_spk_id)

    plot_embedding(out_dir, embedding_acc, embedding_accent_id,colors,labels,filename=str(step)+'embeddingacc.png')
    plot_embedding(out_dir, embedding_acc_sg, embedding_accent_id,colors,labels,filename=str(step)+'embeddingaccsg.png')

    plot_embedding(out_dir, embedding_spk, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingspk.png')

    plot_embedding(out_dir, embedding_acc, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingacc_spklabels.png')
    plot_embedding(out_dir, embedding_acc_sg, embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingacc_sg_spklabels.png')
    plot_embedding(out_dir, embedding_spk, embedding_accent_id,colors,labels,filename=str(step)+'embeddingspk_acclabels.png')

    plot_embedding(out_dir, np.concatenate((embedding_acc,embedding_spk),1), embedding_spk_id,colors2,labels2,filename=str(step)+'embeddingcombined_spklabels.png')
    plot_embedding(out_dir, np.concatenate((embedding_acc,embedding_spk),1), embedding_accent_id,colors,labels,filename=str(step)+'embeddingcombined_acclabels.png')
    # s1=batch[2]
    # s2=batch[3]
    output_inference = model.module.inference(batch[2][0].unsqueeze(0), batch[3][0].unsqueeze(0), batch[6][0].unsqueeze(0), batch[5], batch[11], batch[12][0].unsqueeze(0))

    # output_inference = model.module.inference(*batch[2:4], batch[6], batch[5], *batch[11:])

    loss_means = [loss_sum / len(dataset) for loss_sum in loss_sums]
    # inf_loss_means = [loss_sum / len(dataset) for loss_sum in inf_loss_sums]


    message = "Validation Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}, Guided Attention Loss: {:.4f}, Acc KL Loss: {:.4f}, Spk KL Loss: {:.4f}, Acc_ADV Loss: {:.4f}, Acc_CE Loss: {:.4f}".format(
        *([step] + [l for l in loss_means])
    )
    # message = "Inference at Step {}, Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}, Guided Attention Loss: {:.4f}, Encoder Loss: {:.4f} ".format(
    #     *([step] + [l for l in inf_loss_means])
    # )
    if logger is not None:
        fig, gate_fig, wav_reconstruction, wav_prediction, wav_inference, tag = synth_one_sample(
            batch,
            output,
            output_inference,
            vocoder,
            mel_stats,
            model_config,
            preprocess_config,
            step,
        )

        log(logger, step, losses=loss_means[:7], lossesD=loss_means[7:])
        log(
            logger,
            fig=fig,
            tag="Validation/step_{}_{}".format(step, tag),
        )
        log(
            logger,
            step=step,
            fig=gate_fig,
            tag="Gates/validation",
        )
        sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
        log(
            logger,
            audio=wav_reconstruction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_reconstructed".format(step, tag),
        )
        log(
            logger,
            audio=wav_prediction,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_synthesized".format(step, tag),
        )
        log(
            logger,
            audio=wav_inference,
            sampling_rate=sampling_rate,
            tag="Validation/step_{}_{}_inferred".format(step, tag),
        )

    return message
