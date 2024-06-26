import os
import json
# import re
import argparse
# from string import punctuation

import torch
import numpy as np
from torch.utils.data import DataLoader
# from g2p_en import G2p
# from pypinyin import pinyin, Style

from utils.model import get_model, get_vocoder
from utils.tools import get_configs_of, to_device, infer_one_sample, infer_one_sample_no_figure, plot_embedding #, read_lexicon
from dataset import TextDataset, Dataset
from text import text_to_sequence, sequence_to_text
from utils.tools import pad_2D

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess_english(text, preprocess_config):
    sequence = text_to_sequence(
        text, preprocess_config["preprocessing"]["text"]["text_cleaners"]
    )
    print("Raw Text Sequence: {}".format(text))
    print("Sequence: {}".format(" ".join([str(id) for id in sequence_to_text(sequence)])))
    print("Sequence Input: {}".format(" ".join([str(id) for id in sequence])))
    return np.array(sequence)

def synthesize_sample_stats(model, args, configs, mel_stats, vocoder, batchs, sig_acc, sig_spk):
    preprocess_config, model_config, train_config = configs
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    arraypath = train_config["path"]["array_path"]
    for batch in batchs:

        # max_target_len = batch[4].shape[1]
        # r_len_pad = max_target_len % n_frames_per_step
        # if r_len_pad != 0:
        #     max_target_len += n_frames_per_step - r_len_pad
        #     assert max_target_len % n_frames_per_step == 0
            
        # ss = pad_2D(batch[4], max_target_len)
        batch=to_device(batch, device, mel_stats)
        # batch=to_device((*batch[0:4], ss, *batch[5:]), device, mel_stats)
        # accents2 = torch.LongTensor(accents2).to(device)

        # if flat_acc:
        #     std_acc = sig_acc*torch.ones((1,model_config["accent_encoder"]["z_dim"])).to(device)
        # else:
        #     std_acc = sig_acc*torch.randn((1,model_config["accent_encoder"]["z_dim"])).to(device)

        # if flat_spk:           
        #     std_spk = sig_spk*torch.ones((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        # else:
        #     std_spk = sig_spk*torch.randn((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        
        acc_mu=np.load(os.path.join(arraypath,'inf_acc_mu.npy'))
        acc_var=np.load(os.path.join(arraypath,'inf_acc_var.npy'))

        spk_mu=np.load(os.path.join(arraypath,'inf_spk_mu.npy'))
        spk_var=np.load(os.path.join(arraypath,'inf_spk_var.npy'))

        acc_id=np.load(os.path.join(arraypath,'inf_acc_id.npy'))
        spk_id=np.load(os.path.join(arraypath,'inf_spk_id.npy'))
        
        # z_acc=np.mean(acc_mu[spk_id==batch[2][0].cpu().item()],axis=0)
        z_acc=np.mean(acc_mu[acc_id==batch[8][0].cpu().item()],axis=0)
        z_spk=np.mean(spk_mu[spk_id==batch[2][0].cpu().item()],axis=0)

        # z_acc=acc_mu[acc_id==batch[8][0].cpu().item()]
        # z_acc=z_acc[np.random.randint(80),:]
        # z_spk=np.mean(spk_mu[spk_id==batch[2][0].cpu().item()],axis=0)
        accents2=None

        z_acc=torch.from_numpy(z_acc).unsqueeze(0).to(device)
        z_spk=torch.from_numpy(z_spk).unsqueeze(0).to(device)


        with torch.no_grad():
            #forward
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], batch[4].size(1), accents=batch[-1])
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], torch.tensor(max_target_len).reshape(-1).to(device), accents=batch[-1])
            model.eval()
            output = model.inference_sampling(*batch[2:5], *batch[6:9], accents2, z_acc, z_spk, args)
            # infer_one_sample(
            #         batch,
            #         output,
            #         vocoder,
            #         mel_stats,
            #         model_config,
            #         preprocess_config,
            #         train_config["path"]["result_path"],
            #         args,
            #             )
            infer_one_sample_no_figure(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                        )

def synthesize_sample(model, args, configs, mel_stats, vocoder, batchs, sig_acc, sig_spk, flat_acc, flat_spk, accents2):
    preprocess_config, model_config, train_config = configs
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    
    for batch in batchs:

        # max_target_len = batch[4].shape[1]
        # r_len_pad = max_target_len % n_frames_per_step
        # if r_len_pad != 0:
        #     max_target_len += n_frames_per_step - r_len_pad
        #     assert max_target_len % n_frames_per_step == 0
            
        # ss = pad_2D(batch[4], max_target_len)
        batch=to_device(batch, device, mel_stats)
        # batch=to_device((*batch[0:4], ss, *batch[5:]), device, mel_stats)
        mu_acc = torch.zeros((1,model_config["accent_encoder"]["z_dim"])).to(device)
        mu_spk = torch.zeros((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        accents2 = torch.LongTensor(accents2).to(device)

        if flat_acc:
            std_acc = sig_acc*torch.ones((1,model_config["accent_encoder"]["z_dim"])).to(device)
        else:
            std_acc = sig_acc*torch.randn((1,model_config["accent_encoder"]["z_dim"])).to(device)

        if flat_spk:           
            std_spk = sig_spk*torch.ones((1,model_config["speaker_encoder"]["z_dim"])).to(device)
        else:
            std_spk = sig_spk*torch.randn((1,model_config["speaker_encoder"]["z_dim"])).to(device)

        z_acc=mu_acc+std_acc
        z_spk=mu_spk+std_spk

        with torch.no_grad():
            #forward
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], batch[4].size(1), accents=batch[-1])
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], torch.tensor(max_target_len).reshape(-1).to(device), accents=batch[-1])
            model.eval()
            output = model.inference_sampling(*batch[2:5], *batch[6:9], accents2, z_acc, z_spk, args)
            infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                        )



def synthesize_single(model, args, configs, mel_stats, vocoder, batchs):
    preprocess_config, model_config, train_config = configs
    n_frames_per_step = model_config["decoder"]["n_frames_per_step"]
    
    for batch in batchs:
        # max_target_len = batch[4].shape[1]
        # r_len_pad = max_target_len % n_frames_per_step
        # if r_len_pad != 0:
        #     max_target_len += n_frames_per_step - r_len_pad
        #     assert max_target_len % n_frames_per_step == 0
            
        # ss = pad_2D(batch[4], max_target_len)
        batch=to_device(batch, device, mel_stats)
        # batch=to_device((*batch[0:4], ss, *batch[5:]), device, mel_stats)
        with torch.no_grad():
            #forward
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], batch[4].size(1), accents=batch[-1])
            # output = model(*batch[2:4], batch[5], batch[5], batch[4], torch.tensor(max_target_len).reshape(-1).to(device), accents=batch[-1])
            model.eval()
            output = model.inference(*batch[2:5], *batch[6:9])
            infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                    )
            


def synthesize_batch(model, args, configs, mel_stats, vocoder, loader):
    preprocess_config, model_config, train_config = configs
    out_dir ='output/plots'
    embedding = []
    colors = 'r','b','g','y'
    labels = preprocess_config["accents"]
    embedding_accent_id = []

    for batchs in loader:
        for batch in batchs:
            batch= to_device(batch, device, mel_stats)
            with torch.no_grad():
                ids,raw_texts, speakers, texts, text_lens,max_text_lens, mels,mel_lens,max_target_len,r_len_pad,gates,spker_embeds,accents = batch
                batch  = (ids, raw_texts, speakers, texts, mels,text_lens, max_text_lens, spker_embeds, accents)
                #forward
                model.eval()
                output = model.inference(*batch[2:5], *batch[6:9])
                infer_one_sample(
                    batch,
                    output,
                    vocoder,
                    mel_stats,
                    model_config,
                    preprocess_config,
                    train_config["path"]["result_path"],
                    args,
                )
                prob_ = output[4]
                embedding.append(prob_[1].squeeze(0).cpu().detach())
                embedding_accent_id.append(batch[8].cpu().detach())
            
    
    embedding = np.array([np.array(xi) for xi in embedding])
    embedding_accent_id = np.array([np.array(id_[0]) for id_ in embedding_accent_id])
    plot_embedding(out_dir, embedding, embedding_accent_id,colors,labels,filename='embedding.png')






if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.dataset='L2CMU'
    # args.source='val.txt'
    args.source=None
    args.mode='sample_stats'
    args.restore_step=200000

    spklist=["SLT","SVBI","HKK","NCC","THV","ABA","EBVS"]
    acclist=["American","Arabic", "Chinese", "Hindi", "Korean", "Spanish", "Vietnamese"]
    fulltxtlist=["For the twentieth time that evening the two men shook hands",
    "Will we ever forget it",
    "And you always want to see it in the superlative degree",
    "I came for information more out of curiosity than anything else",
    "What was the object of your little sensation",
    "But what they want with your toothbrush is more than I can imagine",
    "I graduated last of my class",
    "He will knock you off a few sticks in no time",
    "How old are you daddy",
    "I will go over tomorrow afternoon"]

    fullnamelist=["arctic_a0003","arctic_a0005","arctic_a0007","arctic_a0058","arctic_a0071","arctic_a0285","arctic_a0304","arctic_a0334","arctic_a0379","arctic_a0390"]

    # indexlist=[0,1,3]
    indexlist=[0,1,2,3,4,5,6,7,8,9]
    txtlist=[fulltxtlist[i] for i in indexlist]
    namelist=[fullnamelist[i] for i in indexlist]

    # Read Config
    preprocess_config, model_config, train_config = get_configs_of(args.dataset)
    configs = (preprocess_config, model_config, train_config)
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        mel_stats = stats["mel"]
    os.makedirs(
        os.path.join(train_config["path"]["result_path"], str(args.restore_step)), exist_ok=True)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)
    i=0

    for spk in spklist:
        for txt in txtlist:
            for acc in acclist:
                    

                print(i)
                i+=1
                args.speaker_id=spk
                args.accent=acc
                args.text=txt
                args.siga=0.001
                args.sigs=-0.001
                args.flata=True
                args.flats=True


                if args.mode == "sample_stats":
                    ids = raw_texts = [args.text[:100]]
                    sig_acc = args.siga
                    sig_spk = args.sigs
                    flat_acc = args.flata
                    flat_spk = args.flats

                    # Speaker Info
                    load_spker_embed = model_config["multi_speaker"] \
                        and preprocess_config["preprocessing"]["speaker_embedder"] != 'none'
                    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "speakers.json")) as f:
                        speaker_map = json.load(f)
                    speakers = np.array([speaker_map[args.speaker_id]]) if model_config["multi_speaker"] else np.array([0]) # single speaker is allocated 0
                    spker_embed = np.load(os.path.join(
                        preprocess_config["path"]["preprocessed_path"],
                        "spker_embed",
                        "{}-spker_embed.npy".format(args.speaker_id),
                    )) if load_spker_embed else None

                    if preprocess_config["preprocessing"]["text"]["language"] == "en":
                        texts = np.array([preprocess_english(args.text, preprocess_config)])
                    else:
                        raise NotImplementedError
                    acc_name=args.accent
                    # acc_name2=args.accent2

                    text_lens = np.array([len(texts[0])])


                    with open(os.path.join(preprocess_config["path"]["preprocessed_path"], "accents.json")) as f:
                        accent_map = json.load(f)
                        
                    accents_to_indices = dict()
                    
                    for _idx, acc in enumerate(preprocess_config['accents']):
                        accents_to_indices[acc] = _idx
                    
                    mel=np.zeros((1,1,1))
                    accents = np.array([accents_to_indices[acc_name]])
                    # accents2 = np.array([accents_to_indices[acc_name2]])
                    loader = [(ids, raw_texts, speakers, texts, mel, text_lens, max(text_lens), spker_embed, accents)]
                    
                    synthesize_sample_stats(model, args, configs, mel_stats, vocoder, loader, sig_acc, sig_spk)
