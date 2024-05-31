import argparse
import os
import json

import torch
import yaml
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utils.model import get_model, get_vocoder, get_param_num
from utils.tools import get_configs_of, to_device, log, synth_one_sample
from model import Tacotron2Loss, DLoss
from dataset import Dataset

from evaluate import evaluate
import shutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print("Prepare training ...")

    preprocess_config, model_config, train_config = configs

    # Get dataset
    dataset = Dataset(
        "train.txt", preprocess_config, model_config, train_config, sort=True, drop_last=True
    )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4  # Set this larger than 1 to enable sorting in Dataset
    assert batch_size * group_size < len(dataset)
    loader = DataLoader(
        dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
    )
    # Prepare model
    model, optimizer = get_model(args, configs, device, train=True)
    model = nn.DataParallel(model)
    print(model)
    # exit()
    num_param = get_param_num(model)
    LossG = Tacotron2Loss(preprocess_config, model_config, train_config).to(device) # new
    LossD = DLoss(preprocess_config, model_config, train_config).to(device) # NEW
    print("Number of Tacotron2 Parameters:", num_param)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)

    # save configs
    run_name = os.path.basename(train_config["path"]["log_path"])
    os.makedirs('./output/configs/'+run_name, exist_ok=True)
    config_dir = os.path.join("./config", args.dataset)
    shutil.copy2(os.path.join(config_dir,'train.yaml'),os.path.join('./output/configs/',run_name,'train.yaml'))
    shutil.copy2(os.path.join(config_dir,'model.yaml'),os.path.join('./output/configs/',run_name,'model.yaml'))
    shutil.copy2(os.path.join(config_dir,'preprocess.yaml'),os.path.join('./output/configs/',run_name,'preprocess.yaml'))


    # Training
    step = args.restore_step + 1
    # gstep = True
    epoch = 1
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    normalize = preprocess_config["preprocessing"]["mel"]["normalize"]
    with open(
            os.path.join(preprocess_config["path"]["preprocessed_path"], "stats.json")
    ) as f:
        stats = json.load(f)
        mel_stats = stats["mel"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n = args.restore_step
    outer_bar.update()

    while True:
        inner_bar = tqdm(total=len(loader), desc="Epoch {}".format(epoch), position=1)
        for batchs in loader:
            for batch in batchs:
                batch = to_device(batch, device, mel_stats if normalize else None)

                # 0ids,
                # 1raw_texts,
                # 2speakers,
                # 3texts,
                # 4src_lens,
                # 5max_src_len,
                # 6mels,
                # 7mel_lens,
                # 8max_mel_len,
                # 9r_len_pad,
                # 10gates,
                # 11spker_embeds,
                # 12accents,

                # same batch option
                # run G, backprop it, run D, backprop it
                # optimizer can remain the same for both, given that we zero the gradients every time
                # in G, freeze classifiers, then call model forward...
                # in D, unfreeze classifiers, then call with torch.no_grad() on MLVAE, then run forward
                # of classifiers and backprop it, same optimizer...
                #put a condition inside the forward function
                    # freeze 
                    
                #Gstep
                # model.adv_classifiers.parameters.requires_grad=False
                for m in model.module.adv_classifiers.parameters():
                    m.requires_grad=False
                    # print(m.requires_grad)
                # Forward G with frozen classifiers
                output = model(*(batch[2:]))

                # G loss, take total loss and backprop
                lossesG = LossG(batch, output, step)
                total_loss = lossesG[0]

                # Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                # zero all gradients!!!
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()


                #Dstep
                # freeze the model, except for classifiers
                for m in model.module.adv_classifiers.parameters():
                    m.requires_grad=True
                smallbatch=(batch[6],batch[12],batch[2])
                with torch.no_grad():
                    (z_acc, z_spk, z_acc_sg, mlvae_stats) = model.module.MLVAEencoder(smallbatch[0], acc_labels=smallbatch[1], spk_labels=smallbatch[2], input_lengths=None)

                acc_prob = model.module.adv_classifiers(z_spk)

                # D loss
                total_lossD, acc_ce_loss = LossD(smallbatch, acc_prob, step)
                # lossesD = LossD(batch, output, step)
                # total_loss = losses[0]

                # Backward
                total_lossD = total_lossD / grad_acc_step
                total_lossD.backward()

                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                #     # run the refenc, mlvae to get acc and spk embs
                # # continue with classifiers with grad
                #  #nebo []?
                # # Forward D step
                # #output just spk and acc probs
                # output = model.dstep(*batch[6],*batch[2],*batch[12]) #mel, spk, acc

                # # D loss
                # lossesD = LossD(smallbatch, output, step)
                # lossesD = LossD(batch, output, step)
                # total_loss = losses[0]
                    
                # # Backward
                # total_loss = total_loss / grad_acc_step
                # total_loss.backward()




                # alternate batches option
                # # 0ids,
                # # 1raw_texts,
                # # 2speakers,
                # # 3texts,
                # # 4src_lens,
                # # 5max_src_len,
                # # 6mels,
                # # 7mel_lens,
                # # 8max_mel_len,
                # # 9r_len_pad,
                # # 10gates,
                # # 11spker_embeds,
                # # 12accents,

                # #put a condition inside the forward function
                # if gstep: #gstep is true, do G step
                #     # freeze classifiers
                    
                #     # Forward with G condition
                #     output = model.gstep(*(batch[2:]))

                #     # G loss, take total loss and backprop
                #     lossesG = LossG(batch, output, step)
                #     total_loss = lossesG[0]

                #     # Backward
                #     total_loss = total_loss / grad_acc_step
                #     total_loss.backward()

                #     gstep = False # change to D step now
                # else: #gstep is not true, do D step
                #     # freeze the model, except for classifiers
                #     # with torch.no_grad():
                #         # run the refenc, mlvae to get acc and spk embs
                #     # continue with classifiers with grad
                #     smallbatch=(batch[6],batch[2],batch[12]) #nebo []?
                #     # Forward D step
                #     #output just spk and acc probs
                #     output = model.dstep(*batch[6],*batch[2],*batch[12]) #mel, spk, acc

                #     # D loss
                #     lossesD = LossD(smallbatch, output, step)
                #     lossesD = LossD(batch, output, step)
                #     total_loss = losses[0]
                        
                #     # Backward
                #     total_loss = total_loss / grad_acc_step
                #     total_loss.backward()

                #     gstep = True





                if step % log_step == 0:
                    losses = [l.item() for l in lossesG]
                    message1 = "Step {}/{}, ".format(step, total_step)
                    message2 = "Total Loss: {:.4f}, Mel Loss: {:.4f}, Gate Loss: {:.4f}, Guided Attention Loss: {:.4f}, Acc KL Loss: {:.4f}, Spk KL Loss: {:.4f}, Acc_ADV Loss: {:.4f}, Acc_CE Loss: {:.4f}".format(
                        *losses, acc_ce_loss
                    )

                    with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                        f.write(message1 + message2 + "\n")

                    outer_bar.write(message1 + message2)

                    log(train_logger, step, losses=losses, lossesD=[acc_ce_loss], grad_norm=grad_norm)

                if step % synth_step == 0:
                    model.eval()
                    output_inference = model.module.inference(batch[2][0].unsqueeze(0), batch[3][0].unsqueeze(0), batch[6][0].unsqueeze(0), batch[5], batch[11], batch[12][0].unsqueeze(0))
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
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    log(
                        train_logger,
                        step=step,
                        fig=gate_fig,
                        tag="Gates/training",
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]
                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_inference,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_inferred".format(step, tag),
                    )
                    model.train()


                if step % val_step == 0:
                    model.eval()
                    # message = evaluate(model, step, configs, mel_stats, val_logger, vocoder, len(losses))
                    message = evaluate(model, step, configs, mel_stats, val_logger, vocoder, 8)

                    with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                        f.write(message + "\n")
                    outer_bar.write(message)

                    model.train()

                if step % save_step == 0:
                    torch.save(
                        {
                            "model": model.module.state_dict(),
                            "optimizer": optimizer._optimizer.state_dict(),
                        },
                        os.path.join(
                            train_config["path"]["ckpt_path"],
                            "{}.pth.tar".format(step),
                        ),
                    )

                # if step % conversion_step == 0:
                #     model.eval()
                #     # message = evaluate(model, step, configs, mel_stats, val_logger, vocoder, len(losses))
                #     training_conversion(model, step, configs, mel_stats, val_logger, vocoder, 10)

                #     outer_bar.write('CONVERSION INFERENCE COMPLETE')

                #     model.train()

                if step == total_step:
                    quit()
                step += 1
                outer_bar.update(1)

            inner_bar.update(1)
        epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        default='L2arctic',
        help="name of dataset",
    )
    args = parser.parse_args()

    # Read Config
    configs = get_configs_of(args.dataset)

    main(args, configs)
