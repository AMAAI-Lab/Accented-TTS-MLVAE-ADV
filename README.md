# Accent Conversion in Text-To-Speech Using Multi-Level VAE and Adversarial Training
This repo combines a Tacotron2 model with a ML-VAE and adversarial learning to target accent conversion in TTS settings (pick a speaker A with and assign them accent B).

Paper link: TBA

Samples link: https://amaai-lab.github.io/Accented-TTS-MLVAE-ADV/

## Training
First preprocess your data into mel spectrogram .npy arrays with the preprocess.py script. We used L2CMU in this paper, which stands for a combination of L2Arctic (24 speakers) and CMUArctic (4 speakers). Then run CUDA_VISIBLE_DEVICES=X python train.py --dataset L2CMU

## Inference
Once trained, you can run extract_stats.py to retrieve the accent and speaker embeddings of your evaluation set and store them. Then, you can synthesize with one of the synth scripts. :-)

Once trained, you can run CUDA_VISIBLE_DEVICES=X python synthesize.py --dataset L2Arctic --restore_step [N] --mode [batch/single] --text [TXT] --speaker_id [SPID] --accent [ACC]
