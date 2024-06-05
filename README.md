# Accent Conversion in Text-To-Speech Using Multi-Level VAE and Adversarial Training
This repo combines a Tacotron2 model with a ML-VAE and adversarial learning to target accent conversion in TTS settings (pick a speaker A with and assign them accent B).

Paper link: https://arxiv.org/abs/2406.01018

Samples link: https://amaai-lab.github.io/Accented-TTS-MLVAE-ADV/

![alt text](https://github.com/AMAAI-Lab/Accented-TTS-MLVAE-ADV/blob/main/schematic.png)

This code is built upon Comprehensive-TTS: https://github.com/keonlee9420/Comprehensive-Transformer-TTS

## Training
First download your dataset and preprocess the audio data into mel spectrogram `.npy` arrays with the `preprocess.py script`. We used L2CMU in this paper, which stands for a combination of L2Arctic (24 speakers) and CMUArctic (4 speakers). Then run ``CUDA_VISIBLE_DEVICES=X python train.py --dataset L2CMU``

## Inference
Once trained, you can run `extract_stats.py` to retrieve the accent and speaker embeddings of your evaluation set and store them. Then, you can synthesize with one of the synth scripts. :-)

Once trained, you can run ``CUDA_VISIBLE_DEVICES=X python synthesize.py --dataset L2Arctic --restore_step [N] --mode [batch/single] --text [TXT] --speaker_id [SPID] --accent [ACC]``

## BibTeX citation
```
@article{melechovsky2024accent,
      title={Accent Conversion in Text-To-Speech Using Multi-Level VAE and Adversarial Training}, 
      author={Jan Melechovsky and Ambuj Mehrish and Berrak Sisman and Dorien Herremans},
      journal={arXiv preprint arXiv:2406.01018},
      year={2024}
}
```
