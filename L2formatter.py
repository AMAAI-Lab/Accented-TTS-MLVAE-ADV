def l2format(root_path, meta_files=None, ignored_speakers=None):
 
"INPUT ROOT_PATH AS L2ARCTIC/TRAIN OR L2ARCTIC/EVAL OR L2ARCTIC/TEST"

"Takes in dataset path, outputs a labeled list, where each entry has wavpath, textpath, acc label (int) and spkname (string)"


    root_path_l2arctic = root_path

    # file_ext = "flac"
    items = []
   
    meta_files = glob(f"{root_path_l2arctic}/*/transcript/*.txt", recursive=True)

    wav_files = glob(f"{root_path_l2arctic}/*/wav/*.wav", recursive=True)

    readme_file = glob(os.path.join(root_path_l2arctic,'README.md'))[0]
    
    with open(readme_file,'r', encoding="utf-8") as ttf:
        text=ttf.readlines()[34:58]
    sp_list = [x.split("|")[1] for x in text]
    acc_list = [x.split("|")[3] for x in text]
    unique_list = list(set(acc_list))
    
    
    meta_counter=0
    for meta_file in meta_files:
        with open(meta_file, "r", encoding="utf-8") as ttf:
            text=ttf.readlines()[0]
            speaker_name = os.path.basename(os.path.dirname(os.path.dirname(meta_file)))
            
            acc_id = sp_list.index(speaker_name)
            acc_label = unique_list.index(acc_list[acc_id]) #0 to 5
            
            wav_file=wav_files[meta_counter]
            meta_counter=meta_counter+1
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_name in ignored_speakers:
                    continue
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "labels": acc_label})
    for item in items:
        assert os.path.exists(item["audio_file"]), f" [!] wav files don't exist - {item[1]}"
    return items