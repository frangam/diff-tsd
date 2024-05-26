#!/home/fmgarmor/miot_env/bin/python3

'''
Generation of images for train/test and assess the quality of synthetic.
Synthetic images are the train set.
TEST set are real data, which are used to validate the model in the training phase
Test set are real data, which is used to test the trained model
'''

from functools import partial
import shutil
from PIL import Image
import glob
import pandas as pd
import os
import torch
from torchvision import datasets, transforms, utils
from torch.utils import data
from tqdm.auto import trange, tqdm
from sklearn.model_selection import train_test_split


from huggingface_hub import login
from datasets import load_dataset
import argparse

import k_diffusion as K

from utils import set_gpu


# Function to save dataset to a folder
def save_dataset(images, labels, folder, prefix, is_train=True):
    os.makedirs(folder, exist_ok=True)
    names = []
    for id, src_path in enumerate(images):
        print("IMG source path:", src_path)
        dst_path = f"{folder}{prefix}_{id}.png"
        print("IMG destiny path:", dst_path)
        shutil.copyfile(src_path, dst_path) 
        names.append(f"{prefix}_{id}.png")
    pd.DataFrame({"filename": names, "label": labels}).to_csv(f"{folder}{'training' if is_train else 'test'}_labels.csv")

# Function to load dataset and apply transformations
def load_and_transform_dataset(dataset_config, fold, cl_idx, transform, batch_size, subset):
    dataset = load_dataset(f"{dataset_config['location']}{fold}", split=subset)
    dataset = dataset.filter(lambda t: t["label"] == cl_idx)
    dataset.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=transform, image_key="image"))
    return data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)


def stratified_sample(x, y, proportion):
    """Sample a stratified subset of data."""
    if proportion == 0:
        return [], []
    if proportion == 1:
        return x, y
    _, x_sample, _, y_sample = train_test_split(x, y, test_size=proportion, stratify=y, random_state=42)
    return x_sample, y_sample


def main():
  '''Examples of runs:
  ** LOSO APPROACH
  $  nohup ./tgen/data.py --config configs/config_wisdm_128x128_loso.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > logs/data_splits.log &
  
  - only 2 clases (1, 3) and only the split 3 (real train images)
  $  nohup ./tgen/data.py --prefix exp-classes-1-3 --class-names 1,3 --splits 2 > logs/data_splits.log &


  ** LOTO approach
  $  nohup ./tgen/data.py --config configs/config_wisdm_128x128_loto.json --prefix exp-classes-all-classes --class-names 0,1,2,3,4 --splits 0,1,2 > logs/data_splits-loto.log &

  '''
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  p.add_argument('--gpu-id', type=int, default=0, help='the GPU device ID')

  p.add_argument('--data-name', type=str, default="WISDM", help='the database name')


  p.add_argument('--batch-size', type=int, default=16,
                  help='the batch size')
  p.add_argument('--image-size', type=int, default=128,
                  help='the image resolution size')
  p.add_argument('--n-folds', type=int, default=3,
                  help='the number of folds')
  p.add_argument('--class-names', type=str, default="0,1,2,3,4", #numbers: [0:"Walking",1:"Jogging",2:"Stairs",3:"Sitting",4:"Standing"]
                  help='the number of classes')
  p.add_argument('--splits', type=str, default="0,1,2", #numbers: 0: train split (synthetic images); 1: test split, 2: real train (real images for train)
                  help='the number of classes')
  p.add_argument('--prefix', type=str, default='exp-all-classes',
                  help='the output prefix')
  p.add_argument('--config', type=str, default="configs/config_wisdm_128x128_loso.json",help='the configuration file')
  
  args = p.parse_args()

  set_gpu(args.gpu_id)


  folds = args.n_folds
  data_name = args.data_name
  classes = [int(c) for c in args.class_names.split(",")]
  splits = [int(c) for c in args.splits.split(",")]
  print("Classes")

  config = K.config.load_config(open(args.config))
  # model_config = config['model']
  dataset_config = config['dataset']
  sampling_method = dataset_config["sampling"]

  CLASS_NAMES = ["Walking","Jogging","Stairs","Sitting","Standing"]

  if data_name == "ADL_Dataset":
    CLASS_NAMES = ["Walk", "Descend_stairs", "Climb_stairs", "Sitdown_chair", "Standup_chair"]


  tf = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.CenterCrop(args.image_size),
        K.augmentation.KarrasAugmentationPipeline(0.0),
    ])


  for fold in range(folds):
    x_synth_train, y_synth_train, subjects_train, file_id_train,  x_real_train, y_real_train, subjects_real_train, file_id_real_train,  x_test, y_test, subjects_test, file_id_test, x_val, y_val, subjects_val, file_id_val, x_real_and_synth_train, y_real_and_synth_train, subjects_real_and_synth_train, file_id_real_and_synth_train, x_real_and_synth_test, y_real_and_synth_test, subjects_real_and_synth_test, file_id_real_and_synth_test = [], [], [], 0, [], [], [], 0, [], [], [], 0, [], [], [], 0, [], [], [], 0, [], [], [], 0

    for cl_idx in classes:
      cl = CLASS_NAMES[cl_idx]
      print(f"[Fold-{fold}] Coping files class: {cl} [{cl_idx}]")
      # test_src = f"data/WISDM/plots/recurrence_plot/{folds}-fold/fold-{fold}/test/{cl_idx}/"
      # val_src = f"data/WISDM/plots/recurrence_plot/validation/{cl_idx}/"
      
      #----------------------------------------------------
      #SYNTHETIC (TRAIN) files are the train dataset
      #----------------------------------------------------
      if 0 in splits:
        print("Generating SYNTHETIC (TRAIN) split")
        synth_dir = f"results/{sampling_method}/fold_{fold}/class_{cl_idx}/sample/"
        synth_dst_dir = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/train/"
        os.makedirs(synth_dst_dir, exist_ok=True)

        for i,f in enumerate( os.listdir(synth_dir)):
          # print(f"From: {synth_dir}{f}. TO: {dst}x{next_f+i}.png")      
          src_path = os.path.join(synth_dir, f)
          dst_path = os.path.join(synth_dst_dir, f"synth_{file_id_train}.png")
          shutil.copyfile(src_path, dst_path) 
          x_synth_train.append(dst_path)
         
          #   subjects_train.append(-1) #synthetic will be subject ID -1
          y_synth_train.append(cl) #here, the class label is the index of class list
          file_id_train += 1  
        pd.DataFrame({"filename": [os.path.basename(fp) for fp in x_synth_train], "label": y_synth_train}).to_csv(f"{synth_dst_dir}/training_labels.csv")

      #----------------------------------------------------
      #TEST files
      #----------------------------------------------------
      if 1 in splits:
        print("Generating TEST split")
        test_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/test/"
        os.makedirs(test_dst, exist_ok=True)
        

        test_set = load_dataset(f"{dataset_config['location']}{fold}", split="test")
        test_set = test_set.filter(lambda t: t["label"] == cl_idx)
        test_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        test_dl = data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(test_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            # K.utils.to_pil_image(img).save(f"{test_dst}test_{file_id_test}.png")
            # x_test.append(f"test_{file_id_test}.png")
            dst_path = os.path.join(test_dst, f"test_{file_id_test}.png")
            K.utils.to_pil_image(img).save(dst_path)
            x_test.append(dst_path)
            y_test.append(cl)
            file_id_test += 1
        pd.DataFrame({"filename": [os.path.basename(fp) for fp in x_test], "label": y_test}).to_csv(f"{test_dst}/test_labels.csv")

      
      #----------------------------------------------------
      #REAL TRAIN files
      #----------------------------------------------------
      if 2 in splits:
        print("Generating REAL TRAIN split")
        real_train_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/real_train/"
        os.makedirs(real_train_dst, exist_ok=True)

        real_train_set = load_dataset(f"{dataset_config['location']}{fold}", split="train")
        real_train_set = real_train_set.filter(lambda t: t["label"] == cl_idx)
        real_train_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        real_train_dl = data.DataLoader(real_train_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(real_train_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            # K.utils.to_pil_image(img).save(f"{real_train_dst}real_train_{file_id_real_train}.png")
            # x_real_train.append(f"real_train_{file_id_real_train}.png")
            dst_path = os.path.join(real_train_dst, f"real_train_{file_id_real_train}.png")
            K.utils.to_pil_image(img).save(dst_path)
            x_real_train.append(dst_path)
            y_real_train.append(cl)
            file_id_real_train += 1
        pd.DataFrame({"filename": [os.path.basename(fp) for fp in x_real_train], "label": y_real_train}).to_csv(f"{real_train_dst}/real_training_labels.csv")

      #----------------------------------------------------
      #VALIDATION files
      #----------------------------------------------------
      if 3 in splits:
        print("Generating VALIDATION split")
        val_dst = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}/data/fold_{fold}/validation/"
        os.makedirs(val_dst, exist_ok=True)

        val_set = load_dataset(f"{dataset_config['location']}{fold}", split="validation")
        val_set = val_set.filter(lambda t: t["label"] == cl_idx)
        val_set.set_transform(partial(K.utils.hf_datasets_augs_helper, transform=tf, image_key="image"))
        val_dl = data.DataLoader(val_set, batch_size=args.batch_size, shuffle=False, drop_last=True)
        for batch in tqdm(val_dl):
          reals, reals_no_aug, aug_cond = batch["image"]
          # print(reals_no_aug.shape)
          for img in reals_no_aug:
            # print(img.shape)
            # K.utils.to_pil_image(img).save(f"{val_dst}val_{file_id_val}.png")
            # x_val.append(f"val_{file_id_val}.png")
            dst_path = os.path.join(val_dst, f"val_{file_id_val}.png")
            K.utils.to_pil_image(img).save(dst_path)
            x_val.append(dst_path)
            y_val.append(cl)
            file_id_val += 1
        pd.DataFrame({"filename": [os.path.basename(fp) for fp in x_val], "label": y_val}).to_csv(f"{val_dst}/val_labels.csv")


      #----------------------------------------------------
      #BOTH REAL + SYNTHETIC TRAIN files
      #with a percentage over the total real data (e.g. 75% real + 25% synth for every fold)
      #----------------------------------------------------
      if 4 in splits:
        proportions = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.25, 0.0]

        print("Generating BOTH: REAL + SYNTHETIC TRAIN split")
        print(f"Generating data for different proportions of real and synthetic data. Proportions: {proportions}")
        real_data_count = len(x_real_train)
        synth_data_count = len(x_synth_train)

        for p_real in proportions:
          if p_real > 0:
              real_subset_x, real_subset_y = stratified_sample(x_real_train, y_real_train, p_real)
          else:
              real_subset_x, real_subset_y = [], []

          if p_real < 1:
              p_synth = round(1 - p_real,2) * (real_data_count / synth_data_count)
              synth_subset_x, synth_subset_y = stratified_sample(x_synth_train, y_synth_train, p_synth)
          else:
              p_synth = 0
              synth_subset_x, synth_subset_y = [], []

          print(f"[Fold-{fold} Class-{cl_idx}] Proportion for real images: {p_real} || Shapes >> real_subset_x {len(real_subset_x)} synth_subset_x: {len(synth_subset_x)}")

          combined_data = real_subset_x + synth_subset_x
          combined_labels = real_subset_y + synth_subset_y

          print(f"[Fold-{fold} Class-{cl_idx}] Proportion for real images: {p_real} || Shapes >> combined_data: {len(combined_data)}")


          folder_name = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}_{int(p_real*100)}-real-{int(round(1-p_real,2)*100)}-synth/data/fold_{fold}/train/"
          save_dataset(combined_data, combined_labels, folder_name, prefix="data_scarcity", is_train=True)

          # Save the corresponding test set
          test_folder_name = f"results/evaluation_synthetic_quality/{sampling_method}/{args.prefix}_{int(p_real*100)}-real-{int(round(1-p_real,2)*100)}-synth/data/fold_{fold}/test/"
          save_dataset(x_test, y_test, test_folder_name, prefix="data_scarcity", is_train=False)

     
if __name__ == '__main__':
    main()