import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import pydicom as dcm
import seaborn as sns
import matplotlib.pyplot as plt
from pydicom.pixel_data_handlers.util import apply_voi_lut
import pdb

ROOT = "/home/kshitiz/scratch/MAMMO/DATA5/RSNA/"
# ROOT = "/home/data/submit/WORKING/kshitiz/mammo/DATA3/RSNA"
DATA_DIR = os.path.join(ROOT, "RAW_data")


train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
# train_df = train_df[train_df['cancer']==1]
test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
# print("Train / test data shape: ", train_df.shape, test_df.shape)
# print("Train images folders: ", len(os.listdir(os.path.join(DATA_DIR,"train_images"))))
# print("Test images folders: ", len(os.listdir(os.path.join(DATA_DIR,"test_images"))))

import cv2
from concurrent.futures import ThreadPoolExecutor


def load_image_pydicom(dataset, voi_lut=False):
    img = dataset.pixel_array
    img = apply_voi_lut(img, dataset)
    if dataset.PhotometricInterpretation == "MONOCHROME2":
        img = img - np.min(img)
    else:
        img = np.amax(img) - img
    img = img / np.max(img)
    img=(img * 255).astype(np.uint8)
    return img


def get_dicom_imgs(args):
    data_path, patient_id = args
    images_path = os.path.join(data_path,patient_id)
    image_list = os.listdir(images_path)
    new_path = images_path.split("/")
    new_path[7] = "neg_patients"
    new_path = "/".join(new_path)
    if(os.path.isdir(new_path)):
        print("Already done", patient_id)
        return
    print(patient_id)
    os.makedirs(new_path, exist_ok=True)
    for image in image_list:
        if(image[-4:]==".dcm"):
            image_id = image.split(".dcm")[0]
            image_path = os.path.join(images_path, image)

            data_row_img_data = dcm.dcmread(image_path)
            img_data = load_image_pydicom(data_row_img_data)
            new_image_path1 = os.path.join(new_path, image_id+".png")
            # new_image_path2 = os.path.join(new_path, image_id+"_2.png")
            # plt.imsave(new_image_path1, img_data)
            cv2.imwrite(new_image_path1, img_data)



set_diff = np.setdiff1d(train_df[train_df['cancer']==0].patient_id.unique(), train_df[train_df['cancer']==1].patient_id.unique())
# np.random.shuffle(set_diff)
# set_diff = train_df[train_df['cancer']==1].patient_id.unique()
# all_patients = train_df.patient_id.unique()

args = []
for i in set_diff:
    args.append((os.path.join(DATA_DIR,"train_images"), str(i)))

# print(len(args))
# exit(0)
# get_dicom_imgs((os.path.join(DATA_DIR,"train_images"), "10025"))

# import pdb
# pdb.set_trace()

with ThreadPoolExecutor(max_workers = 32) as executor:
      results = executor.map(get_dicom_imgs, args)

# for patient_id in tqdm(train_df.patient_id.unique()):
# get_dicom_imgs((os.path.join(DATA_DIR,"train_images"), str(10006)))
# get_dicom_imgs((os.path.join(DATA_DIR,"train_images"), str(10011)))