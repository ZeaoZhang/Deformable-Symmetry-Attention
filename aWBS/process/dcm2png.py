import random
import pickle
import os
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from PIL import Image, ImageOps
import pandas as pd

def get_LUT_value(data, window, level):
    """Apply the RGB Look-Up Table for the given
       data and window/level value."""
    try:
        window = window[0]
    except TypeError:
        pass
    try:
        level = level[0]
    except TypeError:
        pass

    return np.piecewise(data,
                        [data <= (level - 0.5 - (window - 1) / 2),
                         data > (level - 0.5 + (window - 1) / 2)],
                        [0, 255, lambda data: ((data - (level - 0.5)) /
                                               (window - 1) + 0.5) * (255 - 0)])

def get_ant_post(DS_files):
    width = 47
    center = 23.5
    if len(DS_files) == 1:
        file_path = DS_files[0]
        dcm_img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))
        image = get_LUT_value(dcm_img[0], width, center)
        ant_img = Image.fromarray(image).convert('L')
        image = get_LUT_value(dcm_img[1], width, center)
        post_img = Image.fromarray(image).convert('L')
    elif len(DS_files) == 2:
        file_path = DS_files[0]
        dcm_img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))[0]
        image = get_LUT_value(dcm_img, width, center)
        img = Image.fromarray(image).convert('L')
        if pd.Series(file_path).str.contains("ANT|Ant|ant")[0]:
            ant_img = img
        elif pd.Series(file_path).str.contains("POST|post")[0]:
            post_img = img
        else:
            raise(ValueError("please check file path,%s" % file_path))
        file_path = DS_files[1]
        dcm_img = sitk.GetArrayFromImage(sitk.ReadImage(file_path))[0]
        image = get_LUT_value(dcm_img, width, center)
        img = Image.fromarray(image).convert('L')
        if pd.Series(file_path).str.contains("ANT|Ant|ant")[0]:
            ant_img = img
        elif pd.Series(file_path).str.contains("POST|post")[0]:
            post_img = img
        else:
            raise(ValueError("please check file path,%s" % file_path))
    else:
        raise ValueError("folder is not standard,%s" % DS_files[0])
    return ant_img, post_img


if __name__ == '__main__':  
    root_dir = "./archived_studies/"
    save_dir = "./png_data/"

    records = pd.read_csv("./records.excel")

    for i, record in records.iterrows():
        if record["file_names"]=='original data not saved!':
            continue
        DS_files = record["file_names"]
        check_id = record["_id"]
        if os.path.exists(os.path.join(save_dir,"%s.png"%check_id)):
            continue
        try:
            # print(records.iloc[i]["_id"])
            study_id = record["study_iuid"]
            DS_files = [os.path.join(root_dir,study_id,fname) for fname in DS_files]
            ant_img, post_img = get_ant_post(DS_files)
            if 512 in ant_img.size:
                ant_img = ant_img.crop([128, 0, 384, 1024])
                post_img = post_img.crop([128, 0, 384, 1024])
            merge_img = Image.fromarray(np.hstack([np.array(ant_img), np.array(post_img)]))
            merge_img = ImageOps.invert(merge_img)
            merge_img.save(os.path.join(save_dir,"%s.png"%check_id))
        except:
            print("convert %s, failed!"%check_id)


