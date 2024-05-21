import argparse
import os
import json
import pandas as pd
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-job", type=str, help="Path to job config.")
    parser.add_argument("--cxr_path", type=str, default="f")
    parser.add_argument("--cxp_path", type=str, default="f")
    parser.add_argument("--pc_path", type=str, default="f")
    parser.add_argument("--dest_path", type=str, default="f")
    parser.add_argument("--seed", type=int, default=2024)

    args = parser.parse_args()
    dest_path = os.path.join(args.dest_path, "combo_datalists", f"seed_{args.seed}")
    os.makedirs(dest_path, exist_ok=True)

    # CXR
    data_cxr = pd.read_csv(
        os.path.join(args.cxr_path, "Data_Entry_2017_v2020.csv"),
        usecols=["Image Index", "Patient ID", "Finding Labels"],
    )
    # encode the labels and filter out all other columns
    labels = data_cxr["Finding Labels"].str.get_dummies(sep="|")
    data_cxr = pd.concat([data_cxr, labels], axis=1)
    data_cxr.drop(columns=["Finding Labels"], inplace=True)

    # filter the training/test rows
    train_list = pd.read_csv(os.path.join(args.cxr_path, "train_val_list.txt"), header=None)[0].to_list()
    train_data_cxr = data_cxr[data_cxr["Image Index"].isin(train_list)]
    valid_list = pd.read_csv(os.path.join(args.cxr_path, "test_list.txt"), header=None)[0].to_list()
    valid_data_cxr = data_cxr[data_cxr["Image Index"].isin(valid_list)]

    # get the unique patient IDs for the training set
    patients = train_data_cxr["Patient ID"].unique()

    # randomly choose 20% of the train data_cxr to keep
    np.random.seed(args.seed)
    chosen_patients = np.random.choice(patients, int(0.2 * len(patients)), replace=False)
    train_data_cxr = train_data_cxr[train_data_cxr["Patient ID"].isin(chosen_patients)]
    json_data_cxr = {"training": [], "validation": []}
    for idx, row in train_data_cxr.iterrows():
        new_item = {}
        new_item["image"] = row["Image Index"]
        new_item["label"] = row.iloc[2:].values.tolist()
        json_data_cxr["training"].append(new_item)
    for idx, row in valid_data_cxr.iterrows():
        new_item = {}
        new_item["image"] = row["Image Index"]
        new_item["label"] = row.iloc[2:].values.tolist()
        json_data_cxr["validation"].append(new_item)
    with open(f"{dest_path}/cxr.json", "w") as f:
        json.dump(json_data_cxr, f, indent=4)

    class_sum = {}
    class_sum["cxr"] = data_cxr[data_cxr["Patient ID"].isin(chosen_patients)].iloc[:, 2:].sum().to_dict()
    cnt = 0
    for patient in chosen_patients:
        cnt += len(data_cxr[data_cxr["Patient ID"] == patient])
    class_sum["cxr"]["samples"] = cnt

    sum_file_name = os.path.join(args.dest_path, "summary_cxr.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write("Class counts for cxr: \n")
        sum_file.write(json.dumps(class_sum))

    # CXP
    train_data_cxp = pd.read_csv(
        os.path.join(args.cxp_path, "train.csv"),
        usecols=["filename", "label"],
    )
    valid_data_cxp = pd.read_csv(
        os.path.join(args.cxp_path, "valid.csv"),
        usecols=["filename", "label"],
    )
    train_data_cxp["label"] = train_data_cxp.iloc[3:].values.tolist()
    valid_data_cxp["label"] = train_data_cxp.iloc[3:].values.tolist()

    json_data_cxp = {"training": [], "validation": []}
    for idx, row in train_data_cxp.iterrows():
        new_item = {}
        new_item["image"] = row["filename"]
        new_item["label"] = row["label"]
        json_data_cxp["training"].append(new_item)
    for idx, row in valid_data_cxp.iterrows():
        new_item = {}
        new_item["image"] = row["filename"]
        new_item["label"] = row["label"]
        json_data_cxp["validation"].append(new_item)
    with open(f"{dest_path}/cxp.json", "w") as f:
        json.dump(json_data_cxp, f, indent=4)

    class_sum = {}
    class_sum["cxp"] = (
        train_data_cxp[train_data_cxr["Patient ID"].isin(chosen_patients_cxp)].iloc[:, 2:].sum().to_dict()
    )
    cnt = 0
    for patient in chosen_patients_cxp:
        cnt += len(train_data_cxp[train_data_cxp["Patient ID"] == patient])
    class_sum["cxp"]["samples"] = cnt

    sum_file_name = os.path.join(args.dest_path, "summary_cxp.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write("Class counts for cxp: \n")
        sum_file.write(json.dumps(class_sum))

    # PC
    data = pd.read_csv(
        os.path.join(args.pc_path, "metadata.csv"),
        usecols=["filename", "label"],
    )
    data["label"] = data["label"].str.split(" ")
    json_data = {"training": [], "validation": []}
    for idx, row in data.iterrows():
        new_item = {}
        new_item["image"] = row["filename"]
        new_item["label"] = row["label"]
        json_data["training"].append(new_item)
    with open(f"{dest_path}/pc.json", "w") as f:
        json.dump(json_data, f, indent=4)

    class_sum = {}
    class_sum["pc"] = data.iloc[:, 1].apply(lambda x: len(x)).sum()
    class_sum["samples"] = len(data)
    sum_file_name = os.path.join(args.dest_path, "summary_pc.txt")
    with open(sum_file_name, "w") as sum_file:
        sum_file.write("Class counts for pc: \n")
        sum_file.write(json.dumps(class_sum))

    # Meta
    meta = {
        "cxr": os.path.join(dest_path, "cxr.json"),
        "cxp": os.path.join(dest_path, "cxp.json"),
        "pc": os.path.join(dest_path, "pc.json"),
    }
    with open(f"{dest_path}/meta.json", "w") as f:
        json.dump(meta, f, indent=4)

    print("Done!")
    return


# CXR columns (from Data_Entry_2017_v2020.csv)


# CXP columns
# Path,Sex,Age,Frontal/Lateral,AP/PA,
# No Finding,
# Enlarged Cardiomediastinum,
# Cardiomegaly,
# Lung Opacity,
# Lung Lesion,
# Edema,
# Consolidation,
# Pneumonia,
# Atelectasis,
# Pneumothorax,
# Pleural Effusion,
# Pleural Other,
# Fracture,
# Support Devices

# PC diseases...352 of them

if __name__ == "__main__":
    main()

common_diseases = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "No Finding",
    "Pneumonia",
    "Pneumothorax",
]

uncommon_diseases_cxr = [
    "Emphysema",
    "Fibrosis",
    "Hernia",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pleural_Thickening",
]

uncommon_diseases_cxp = [
    "Enlarged Cardiomediastinum",
    "Lung Opacity",
    "Lung Lesion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

# the diseases in PC are too many to list here
# no finding appears as normal in PC
# edema appears as pulmonary edema in PC


# go through the dataframe and drop rows where there are no 1s for any common diseases
# cnt = 0
# for idx, row in data.iterrows():
#     if row[common_diseases].sum() == 0:
#         data.drop(idx, inplace=True)
#         cnt += 1
# print(f"Dropped {cnt} rows from the dataset.")

# drop any column past 3 whose name is not in common_diseases
# data.drop(columns=[col for col in data.columns[3:] if col not in common_diseases], inplace=True)


# rearrange the disease columns so that they are in the same order as common_diseases
# data = data[["filename", "label"] + common_diseases]

# save the dataframe to a new csv file
# data.to_csv("metadata.csv", index=True) 
