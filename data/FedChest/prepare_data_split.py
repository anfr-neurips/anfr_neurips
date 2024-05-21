import argparse
import json
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str)
parser.add_argument(
    "--site_name",
    type=str, choices=["cxr14", "cxp_old", "cxp_young", "padchest"]
)
parser.add_argument("--out_path", type=str)


def partition_data(data_dir, site_name, out_path):

    json_data = {"training": [], "validation": []}
    json_test = {"testing": []}
    train_data = pd.read_csv(f"{site_name}_train.csv")
    valid_data = pd.read_csv(f"{site_name}_val.csv")
    # if "cxp" in site_name:
    #     if "old" in site_name:
    #         test_data = None
    #     else:
    #         test_data = pd.read_csv("cxp_test.csv")
    # else:
    #     test_data = pd.read_csv(f"{site_name}_test.csv")
    test_data = pd.read_csv(f"{site_name}_test.csv")

    for idx, row in train_data.iterrows():
        new_item = {}
        new_item["image"] = os.path.join(os.getcwd(), data_dir, row["Path"])
        new_item["label"] = row.iloc[3:].values.tolist()
        json_data["training"].append(new_item)
    for idx, row in valid_data.iterrows():
        new_item = {}
        new_item["image"] = os.path.join(os.getcwd(), data_dir, row["Path"])
        new_item["label"] = row.iloc[3:].values.tolist()
        json_data["validation"].append(new_item)
    with open(f"{out_path}/client_{site_name}.json", "w") as f:
        json.dump(json_data, f, indent=4)

    if test_data is not None:
        for idx, row in test_data.iterrows():
            new_item = {}
            new_item["image"] = os.path.join(
                os.getcwd(), data_dir, row["Path"])
            new_item["label"] = row.iloc[3:].values.tolist()
            json_test["testing"].append(new_item)
        with open(f"{out_path}/client_{site_name}_test.json", "w") as f:
            json.dump(json_test, f, indent=4)

    class_sum = train_data.iloc[:, 3:].sum().to_dict()
    class_sum.update({"samples": len(train_data)})
    with open(f"{out_path}/{site_name}_summary.txt", "w") as sum_file:
        sum_file.write(json.dumps(class_sum))


if __name__ == "__main__":
    args = parser.parse_args()
    partition_data(
        data_dir=args.data_dir,
        site_name=args.site_name,
        out_path=args.out_path,
    )
