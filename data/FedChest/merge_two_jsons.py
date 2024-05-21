import argparse
import copy
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_1", action="store", required=True)
    parser.add_argument("--json_2", action="store")
    parser.add_argument("--json_out", action="store")
    parser.add_argument("--keys", action="store", required=True)
    args = parser.parse_args()

    with open(args.json_1) as a:
        json_1_data = json.load(a)

    with open(args.json_2) as b:
        json_2_data = json.load(b)

    json_data = copy.deepcopy(json_1_data)
    for key in args.keys.split(","):
        json_data[key].extend(json_2_data[key])

    with open(args.json_out, "w") as f:
        json.dump(json_data, f, indent=4)

    return


if __name__ == "__main__":
    main()
