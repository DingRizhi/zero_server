import os
import glob
import json


def parse_json_result(json_file):

    label_count = {}
    with open(json_file, "r") as f:
        json_info = json.load(f)

        shapes = json_info["shapes"]

        for shape in shapes:
            label_name = shape["label"]
            points = shape["points"]
            score = shape["score"] if "score" in shape else 0

            if label_name not in label_count:
                label_count[label_name] = 0
            label_count[label_name] += 1
        print(f"---------------- json_file_name: {json_file}, count info:  ------------------")
        keys = sorted(label_count.keys())
        for k in keys:
            print(k, label_count[k])
        return label_count


def statistics_json_infos(json_dir):
    json_file_list = glob.glob(f"{json_dir}/*.json")
    total_count = {}

    json_file_list = sorted(json_file_list, key=lambda x: int(os.path.basename(x).split("-")[1]))
    for json_file in json_file_list:
        label_count = parse_json_result(json_file)
        for k, v in label_count.items():
            if k not in total_count:
                total_count[k] = v
            else:
                total_count[k] += v

    print(f"ALL:")
    for k, v in total_count.items():
        print(k, v)


if __name__ == '__main__':
    # statistics_json_infos("/home/log/PycharmProjects/zero_server/inference/1_output_no_post/NG")
    statistics_json_infos("/home/log/PycharmProjects/zero_server/inference/1_output_resnet18/NG")