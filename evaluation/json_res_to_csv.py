import pandas as pd
import os
import json
import datetime
from config import EXEC_RES_PATH, CSVS_PATH

# CHANGE EVERY TIME
# _________________________________________________
DATE = "2020-09-08"
# _________________________________________________

JSON_PATH = os.path.join(EXEC_RES_PATH, DATE)
CSV_PATH = os.path.join(CSVS_PATH, DATE)
if not os.path.isdir(CSV_PATH):
    os.makedirs(CSV_PATH)

with open(os.path.join(JSON_PATH, "run_results.json"), "r") as file:
    res_dict = json.load(file)

for topic in os.listdir(JSON_PATH):
    if os.path.isfile(os.path.join(JSON_PATH, topic)):
        continue

    dirs = os.listdir(os.path.join(os.getcwd(), "input", topic))
    fracs = list(res_dict[topic]["run_params"])
    j = -1

    for i, dir_ in enumerate(dirs):
        if i % 2 != 0:
            continue

        if i % 4 == 0:
            j += 1

        with open(os.path.join(JSON_PATH, topic, dirs[i]), "r") as file:
            labels_dict = json.load(file)

        with open(os.path.join(JSON_PATH, topic, dirs[i + 1]), "r") as file:
            outliers = json.load(file)

        max_ = max([len(v) for v in labels_dict.values()] + [len(outliers)])
        labels_dict = {k: v for k,v in sorted(labels_dict.items(), reverse=True, key=lambda x: len(x[1]))}
        new_labels_dict = {k: [""] * 4 + v + [""] * (max_ - len(v)) for k,v in labels_dict.items()}

        w_spaces_labels_dict = {}
        for i, (k, v) in enumerate(new_labels_dict.items()):
            w_spaces_labels_dict.update({
                k: v,
                "bl1_" + str(i): [""] * len(v)
            })

        data_df = pd.DataFrame(w_spaces_labels_dict)
        now = datetime.datetime.now()
        data_df.to_csv(os.path.join(CSV_PATH,
                                    now.strftime("%Y-%m-%d_%H-%M") + "_" + str(fracs[j]) + "_" +
                                    dir_.split("_")[-3] + "_" + topic + ".csv"),
                       encoding='utf-8-sig')
    print("JSONs to CSV converted to the folder " + JSON_PATH)
