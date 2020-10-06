import pandas as pd
import os, json
import re
import numpy as np
import shortuuid
from config import EXCEL_PATH, EXEC_RES_PATH, EVAL_PATH

# TO MODIFY
# ----------------
DATE = "2020-09-08"
#-----------------

TOPIC = "topic"
FRAC = "fraction"
TERMS = "terms"
HEADS = "heads"
CORE_TERMS = "core_terms"
GROUPS = "groups"
AVG_SIZE = "avg_size"
WEIGHT_GR = "weighted_group"
WEIGHT_L = "weighted_label"
APPR = "approach"
WEIGHT_TYPE = "w_type"
SCORE = "score"
ANEA = "ANEA_annotator"
HC = "HC"
PART = "participant"
BAL_GR = "balanced_group"
BAL_L = "balanced_label"


if DATE not in os.listdir(EXCEL_PATH):
    raise("No folder " + DATE + " in the " + EXCEL_PATH + ". No eval results calculated.")

# get res json
f = None
for f in os.listdir(os.path.join(EXEC_RES_PATH, DATE)):
    if os.path.isfile(os.path.join(EXEC_RES_PATH, DATE, f)):
        break

with open(os.path.join(EXEC_RES_PATH, DATE, f), "r") as file:
    results_dict = json.load(file)

allowed_areas = {"SC": "processing",
                 "SD": "softwaredev",
                 "DB": "databases",
                 "R": "travel"}

appr_tab_dict = {HC: [1,3,5,7],
                 ANEA: [2,4,6,8]}

used_fractions = {list(allowed_areas)[list(allowed_areas.values()).index(topic)]: [int(l) for l in list(runs["run_params"])]
                  for topic, runs in results_dict.items()}
j = -1
i = 0
eval_score_df = pd.DataFrame(columns=[PART, TOPIC, APPR, FRAC, TERMS, CORE_TERMS, WEIGHT_GR, WEIGHT_L])
EVAL_RESPATH = os.path.join(EXCEL_PATH, DATE)
eval_folders = os.listdir(EVAL_RESPATH)
examples_dict = {}

for eval_folder in eval_folders:
    if os.path.isfile(os.path.join(EVAL_RESPATH, eval_folder)):
        continue

    if not len(re.findall(r'\d', eval_folder)):
        continue

    eval_folder_full = os.path.join(os.path.join(EVAL_RESPATH, eval_folder))

    for ds_file in os.listdir(eval_folder_full):
        ds = ds_file.split(".")[0].split("_")[-1]
        topic = allowed_areas[ds]

        config_dict = {}
        for appr, tabs in appr_tab_dict.items():

            appr_dict = {}
            for tab, frac in zip(tabs, used_fractions[ds]):
                eval_sheet_df_ = pd.read_excel(os.path.join(eval_folder_full, ds_file), sheet_name=tab)
                eval_sheet_df = eval_sheet_df_.iloc[:, list(range(3, len(eval_sheet_df_.columns), 2))]
                scores_group = eval_sheet_df.iloc[2].values
                scores_labels = eval_sheet_df.iloc[4].values
                terms_df = eval_sheet_df.iloc[5:, :]
                terms_df.columns = eval_sheet_df.iloc[0] if not np.isnan(np.sum(scores_labels)) else ["cl"+str(j)
                                                                          for j in range(len(eval_sheet_df.columns))]
                group_lengths = [sum([not t for t in terms_df[col].isnull()]) for col in list(terms_df.columns)]
                weight_gr = np.sum([s*l for s, l in zip(scores_group, group_lengths)])/np.sum(group_lengths)
                weight_l = 0.0 if np.isnan(np.sum(scores_labels)) \
                                else np.sum([s*l for s, l in zip(scores_labels, group_lengths)])/np.sum(group_lengths)
                eval_score_df = eval_score_df.append(pd.DataFrame({
                    PART: eval_folder,
                    TOPIC: topic,
                    APPR: appr,
                    FRAC: frac,
                    TERMS: float(results_dict[topic]["run_params"][str(frac)]["terms_to_cluster"]),
                    CORE_TERMS: float(np.sum(group_lengths)),
                    WEIGHT_GR: weight_gr,
                    WEIGHT_L: weight_l
                    }, index=[i]))
                i += 1

                if frac not in appr_dict and topic not in examples_dict:
                    appr_dict[frac] = pd.DataFrame({
                        TOPIC: [topic] * len(terms_df.columns),
                        APPR: [appr] * len(terms_df.columns),
                        "labels": list(terms_df.columns),
                        "terms": [", ".join([v for v in terms_df[col].values if type(v) == str])
                                                for j, col in enumerate(list(terms_df.columns))]
                    }, index=[shortuuid.random()[:7] for j in range(len(terms_df.columns))])

            if appr not in config_dict and topic not in examples_dict:
                config_dict[appr] = appr_dict

        if topic not in examples_dict:
            examples_dict[topic] = config_dict


eval_score_df_short = eval_score_df.groupby(by=[TOPIC, APPR, FRAC]).mean()
eval_score_df_short = eval_score_df_short.reset_index()
eval_score_df.sort_values(by=[TOPIC, PART, APPR, FRAC], ascending=[True, True, False, True])
eval_score_df_short[BAL_GR] = np.divide(np.multiply(eval_score_df_short[WEIGHT_GR].values, eval_score_df_short[CORE_TERMS].values), eval_score_df_short[TERMS].values)
eval_score_df_short[BAL_L] = np.divide(np.multiply(eval_score_df_short[WEIGHT_L].values, eval_score_df_short[CORE_TERMS].values), eval_score_df_short[TERMS].values)
eval_score_df_short["best"] = np.multiply(eval_score_df_short[BAL_GR].values, eval_score_df_short[BAL_L].values)

example_qual_df = pd.DataFrame()

for topic in set(eval_score_df_short[TOPIC].values):
    local_df = eval_score_df_short[eval_score_df_short[TOPIC] == topic]
    row_max = local_df["best"].idxmax()

    print("topic = "+ topic, ", frac = " + str(local_df.loc[row_max, FRAC]))
    example_qual_df = example_qual_df.append(examples_dict[topic][local_df.loc[row_max, APPR]][local_df.loc[row_max, FRAC]])
    hc_df = local_df[local_df[APPR] == HC]
    row_hc_max = hc_df[BAL_GR].idxmax()
    example_qual_df = example_qual_df.append(examples_dict[topic][local_df.loc[row_hc_max, APPR]][local_df.loc[row_hc_max, FRAC]])

i = 0
res_df = pd.DataFrame(columns=[TOPIC, FRAC, HEADS, TERMS, APPR, CORE_TERMS, GROUPS, AVG_SIZE, WEIGHT_GR, BAL_GR, WEIGHT_L, BAL_L])

for topic in list(allowed_areas.values()):
    dataset_params = results_dict[topic]

    for frac, run_params in dataset_params["run_params"].items():
        frac = int(frac)
        # HC results
        res_df = res_df.append(pd.DataFrame({
            TOPIC: topic,
            FRAC: frac,
            HEADS: run_params["sel_unique_heads"],
            TERMS: run_params["terms_to_cluster"],
            APPR: HC,
            CORE_TERMS: run_params["clustered_agl"],
            GROUPS: run_params["groups_cl"],
            AVG_SIZE: round(run_params["clustered_agl"]/run_params["groups_cl"]),
            WEIGHT_GR: eval_score_df_short[(eval_score_df_short[TOPIC]==topic) & (eval_score_df_short[APPR]==HC) &
                                           (eval_score_df_short[FRAC]==frac)][WEIGHT_GR].iloc[0],
            WEIGHT_L: 0.0,
            BAL_GR: eval_score_df_short[(eval_score_df_short[TOPIC]==topic) & (eval_score_df_short[APPR]==HC) &
                                        (eval_score_df_short[FRAC]==frac)][BAL_GR].iloc[0],
            BAL_L: 0.0
            }, index=[i]), sort=False)
        i += 1

        # ANEA results
        res_df = res_df.append(pd.DataFrame({
            TOPIC: topic,
            FRAC: frac,
            HEADS: run_params["sel_unique_heads"],
            TERMS: run_params["terms_to_cluster"],
            APPR: ANEA,
            CORE_TERMS: run_params["clustered_sem"],
            GROUPS: run_params["groups_sem"],
            AVG_SIZE: round(run_params["clustered_sem"]/run_params["groups_sem"]),
            WEIGHT_GR: eval_score_df_short[(eval_score_df_short[TOPIC] == topic) & (eval_score_df_short[APPR] == ANEA) &
                                           (eval_score_df_short[FRAC] == frac)][WEIGHT_GR].iloc[0],
            WEIGHT_L: eval_score_df_short[(eval_score_df_short[TOPIC] == topic) & (eval_score_df_short[APPR] == ANEA) &
                                          (eval_score_df_short[FRAC] == frac)][WEIGHT_L].iloc[0],
            BAL_GR: eval_score_df_short[(eval_score_df_short[TOPIC] == topic) & (eval_score_df_short[APPR] == ANEA) &
                                        (eval_score_df_short[FRAC] == frac)][BAL_GR].iloc[0],
            BAL_L: eval_score_df_short[(eval_score_df_short[TOPIC] == topic) & (eval_score_df_short[APPR] == ANEA) &
                                       (eval_score_df_short[FRAC] == frac)][BAL_L].iloc[0]
        }, index=[i]), sort=False)
        i += 1

eval_score_df.to_csv(os.path.join(EVAL_PATH, DATE, "user_assessment.csv"), float_format='%.1f')
res_df.to_csv(os.path.join(EVAL_PATH, DATE, "res_table_full.csv"), float_format='%.1f')

res_df.sort_values(by=[APPR, TOPIC, FRAC], ascending=[False, True, True], inplace=True)
res_df.to_csv(os.path.join(EVAL_PATH, DATE, "res_table_sorted.csv"), float_format='%.1f')

example_qual_df.to_csv(os.path.join(EVAL_PATH, DATE, "qual_res_all.csv"), float_format='%.1f', encoding='utf-8-sig')
print("Results calculated and saved into " + os.path.join(EVAL_PATH, DATE) + ". \n")
