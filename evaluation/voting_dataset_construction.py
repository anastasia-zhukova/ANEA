import pandas as pd
import json
import re
import numpy as np
import pickle
from config import *


def count_datasets(date: str):
    if date not in os.listdir(EXCEL_PATH):
        raise("No folder " + date + " in the " + EXCEL_PATH + ". No eval results calculated.")

    # get res json
    f = None
    for f in os.listdir(os.path.join(EXEC_RES_PATH, date)):
        if os.path.isfile(os.path.join(EXEC_RES_PATH, date, f)) and "run_results" in f:
            with open(os.path.join(EXEC_RES_PATH, date, f), "r") as file:
                results_dict = json.load(file)

        if os.path.isfile(os.path.join(EXEC_RES_PATH, date, f)) and "alllabels" in f:
            with open(os.path.join(EXEC_RES_PATH, date, f), "r") as file:
                all_labels = json.load(file)

        if os.path.isfile(os.path.join(EXEC_RES_PATH, date, f)) and "allvocab" in f:
            with open(os.path.join(EXEC_RES_PATH, date, f), "r") as file:
                all_vocab = json.load(file)

    allowed_areas = {"SC": "processing",
                     "SD": "softwaredev",
                     "DB": "databases",
                     "R": "travel"}
    allowed_areas_inv = dict(zip(allowed_areas.values(), allowed_areas.keys()))

    appr_tab_dict = {HC: [1,3,5,7],
                     ANEA: [2,4,6,8]}

    used_fractions = {list(allowed_areas)[list(allowed_areas.values()).index(topic)]: [int(l)
                                           for l in list(runs["run_params"])]
                                            for topic, runs in results_dict.items()}

    EVAL_RESPATH = os.path.join(EXCEL_PATH, date)
    eval_folders = os.listdir(EVAL_RESPATH)

    df_res_dict = {}
    df_count_dict = {}

    for topic in list(all_vocab):
        output = []

        for i, val1 in enumerate(used_fractions[allowed_areas_inv[topic]]):
            beg = str(val1)
            for j, val2 in enumerate(used_fractions[allowed_areas_inv[topic]]):
                if j <= i:
                    continue
                beg += "+" + str(val2)
                output.append(beg)

        vocab_df = pd.DataFrame(np.zeros((len(all_vocab[topic]), len(all_vocab[topic]))), columns=all_vocab[topic], index=all_vocab[topic])
        label_df = pd.DataFrame(np.zeros((len(all_vocab[topic]), len(all_labels[topic]))), columns=all_labels[topic], index=all_vocab[topic])
        df_res_dict[topic] = { appr + "_" + str(k): {VOCAB: vocab_df.copy(), LABELS: label_df.copy()}
                               for appr in list(appr_tab_dict) for k in used_fractions[allowed_areas_inv[topic]]}
        df_res_dict[topic].update({ appr + "_" + k: {VOCAB: vocab_df.copy(), LABELS: label_df.copy()} for appr in list(appr_tab_dict)
                                    for k in output})
        df_count_dict[topic] = {appr + "_" + str(k) : {VOCAB: vocab_df.copy(), LABELS: label_df.copy()} for appr in list(appr_tab_dict)
                                for k in used_fractions[allowed_areas_inv[topic]]}
        df_count_dict[topic].update({appr + "_" + k : {VOCAB: vocab_df.copy(), LABELS: label_df.copy()} for appr in list(appr_tab_dict)
                                for k in output})

    all_scores = {"processing": {},
                   "softwaredev": {},
                   "databases": {},
                   "travel": {}}

    terms_groups_dict = {}

    for eval_folder in eval_folders:
        logger.info(eval_folder)
        if os.path.isfile(os.path.join(EVAL_RESPATH, eval_folder)):
            continue

        if not len(re.findall(r'\d', eval_folder)):
            continue

        eval_folder_full = os.path.join(os.path.join(EVAL_RESPATH, eval_folder))

        for ds_file in os.listdir(eval_folder_full):
            ds = ds_file.split(".")[0].split("_")[-1]
            topic = allowed_areas[ds]
            terms_groups_dict[topic] = {}

            for appr, tabs in appr_tab_dict.items():

                for tab, frac in zip(tabs, used_fractions[ds]):
                    logger.info(appr + " " + str(frac))
                    frac_type = appr + "_" + str(frac)
                    eval_sheet_df_ = pd.read_excel(os.path.join(eval_folder_full, ds_file), sheet_name=tab)
                    eval_sheet_df = eval_sheet_df_.iloc[:, list(range(3, len(eval_sheet_df_.columns), 2))]
                    scores_group = eval_sheet_df.iloc[2].fillna(6).values

                    if frac_type not in all_scores[topic]:
                        all_scores[topic][frac_type] = []

                    all_scores[topic][frac_type].extend(scores_group)
                    scores_labels = eval_sheet_df.iloc[4].values
                    terms_df = eval_sheet_df.iloc[5:, :]
                    terms_df.columns = eval_sheet_df.iloc[0] if not np.isnan(np.sum(scores_labels)) else ["cl_"+str(j)
                                                                              for j in range(len(eval_sheet_df.columns))]
                    terms = {col: [t.strip() for t in terms_df[col] if t is not np.nan and type(t) == str and len(t.strip())]
                             for col in list(terms_df.columns)}
                    terms_groups_dict[topic][frac_type] = terms

                    for k, (col, t_gr) in enumerate(terms.items()):
                        t_gr = [v.strip() for v in t_gr if v in list(df_res_dict[topic][frac_type][VOCAB].index)]

                        for t in t_gr:
                            df_res_dict[topic][frac_type][VOCAB].loc[t, t_gr] += scores_group[k]
                            df_count_dict[topic][frac_type][VOCAB].loc[t, t_gr] += 1

                        if "cl_" not in col:
                            df_res_dict[topic][frac_type][LABELS].loc[t_gr, col] += scores_labels[k]
                            df_count_dict[topic][frac_type][LABELS].loc[t_gr, col] += 1

    for topic in list(df_res_dict):
        orig_frac = [frac_type for frac_type in list(df_res_dict[topic]) if "+" not in frac_type]
        frac_groups = {}
        for frac in orig_frac:
            frac_groups[frac] = [frac_type for frac_type in list(df_res_dict[topic]) if frac.split("_")[-1] in frac_type
                                 and frac.split("_")[0] in frac_type and frac_type != frac]

        for frac_shares, fracs in frac_groups.items():
            shared_res_dict = df_res_dict[topic][frac_shares]
            shared_count_dict = df_count_dict[topic][frac_shares]
            for frac_type in fracs:
                logger.info(topic + " " + frac_shares + " " + frac_type)
                if frac_type not in all_scores[topic]:
                    all_scores[topic][frac_type] = []
                all_scores[topic][frac_type].extend(all_scores[topic][frac_shares])

                df_res_dict[topic][frac_type][VOCAB] += df_res_dict[topic][frac_shares][VOCAB].copy().values
                df_count_dict[topic][frac_type][VOCAB] += shared_count_dict[VOCAB].values

                if HC not in frac_type:
                    df_res_dict[topic][frac_type][LABELS]  += shared_res_dict[LABELS].values
                    df_count_dict[topic][frac_type][LABELS] += shared_count_dict[LABELS].values

    coders_num = {"processing": 4, "databases": 3, "softwaredev": 3, "travel": 2}

    def _expand_chain_count(set_to_check, round_, frac, recursion_level=1):
        output = set()
        for w in set_to_check:
            if round_ == 0:
                if w in checked_vocab:
                    continue
            else:
                if w in checked_vocab - set_to_check:
                    continue
            suitable_df = df_count_dict[topic][frac][VOCAB][df_count_dict[topic][frac][VOCAB][w] >= 2 * coders_num[topic]]
            checked_vocab.add(w)
            diff = {v for v in set(suitable_df.index) - set_to_check if v not in checked_vocab}
            output = output.union(diff)

        if len(output) > 1 and recursion_level <= 1:
            output = output.union(_expand_chain_count(output, round_=round_, frac=frac, recursion_level=recursion_level+1))

        return output.union(set_to_check)

    count_test_dict_modif = {}
    j = 0
    for topic in list(df_count_dict):
        frac_dict = {}
        count_test_dict_modif[topic] = {}

        for frac in list(df_count_dict[topic]):
            if "+" not in frac:
                continue
            else:
                classes_list = []
                checked_vocab = set()
                for word in list(df_count_dict[topic][frac][VOCAB].mean(axis=0).sort_values(ascending=False).index):
                    if word in checked_vocab:
                        continue

                    chain = list(_expand_chain_count({word}, frac=frac, round_=0))

                    for w in chain:
                        checked_vocab.add(w)

                    if len(chain) < 5:
                        continue

                    potential_labels_df = df_count_dict[topic][frac][LABELS].loc[chain]
                    pot_labels_dict = {}
                    for col in list(potential_labels_df.columns):
                        values = [v for v in potential_labels_df[col].values if v > 0]
                        if len(values) < 2:
                            continue
                        pot_labels_dict[col] = np.mean(values)

                    label = max(pot_labels_dict, key=pot_labels_dict.get) if len(pot_labels_dict) else "cl_" + str(j)
                    j += 1
                    label_score = max(list(pot_labels_dict.values())) if len(pot_labels_dict) else 0

                    classes_list.append({"label": label,
                                    "label_score": label_score,
                                    "terms": chain})
            if not len(classes_list):
                continue
            frac_dict[frac] = classes_list

            count_test_dict_modif[topic][frac] = {}
            for cl in frac_dict[frac]:
                if cl["label"] not in count_test_dict_modif[topic][frac]:
                    if not len(cl["terms"]):
                        continue
                    count_test_dict_modif[topic][frac][cl["label"]] = []
                count_test_dict_modif[topic][frac][cl["label"]] += cl["terms"]

    count_test_reformat = {}
    for topic in list(count_test_dict_modif):
        count_test_reformat[topic] = {}
        for frac, vals in count_test_dict_modif[topic].items():
            count_test_reformat[topic][frac] = []
            for k, v in vals.items():
                count_test_reformat[topic][frac].append({"label": k, "terms": v})

    terms_groups_dict_reformat = {}
    for topic in list(terms_groups_dict):
        terms_groups_dict_reformat[topic] = {}
        for frac, vals in terms_groups_dict[topic].items():
            terms_groups_dict_reformat[topic][frac] = []
            for k, v in vals.items():
                terms_groups_dict_reformat[topic][frac].append({"label": k, "terms": v})

    with open(os.path.join(SILVER_PATH, date, "silver_dfs.pickle"), "rb") as file:
        silver_dfs = pickle.load(file)

    def calc_stats_silver(data_dict):
        stats_dict = {}
        stats_dict_general = {}

        for topic in list(data_dict):
            for frac, vals_ in data_dict[topic].items():
                res_dict = {"avg_terms_score": [], "avg_label_score": []}
                for val in vals_:
                    terms_avgscores = []
                    terms = []
                    for tr in val["terms"]:
                        if tr.strip() in list(silver_dfs[topic][VOCAB].index):
                            terms.append(tr.strip())
                        elif tr in list(silver_dfs[topic][VOCAB].index):
                            terms.append(tr)

                    for t in terms:
                        vals = [v for v in silver_dfs[topic][VOCAB].loc[t, terms].values if v > 0]
                        avg_val = np.mean(vals) if len(vals) else 0
                        terms_avgscores.append(avg_val)

                    if "cl_" in val["label"]:
                        label_avgscore = 0
                    else:
                        vals = [v for v in silver_dfs[topic][LABELS].loc[terms, val["label"]].values if v > 0]
                        label_avgscore = np.mean(vals) if len(vals) else 0

                    res_dict["avg_terms_score"].append(np.mean(terms_avgscores) if len(terms_avgscores) else 0)
                    res_dict["avg_label_score"].append(label_avgscore)
                key = " ".join([topic, frac])
                stats_dict[key] = res_dict
                stats_dict_general[key] = {
                    "avg_terms_score": np.mean(res_dict["avg_terms_score"]) if len(res_dict["avg_terms_score"]) else 0,
                    "avg_label_score": np.mean(res_dict["avg_label_score"]) if len(res_dict["avg_label_score"]) else 0}
        return stats_dict_general

    silver_scores_count = calc_stats_silver(count_test_reformat)
    silver_scores_orig = calc_stats_silver(terms_groups_dict_reformat)
    silver_scores_count.update(silver_scores_orig)

    df_eval_results = pd.DataFrame(silver_scores_count).T.sort_index()
    df_eval_results["topic"] = [""] * len(df_eval_results)
    df_eval_results["appr"] = [""] * len(df_eval_results)
    df_eval_results["frac"] = [""] * len(df_eval_results)
    df_eval_results["categories"] = [0] * len(df_eval_results)
    df_eval_results["terms"] = [0] * len(df_eval_results)
    df_eval_results["size"] = [0] * len(df_eval_results)
    df_eval_results["bins"] = [""] * len(df_eval_results)
    df_eval_results["average_score"] = [0] * len(df_eval_results)

    for config, row in df_eval_results.iterrows():
        frac = config.split(" ")[-1]
        topic = config.split(" ")[0]
        df_eval_results.loc[config, "topic"] = topic
        frac_num = frac.split("_")[-1]
        df_eval_results.loc[config, "frac"] = frac_num
        df_eval_results.loc[config, "appr"] = "_".join(frac.split("_")[:-1])
        type_, count_ = np.unique(all_scores[topic][frac], return_counts=True)
        df_eval_results.loc[config, "bins"] = str({k: v for k, v in zip(list(type_), list(count_))})
        try:
            df_eval_results.loc[config, "categories"] = len(count_test_reformat[topic][frac])
            extr_terms = {v for vals in count_test_reformat[topic][frac] for v in vals["terms"]}
        except KeyError:
            df_eval_results.loc[config, "categories"] = len(terms_groups_dict_reformat[topic][frac])
            extr_terms = {v for vals in terms_groups_dict_reformat[topic][frac] for v in vals["terms"]}

        df_eval_results.loc[config, "terms"] = len(extr_terms)
        df_eval_results.loc[config, "size"] = df_eval_results.loc[config, "terms"] / df_eval_results.loc[
            config, "categories"]
        if df_eval_results.loc[config, "avg_label_score"] > 0:
            df_eval_results.loc[config, "average_score"] = np.mean([df_eval_results.loc[config, "avg_terms_score"],
                                                                df_eval_results.loc[config, "avg_label_score"]])

    silver_stats_df = pd.read_csv(os.path.join(SILVER_PATH, date, "silver_dataset_stats.csv"), index_col=[0])

    all_stats_df = pd.DataFrame(df_eval_results.append(silver_stats_df), columns=["topic", "appr", "frac", "categories", "terms", "size", "bins", "avg_terms_score",
                               "avg_label_score", "average_score"])
    all_stats_df.sort_values(by=["topic", "appr", "frac"], ascending=[True, False, True], inplace=True)
    all_stats_df.to_csv(os.path.join(EVAL_PATH, date, "all_stats.csv"), index=True)

    df_eval_results["max"] = df_eval_results.groupby(by=["topic"], as_index=False)["average_score"].transform(max)
    best_config_df = df_eval_results[df_eval_results["max"] == df_eval_results["average_score"]]

    voting_df = pd.DataFrame()
    i = 0
    for index, row in best_config_df.iterrows():
        cats = count_test_reformat[row["topic"]]["_".join([row["appr"], row["frac"]])]
        for cat in cats:
            voting_df = voting_df.append(pd.DataFrame({"topic": row["topic"], "label": cat["label"],
                                                       "terms": ", ".join(cat["terms"])}, index=[i]))
            i += 1
    voting_df.to_csv(os.path.join(EVAL_PATH, date, "best_categories.csv"), index=False, encoding="utf-8")
    logger.info("Statistics and qualitative results are saved to " + os.path.join(EVAL_PATH, date))


if __name__ == '__main__':
    count_datasets("2020-09-08")