import pandas as pd
import json
import re
import numpy as np
import pickle
from config import *


def build_silver_dataset(date: str):

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

    appr_tab_dict = {HC: [1,3,5,7],
                     ANEA: [2,4,6,8]}

    used_fractions = {list(allowed_areas)[list(allowed_areas.values()).index(topic)]: [int(l) for l in list(runs["run_params"])]
                      for topic, runs in results_dict.items()}
    EVAL_RESPATH = os.path.join(EXCEL_PATH, date)
    eval_folders = os.listdir(EVAL_RESPATH)

    df_res_dict = {}
    df_count_dict = {}

    for topic in list(all_vocab):
        vocab_df = pd.DataFrame(np.zeros((len(all_vocab[topic]), len(all_vocab[topic]))), columns=all_vocab[topic],
                                index=all_vocab[topic])
        label_df = pd.DataFrame(np.zeros((len(all_vocab[topic]), len(all_labels[topic]))), columns=all_labels[topic],
                                index=all_vocab[topic])
        df_res_dict[topic] = {VOCAB: vocab_df, LABELS: label_df}
        df_count_dict[topic] = {VOCAB: vocab_df.copy(), LABELS: label_df.copy()}

    all_scores = {"processing": [],
                   "softwaredev": [],
                   "databases": [],
                   "travel": []}

    all_scores_labels = {"processing": [],
                   "softwaredev": [],
                   "databases": [],
                   "travel": []}

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
                terms_groups_dict[topic][appr] = {}

                for tab, frac in zip(tabs, used_fractions[ds]):
                    logger.info(appr + " " + str(frac))
                    eval_sheet_df_ = pd.read_excel(os.path.join(eval_folder_full, ds_file), sheet_name=tab)
                    eval_sheet_df = eval_sheet_df_.iloc[:, list(range(3, len(eval_sheet_df_.columns), 2))]
                    scores_group = eval_sheet_df.iloc[2].fillna(6).values
                    all_scores[topic].extend(scores_group)
                    scores_labels = eval_sheet_df.iloc[4].values

                    for v in [v for v in scores_labels if v is not np.nan]:
                        try:
                            all_scores_labels[topic].append(int(v))
                        except ValueError:
                            continue

                    terms_df = eval_sheet_df.iloc[5:, :]
                    terms_df.columns = eval_sheet_df.iloc[0] if not np.isnan(np.sum(scores_labels)) else ["cl_"+str(j)
                                                                              for j in range(len(eval_sheet_df.columns))]
                    terms = {col: [t for t in terms_df[col] if t is not np.nan and type(t) == str]
                             for col in list(terms_df.columns)}
                    terms_groups_dict[topic][appr][frac] = terms

                    for k, (col, t_gr) in enumerate(terms.items()):
                        t_gr = [v.strip() for v in t_gr if v in list(df_res_dict[topic][VOCAB].index)]

                        for t in t_gr:
                            df_res_dict[topic][VOCAB].loc[t, t_gr] += scores_group[k]
                            df_count_dict[topic][VOCAB].loc[t, t_gr] += 1

                        if "cl_" not in col:
                            df_res_dict[topic][LABELS].loc[t_gr, col] += scores_labels[k]
                            df_count_dict[topic][LABELS].loc[t_gr, col] += 1
    df_norm_dict = {}
    for topic in list(df_res_dict):
        vocab_df_ = pd.DataFrame(np.zeros((len(df_res_dict[topic][VOCAB].index), len(df_res_dict[topic][VOCAB].columns))),
                                index=df_res_dict[topic][VOCAB].index, columns=df_res_dict[topic][VOCAB].columns)
        labels_df_ = pd.DataFrame(np.zeros((len(df_res_dict[topic][LABELS].index), len(df_res_dict[topic][LABELS].columns))),
                                index=df_res_dict[topic][LABELS].index, columns=df_res_dict[topic][LABELS].columns)

        for index in list(df_res_dict[topic][VOCAB].index):
            for col in list(df_res_dict[topic][VOCAB].columns):
                if not df_res_dict[topic][VOCAB].loc[index, col]:
                    continue
                vocab_df_.loc[index, col] = df_res_dict[topic][VOCAB].loc[index, col]/df_count_dict[topic][VOCAB].loc[index, col]

            for col in list(df_res_dict[topic][LABELS].columns):
                if not df_res_dict[topic][LABELS].loc[index, col]:
                    continue
                labels_df_.loc[index, col] = df_res_dict[topic][LABELS].loc[index, col]/df_count_dict[topic][LABELS].loc[index, col]

        df_norm_dict[topic] = {VOCAB: vocab_df_, LABELS: labels_df_}

    topic_threshold = {}
    topic_threshold_delta = {}

    for topic in list(all_scores):
        unique, counts = np.unique(all_scores[topic], return_counts=True)
        ids = [6,7,8]
        diff = [abs(c - counts[i + 1]) for i, c in enumerate(counts[:-1])]
        max_val = ids[int(np.argmax(counts[min(ids): max(ids) + 1]))]
        if diff[max_val - 1] <= 3:
            thr_val = np.mean([max_val, max_val - 1])
        else:
            thr_val = max_val
        # strict threshold
        topic_threshold[topic] = thr_val
        # more loose threshold
        ids_loose = [5, 6, 7, 8]
        max_val_loose = ids_loose[int(np.argmax(counts[min(ids_loose): max(ids_loose) + 1]))]
        if max_val_loose != max_val:
            topic_threshold_delta[topic] = 1
        else:
            topic_threshold_delta[topic] = 0

    # processing (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64), array([15, 17, 24, 36, 44, 55, 49, 50, 46, 52], dtype=int64))
    # softwaredev (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64), array([10, 17, 36, 40, 44, 55, 59, 62, 40,  6], dtype=int64))
    # databases (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64), array([13,  9, 28, 27, 25, 38, 51, 68, 66, 68], dtype=int64))
    # travel (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int64), array([ 6, 14,  8, 13, 16,  8, 24, 42, 62, 55], dtype=int64))

    def _expand_chain(set_to_check, round_, recursion_level=1):
        output = set()
        thr = topic_threshold[topic] if round_ == 0 else topic_threshold[topic] - topic_threshold_delta[topic]
        for w in set_to_check:
            if round_ == 0:
                if w in checked_vocab:
                    continue
            else:
                if w in checked_vocab - set_to_check:
                    continue
            suitable_df = df_norm_dict[topic][VOCAB][df_norm_dict[topic][VOCAB][w] >= thr]
            checked_vocab.add(w)
            diff = {v for v in set(suitable_df.index) - set_to_check if v not in checked_vocab}
            output = output.union(diff)

        if len(output) > 1 and recursion_level <= 1:
            output = output.union(_expand_chain(output, round_=round_, recursion_level=recursion_level+1))
        return output.union(set_to_check)

    silver_annot_dict = {}
    i = 0
    for topic in list(df_norm_dict):
        checked_vocab = set()
        classes_list = []
        for word in list(df_norm_dict[topic][VOCAB].mean(axis=0).sort_values(ascending=False).index):
            if word in checked_vocab:
                continue

            chain = list(_expand_chain({word}, round_=i))
            for w in chain:
                checked_vocab.add(w)

            if len(chain) < 4:
                continue

            potential_labels_df = df_norm_dict[topic][LABELS].loc[chain]
            pot_labels_dict = {}

            for col in list(potential_labels_df.columns):
                values = [v for v in potential_labels_df[col].values if v > 0]
                if not len(values):
                    continue
                pot_labels_dict[col] = np.mean(values)
            if not len(pot_labels_dict):
                continue
            label = max(pot_labels_dict, key=pot_labels_dict.get)
            label_score = max(list(pot_labels_dict.values()))
            classes_list.append({"label": label,
                            "label_score": label_score,
                            "terms": chain})
        silver_annot_dict[topic] = classes_list

    silver_annot_dict_modif = {}
    i = 1
    for topic in list(df_norm_dict):
        checked_vocab = {v for vals in silver_annot_dict[topic] for v in vals["terms"]}
        classes_list = []
        for vals in silver_annot_dict[topic]:

            chain = list(_expand_chain(set(vals["terms"]), round_=i))
            for w in chain:
                checked_vocab.add(w)

            if len(chain) == 1:
                continue

            potential_labels_df = df_norm_dict[topic][LABELS].loc[chain]
            pot_labels_dict = {}

            for col in list(potential_labels_df.columns):
                values = [v for v in potential_labels_df[col].values if v > 0]
                if len(values) < 4:
                    continue
                pot_labels_dict[col] = np.mean(values)

            if not len(pot_labels_dict):
                continue
            label = max(pot_labels_dict, key=pot_labels_dict.get)
            label_score = max(list(pot_labels_dict.values()))
            classes_list.append({"label": label,
                                 "label_score": label_score,
                                 "terms": chain})

            silver_annot_dict_modif[topic] = {}
            for cl in classes_list:
                if cl["label"] not in silver_annot_dict_modif:
                    silver_annot_dict_modif[topic][cl["label"]] = []
                silver_annot_dict_modif[topic][cl["label"]] += cl["terms"]

    silver_annot_reformat = {}
    for topic, vals in silver_annot_dict_modif.items():
        silver_annot_reformat[topic] = []
        for k, v in vals.items():
            silver_annot_reformat[topic].append({"label": k, "terms": v})

    true_lists = {}
    for topic, classes in silver_annot_reformat.items():
        true_lists[topic] = [v for vals in classes for v in vals["terms"]]

    stats_dict = {}
    stats_dict_general = {}

    for topic, vals in silver_annot_reformat.items():
        res_dict = {"avg_terms_score": [], "avg_label_score": []}

        for val in vals:
            terms_avgscores = []

            for t in val["terms"]:
                avg_val = np.mean([v for v in df_norm_dict[topic][VOCAB].loc[t, val["terms"]].values if v > 0])
                terms_avgscores.append(avg_val)

            label_avgscore = np.mean([v for v in df_norm_dict[topic][LABELS].loc[val["terms"],
                                                                                 val["label"]].values if v > 0])
            res_dict["avg_terms_score"].append(np.mean(terms_avgscores))
            res_dict["avg_label_score"].append(label_avgscore)

        stats_dict[topic] = res_dict
        stats_dict_general[topic] = {"avg_terms_score": np.mean(res_dict["avg_terms_score"]),
                                     "avg_label_score": np.mean(res_dict["avg_label_score"])}

    silver_folder = os.path.join(SILVER_PATH, date)
    df_eval_results = pd.DataFrame(stats_dict_general).T.sort_index()
    df_eval_results["topic"] = [""] * len(df_eval_results)
    df_eval_results["appr"] = [""] * len(df_eval_results)
    df_eval_results["frac"] = ["--"] * len(df_eval_results)
    df_eval_results["categories"] = [0] * len(df_eval_results)
    df_eval_results["terms"] = [0] * len(df_eval_results)
    df_eval_results["size"] = [0] * len(df_eval_results)
    df_eval_results["bins"] = [""] * len(df_eval_results)
    df_eval_results["average_score"] = [0] * len(df_eval_results)

    for topic, row in df_eval_results.iterrows():
        df_eval_results.loc[topic, "topic"] = topic
        df_eval_results.loc[topic, "appr"] = "silver"
        type_, count_ = np.unique(all_scores[topic], return_counts=True)
        df_eval_results.loc[topic, "bins"] = str({k: v for k, v in zip(list(type_), list(count_))})
        df_eval_results.loc[topic, "categories"] = len(silver_annot_reformat[topic])
        extr_terms = {v for vals in silver_annot_reformat[topic]for v in vals["terms"]}
        df_eval_results.loc[topic, "terms"] = len(extr_terms)
        df_eval_results.loc[topic, "size"] = df_eval_results.loc[topic, "terms"]/df_eval_results.loc[topic, "categories"]
        df_eval_results.loc[topic, "average_score"] = np.mean([df_eval_results.loc[topic, "avg_terms_score"],
                                                                df_eval_results.loc[topic, "avg_label_score"]])
    df_eval_results = pd.DataFrame(df_eval_results, columns=["topic", "appr", "frac", "categories", "terms", "size", "bins", "avg_terms_score",
                               "avg_label_score", "average_score"])

    df_eval_results.to_csv(os.path.join(silver_folder, "silver_dataset_stats.csv"), index=True)

    with open(os.path.join(silver_folder, "silver_dfs.pickle"), "wb") as file:
        pickle.dump(df_norm_dict, file)

    with open(os.path.join(silver_folder, "true_categories.json"), "w") as file:
        json.dump(silver_annot_reformat, file)

    with open(os.path.join(silver_folder, "true_lists.json"), "w") as file:
        json.dump(true_lists, file)

    silver_df = pd.DataFrame()
    for i, (topic, cats) in enumerate(silver_annot_reformat.items()):
        for cat in cats:
            silver_df = silver_df.append(pd.DataFrame({"topic": topic, "label": cat["label"],
                                                       "terms": ", ".join(cat["terms"])}, index=[i]))
    silver_df.to_csv(os.path.join(silver_folder, "silver_categories.csv"), index=False, encoding="utf-8")

    for topic, matrixes in df_norm_dict.items():
        for matr_name, df_vals in matrixes.items():
            df_vals.to_csv(os.path.join(silver_folder, "_".join(["scores", topic, matr_name]) + ".csv"), encoding="utf-8")

    logger.info("Stats and silver datasets are saved.")


if __name__ == '__main__':
    build_silver_dataset("2020-09-08")
