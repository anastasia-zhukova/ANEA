import numpy as np
import pandas as pd
import copy
from graph.graph import Graph
import math
from sklearn.metrics.pairwise import cosine_similarity as cs

MAX_LEVELS = 3
MIN_WORDS = 5
MIN_MEAN_SIM = 0.2
MIN_NAME_SIM = 0.3
MIN_LENGTH = 1
NAME_SIM = "name_sim"
MEAN_SIM = "mean_sim"
SUM_SIM = "sum_sim"
IMPACT_SCORE = "impact_score"
SIZE = "size"
LENGTH = "length"
WORDS = "words"


class NEASemanticGraph:
    def __init__(self, graph):
        self.graph = graph
        self.model = graph.model
        self.graph.model = None
        self.label_dict = None
        self.dist_df = None
        self.repr_dict = {}
        self.vector_df = None

    def extract_labels(self):
        # potential categories
        label_dict_short = self._collect_labels()
        self._build_vector_df()
        sum_full_df, sum_df = self._build_init_table(label_dict_short)

        # preselection of the categories
        best_mean = sum_df[sum_df[MEAN_SIM] >= MIN_MEAN_SIM][sum_df[NAME_SIM] >= MIN_NAME_SIM][sum_df[LENGTH] >= MIN_LENGTH]

        # form main categories
        best_reprs = self._larger_reprs(best_mean)
        best_reprs_2 = self._overlap_repr(best_reprs)
        best_reprs_body = self._repr_cleaning(best_reprs_2)
        final_entities = self._add_border_terms(best_reprs_body)
        return final_entities

    def _build_vector_df(self):
        self.vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for label, words in self.label_dict.items():
            vector = self.model.get_vector(label)
            self.vector_df = self.vector_df.append(pd.DataFrame([vector],
                             columns=["d" + str(i) for i in range(self.model.vector_size)],
                             index=[label]))
            for w in words:
                if w in list(self.vector_df.index):
                    continue
                vector = self.model.get_vector(w)
                self.vector_df = self.vector_df.append(pd.DataFrame([vector],
                                                                    columns=["d" + str(i) for i in
                                                                             range(self.model.vector_size)],
                                                                    index=[w]))

    def _collect_labels(self):

        def __get_nodes(leaf):
            if leaf in self.graph.graph_down_top:
                levels = [self.graph.graph_down_top[leaf]]
            else:
                return set()

            checked = {leaf}
            i = 0
            while i < MAX_LEVELS and len(levels[i]):
                new_level = set()
                for node in levels[i]:
                    if node in checked:
                        continue

                    if node != leaf:
                        self.dist_df.loc[leaf, node] = i + 1
                    else:
                        self.dist_df.loc[leaf, leaf] = 0

                    if node in self.graph.graph_down_top:
                        new_level = new_level.union(set(self.graph.graph_down_top[node]))
                    checked.add(node)
                for new_node in copy.copy(new_level):
                    if new_node in checked:
                        new_level.remove(new_node)
                        continue
                    try:
                        if new_node != leaf:
                            self.dist_df.loc[leaf, new_node] = i + 2
                        else:
                            self.dist_df.loc[leaf, leaf] = 0
                    except KeyError:
                        if new_node not in list(self.dist_df.columns):
                            new_level.remove(new_node)
                        continue
                levels.append(list(new_level))
                i += 1
            return levels

        leaves = []
        nodes = []
        for node in list(self.graph.all_nodes.values()):
            if node.term_id is not None:
                leaves.append(node.word)
            else:
                nodes.append(node.word)

        self.dist_df = pd.DataFrame(np.zeros((len(leaves), len(set(nodes + leaves)))), index=set(leaves),
                                    columns=set(nodes + leaves))
        label_dict = {}
        for leaf in leaves:
            labels = list(set().union(*__get_nodes(leaf)))
            for label in labels:
                if label not in label_dict:
                    label_dict[label] = []
                label_dict[label].append(leaf)

        self.label_dict = {k: v for k, v in sorted(label_dict.items(), reverse=True, key=lambda x: len(x[1]))}

        label_dict_short = {k: v for k, v in self.label_dict.items() if len(v) >= MIN_WORDS}
        return label_dict_short

    def _build_init_table(self, label_dict_short):
        label_dict_dist = {}

        for k in list(label_dict_short):
            if k in list(self.dist_df.columns):
                arr = [v for v in self.dist_df[k].values if v > 0]
                if len(arr):
                    label_dict_dist[k] = np.mean(arr)

        sum_df = pd.DataFrame(columns=[SIZE, LENGTH, MEAN_SIM, NAME_SIM, SUM_SIM, IMPACT_SCORE, WORDS])
        for node, words in label_dict_short.items():
            vector_df = self.vector_df.loc[words]
            try:
                mean_ = np.mean(cs(vector_df))
                name_ = cs([np.mean(vector_df.values, axis=0)], [self.vector_df.loc[node].values])[0][0]
                sum_df = sum_df.append(pd.DataFrame({
                    SIZE: len(words),
                    LENGTH: label_dict_dist[node] if node in label_dict_dist else 1, # else 1? otherwise the impact score will be 0
                    MEAN_SIM: mean_,
                    NAME_SIM: name_,
                    SUM_SIM: mean_ + name_,
                    IMPACT_SCORE: 0,
                    WORDS: ", ".join(words)
                }, index=[node]))
                sum_df.loc[node, IMPACT_SCORE] = math.log(sum_df.loc[node, SIZE], 2) * sum_df.loc[node, MEAN_SIM] * \
                                                 sum_df.loc[node, NAME_SIM] * sum_df.loc[node, SUM_SIM] * \
                                                 sum_df.loc[node, LENGTH]
            except ValueError:
                continue

        sum_df.sort_values([SIZE, WORDS, NAME_SIM], ascending=[False, False, False], inplace=True)
        groups = sum_df.groupby(WORDS, as_index="False")
        repr_dict = {v[0]: list(v[1:]) for k, v in groups.groups.items() if len(v) > 1}
        if len(self.repr_dict) == 0:
            self.repr_dict = repr_dict
        else:
            self.repr_dict.update(repr_dict)
        sum_short_df = sum_df.drop_duplicates(subset=[WORDS])
        sum_short_df.sort_values([NAME_SIM, MEAN_SIM, SIZE], ascending=[False, False, False], inplace=True)
        # return sum_df.sort_values(SIZE, ascending=False), sum_short_df.sort_values(SIZE, ascending=False)
        return sum_df.sort_values(SIZE, ascending=False), sum_short_df.sort_values(SIZE, ascending=False)

    def _larger_reprs(self, best_mean):
        # find better representatives bigger in size
        best_mean.sort_values(SIZE, ascending=False, inplace=True)

        # a collection of the better representatives
        overlaps_global = {}
        for m, (index_1, row_1) in enumerate(best_mean.iterrows()):
            overlaps = {}
            words_1 = set(row_1[WORDS].split(", "))

            # find overlaps of smaller groups with larger
            for n, (index_2, row_2) in enumerate(best_mean.iterrows()):
                if n < m:
                    continue

                words_2 = set(row_2[WORDS].split(", "))
                overlap = words_1.intersection(words_2)
                if len(overlap) == len(words_1):
                    # overlaps[index_2] = row_2[LENGTH] * row_2[MEAN_SIM] * row_2[NAME_SIM]
                    overlaps[index_2] = row_2[IMPACT_SCORE]

            overlaps = {k: v for k, v in sorted(overlaps.items(), reverse=True, key=lambda x: x[1])}
            if len(overlaps) == 0:
                continue

            # select a more general representative with the largest impact score
            head = list(overlaps)[0]
            if head not in overlaps_global:
                overlaps_global[head] = {index_1: overlaps[index_1]}
            else:
                overlaps_global[head].update({index_1: overlaps[index_1]})

        best_reprs = best_mean.loc[list(overlaps_global)]
        for k, v in overlaps_global.items():
            if len(v) > 1:
                self.repr_dict.update({k: list(set(v.keys()) - {k})})

        return best_reprs.sort_values(MEAN_SIM, ascending=False)

    def _overlap_repr(self, best_reprs):
        # find best representatives with significant overlap (>= 50%)
        best_reprs.sort_values(IMPACT_SCORE, ascending=False, inplace=True)

        overlap_sim_df = pd.DataFrame(np.zeros((len(best_reprs), len(best_reprs))), index=list(best_reprs.index),
                                      columns=list(best_reprs.index))
        for index_1, rows_1 in best_reprs.iterrows():
            overlaps = {}
            words_1 = set(rows_1[WORDS].split(", "))

            for index_2, row_2 in best_reprs.iterrows():
                words_2 = set(row_2[WORDS].split(", "))
                overlap = words_1.intersection(words_2)
                if len(overlap):
                    overlaps[index_2] = len(overlap)

            overlaps = {k: v for k, v in sorted(overlaps.items(), reverse=True, key=lambda x: x[1])}
            overlap_df = best_reprs.loc[[k for k, v in overlaps.items() if v >= len(words_1) / 2]]
            if not len(overlap_df):
                print(index_1)
                continue
            impact = []
            for index, row in overlap_df.iterrows():
                # impact.append((row[MEAN_SIM] * row[NAME_SIM] * row[SIZE] * row[SUM_SIM]) / row[LENGTH])
                impact.append(row[IMPACT_SCORE])
            overlap_df[IMPACT_SCORE] = impact
            overlap_sim_df.loc[index_1, str(overlap_df[IMPACT_SCORE].idxmax())] = 1

        to_remove = []
        for index in list(overlap_sim_df.index):
            destination = overlap_sim_df.loc[index].idxmax()
            if overlap_sim_df.loc[destination, destination] == 1 and index != destination:
                to_remove.append(index)
                if destination not in self.repr_dict:
                    self.repr_dict[destination] = []
                self.repr_dict[destination].append(index)

        best_reprs_2 = best_reprs.loc[list(set(overlap_sim_df.index) - set(to_remove))]
        return best_reprs_2

    def _repr_cleaning(self, best_reprs_2):
        # resolve minor conflicting words that fall info multiple classes

        # best_reprs_2.sort_values(IMPACT_SCORE, ascending=False, inplace=True)

        words_overlap_dict = {}
        repr_dict_clean = {index: set(row[WORDS].split(", ")) for index, row in best_reprs_2.iterrows()}
        sim_words_df = pd.DataFrame(columns=list(best_reprs_2.index), index=list(best_reprs_2.index))

        for n, (index_1, row_1) in enumerate(best_reprs_2.iterrows()):
            words_1 = set(row_1[WORDS].split(", "))
            for m, (index_2, row_2) in enumerate(best_reprs_2.iterrows()):
                if m == n:
                    continue
                words_2 = set(row_2[WORDS].split(", "))
                intersect = set(words_1).intersection(set(words_2))
                for w in intersect:
                    if w not in words_overlap_dict:
                        words_overlap_dict[w] = set()
                    words_overlap_dict[w] = words_overlap_dict[w].union({words_1, words_2})

                if len(intersect):
                    sim_words_df.loc[index_1, index_2] = ", ".join(intersect)
                    sim_words_df.loc[index_2, index_1] = ", ".join(intersect)
                else:
                    sim_words_df.loc[index_1, index_2] = ""
                    sim_words_df.loc[index_2, index_1] = ""

                repr_dict_clean[index_1] = repr_dict_clean[index_1] - intersect
                repr_dict_clean[index_2] = repr_dict_clean[index_2] - intersect

        clean_df, clean_df_short = self._build_init_table(repr_dict_clean)

        clean_df_short.sort_values(IMPACT_SCORE, ascending=False, inplace=True)

        for n, (index, row) in enumerate(clean_df_short.iterrows()):
            conflicting_words = set().union(*[v.split(", ") for v in sim_words_df.loc[index].values])
            for w in conflicting_words:
                if w not in words_overlap_dict:
                    continue
                candidates = words_overlap_dict[w]
                cand_df = pd.DataFrame(columns=[MEAN_SIM, NAME_SIM, SUM_SIM, LENGTH])



        # sim_df = pd.DataFrame(np.zeros((len(best_reprs_2), len(best_reprs_2))), columns=list(best_reprs_2.index),
        #                       index=list(best_reprs_2.index))


        # for n, (index_1, row_1) in enumerate(best_reprs_2.iterrows()):
        #     words_1 = set(row_1[WORDS].split(", "))
        #     for m, (index_2, row_2) in enumerate(best_reprs_2.iterrows()):
        #         words_2 = set(row_2[WORDS].split(", "))
        #         if m > n:
        #             continue
        #         intersect = set(words_1).intersection(set(words_2))
        #         if len(intersect):
        #             sim_df.loc[str(index_1), str(index_2)] = len(intersect)
        #             sim_df.loc[str(index_2), str(index_1)] = len(intersect)
        #             sim_words_df.loc[index_1, index_2] = ", ".join(intersect)
        #             sim_words_df.loc[index_2, index_1] = ", ".join(intersect)
        #         else:
        #             sim_words_df.loc[index_1, index_2] = ""
        #             sim_words_df.loc[index_2, index_1] = ""
        #
        # selected_label_dict = {index: row[WORDS].split(", ") for index, row in best_reprs_2.iterrows()}
        # best_reprs_2.sort_values(MEAN_SIM, ascending=True, inplace=True)
        #
        # repr_dict = {}
        #
        # # to_remove = []
        # for i, (index, row) in enumerate(best_reprs_2.iterrows()):
        #     # if i > 10:
        #     #     break
        #     words_1 = set(selected_label_dict[index])
        #     inters_dict = {}
        #     for key, val in selected_label_dict.items():
        #         if index == key:
        #             continue
        #         words_2 = set(val)
        #         intersect = words_1.intersection(words_2)
        #         if len(intersect):
        #             inters_dict[key] = intersect
        #     if len(inters_dict) == 0:
        #         # best_reprs_body = best_reprs_body.append(best_reprs_2.loc[index])
        #         repr_dict[index] = row[WORDS].split(", ")
        #         continue
        #     try:
        #         local_df = best_reprs_2[best_reprs_2[MEAN_SIM] > row[MEAN_SIM]].loc[
        #             list(inters_dict.keys())].sort_values(MEAN_SIM, ascending=False).dropna()
        #     except KeyError:
        #         continue
        #     new_composition = words_1
        #     for cand in list(local_df.index):
        #         new_composition = new_composition - inters_dict[cand]
        #         selected_label_dict[index] = new_composition
        #     repr_dict[index] = new_composition

        repr_dict_short = {k: v for k, v in repr_dict_clean.items() if len(v) >= MIN_WORDS}
        best_reprs_body, best_reprs_body_short = self._build_init_table(repr_dict_short)
        return best_reprs_body_short


    def _add_border_terms(self, best_reprs_body):
        # assign the remaining terms to the bins
        title_mean_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for word in list(best_reprs_body.index):
            vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
            words_ = set(best_reprs_body.loc[word, WORDS].split(", "))
            for w in words_:
                vector = self.model.get_vector(w)
                vector_df = vector_df.append(
                    pd.DataFrame([vector],
                                 columns=["d" + str(i) for i in range(self.model.vector_size)],
                                 index=[w])) if vector is not None else vector_df
            mean_vector = np.mean(vector_df.values, axis=0)
            title_mean_df = title_mean_df.append(pd.DataFrame([mean_vector], index=[word],
                                                              columns=["d" + str(i) for i in range(self.model.vector_size)]))

        all_words = set().union(*[v.split(", ") for k, v in best_reprs_body[WORDS].items()])
        all_leaves = [k for k, v in self.graph.all_nodes.items() if v.term_id is not None]
        not_sorted = set(all_leaves) - all_words

        titles_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        for w in list(best_reprs_body.index):
            vector = self.model.get_vector(w)
            titles_df = titles_df.append(
                pd.DataFrame([vector],
                             columns=["d" + str(i) for i in range(self.model.vector_size)],
                             index=[w])) if vector is not None else titles_df

        newly_added = {}
        for i, w in enumerate(not_sorted.union(list(self.graph.not_in_graph))):
            sim_comp_new_df = pd.DataFrame(columns=["sim_name", "sim_vals"])
            vector = self.model.get_vector(w)
            if vector is None:
                continue
            for t, row in titles_df.iterrows():
                sim = cs([row], [vector])
                sim_val = cs([title_mean_df.loc[t].values], [vector])
                sim_comp_new_df = sim_comp_new_df.append(pd.DataFrame({"sim_name": sim[0][0], "sim_vals": sim_val[0][0]},
                                                                      index=[t]))
            # TODO is this eefective?
            if np.max(sim_comp_new_df["sim_name"].values) >= 0.5:
                best_val = sim_comp_new_df["sim_name"].idxmax()
                if best_val not in newly_added:
                    newly_added[best_val] = set()
                newly_added[best_val].add(w)
            elif np.max(sim_comp_new_df["sim_vals"].values) >= 0.5:
                best_val = sim_comp_new_df["sim_vals"].idxmax()
                if best_val not in newly_added:
                    newly_added[best_val] = set()
                newly_added[best_val].add(w)

        final_entities = {k: v.split(", ") + list(newly_added[k]) if k in newly_added else v.split(", ")
                          for k, v in best_reprs_body[WORDS].items()}
        return final_entities
