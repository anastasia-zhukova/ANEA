import numpy as np
import pandas as pd
import copy
import math
import datetime
import json, os
from category_identificator.annotator import Annotator
from sklearn.metrics.pairwise import cosine_similarity as cs
from utils.wordvectors import get_model

from config import EXEC_RES_PATH, MIN_WORDS, logger

MAX_LEVELS = 3
MIN_MEAN_SIM = 0.2
MIN_NAME_SIM = 0.3
MIN_LENGTH = 1
MAX_SIZE_PERC = 0.15
NAME_SIM = "name_sim"
MEAN_SIM = "mean_sim"
SUM_SIM = "sum_sim"
QUALITY_SCORE = "impact_score"
TERM_IMPACT_SCORE = "term_impact_score"
SIZE = "size"
LENGTH = "length"
WORDS = "words"


class ANEAAnnotator(Annotator):
    def __init__(self, graph, topic, date):
        self.graph = graph
        self.model = graph.model
        self.graph.model = None
        self.label_dict = None
        self.dist_df = None
        self.repr_dict = {}
        self.vector_df = None
        self.potential_labels = None
        super().__init__(topic=topic, date=date)

    def _extract_labels(self):
        # potential labels of entity classes
        if self.terms_to_annotate is not None:
            logger.info("Annotating terms")
        else:
            logger.info("Deriving labels and categories.")

        self._collect_labels()
        if self.terms_to_annotate is not None:
            label_dict_large = self.label_dict
        else:
            label_dict_large = {k: v for k, v in self.label_dict.items() if len(v) >= MIN_WORDS}

        self._build_vector_df()
        init_table_long_df, init_table_no_dupl_df = self._build_init_table(label_dict_large)

        len_leaves = len([n for n, node in self.graph.all_nodes.items() if node.term_id is not None])

        # initial filtering of the entity classes
        init_table_filtered_df = init_table_no_dupl_df[init_table_no_dupl_df[MEAN_SIM] >= MIN_MEAN_SIM]
        init_table_filtered_df = init_table_filtered_df[init_table_filtered_df[NAME_SIM] >= MIN_NAME_SIM]
        init_table_filtered_df = init_table_filtered_df[init_table_filtered_df[SIZE] <= MAX_SIZE_PERC * len_leaves]

        # optimization of entity classes
        full_best_reprs_df = self._larger_reprs(init_table_filtered_df)
        substan_overlap_reprs_df = self._overlap_repr(full_best_reprs_df)
        conflicts_resolved_df = self._repr_cleaning(substan_overlap_reprs_df)

        # outliers_all = set(self.graph.terms) - set().union(*[v for v in list(final_entities.values())])
        # final_entities, outliers = self._add_border_terms(conflicts_resolved_df)

        if self.terms_to_annotate is not None and self.cluster_all:
            final_entities, outliers = self._add_border_terms(conflicts_resolved_df)
            outliers = set(self.terms_to_annotate) - set().union(*[v for v in list(final_entities.values())])
        else:
            final_entities = {index: row[WORDS].split(", ") for index, row in conflicts_resolved_df.iterrows()}
            core_terms = [n for n, node in self.graph.all_nodes.items() if node.term_id is not None]
            outliers = set(core_terms) - set().union(*[v for v in list(final_entities.values())])

        # self._save_not_used_labels(init_table_filtered_df, final_entities)
        return final_entities, list(outliers)

    def _save_not_used_labels(self, table, entities):
        table.sort_values(QUALITY_SCORE, ascending=False, inplace=True)
        leftovers = [v for v in list(table.loc[set(table.index) - set(entities)].index) if v.istitle()]
        now = datetime.datetime.now()
        with open(os.path.join(EXEC_RES_PATH, now.strftime("%Y-%m-%d_%H-%M") + "_" + "unusedlabels" + ".json"), "w") as file:
            json.dump(leftovers, file)

    def _build_vector_df(self):
        if self.model is None:
            self.model = get_model()
        self.vector_df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])
        remaining = set(self.graph.terms) - set(self.graph.all_nodes)
        for label in list(self.graph.all_nodes) + list(remaining):
            vector = self.model.get_vector(label)
            self.vector_df = self.vector_df.append(pd.DataFrame([vector],
                             columns=["d" + str(i) for i in range(self.model.vector_size)],
                             index=[label]))
        self.vector_df.drop_duplicates(inplace=True)

    def _collect_labels(self):

        def __get_nodes(leaf):

            if leaf in self.graph.graph_down_top:
                # a term is also a node and can be a representative to itself
                # if self.graph.all_nodes[leaf].term_id is not None and self.graph.all_nodes[leaf].hyponyms is not None:
                #     levels = [self.graph.graph_down_top[leaf] + [leaf]]
                # else:
                levels = [self.graph.graph_down_top[leaf]]
            else:
                return set()

            checked = {leaf}
            i = 0
            while i < MAX_LEVELS and len(levels[i]):
                new_level = set()
                for node in levels[i]:
                    # if node == leaf:
                    #     self.dist_df.loc[leaf, node] = 0.001

                    if node in checked:
                        continue

                    if node != leaf:
                        self.dist_df.loc[leaf, node] = i + 1
                    # else:
                    #     self.dist_df.loc[leaf, leaf] = 0.001

                    if node in self.graph.graph_down_top:
                        if node != leaf:
                            new_level = new_level.union(set(self.graph.graph_down_top[node]))
                    checked.add(node)
                for new_node in copy.copy(new_level):
                    if new_node in checked:
                        new_level.remove(new_node)
                        continue
                    try:
                        if new_node != leaf:
                            self.dist_df.loc[leaf, new_node] = i + 2
                        # else:
                        #     self.dist_df.loc[leaf, leaf] = 0.001
                    except KeyError:
                        if new_node not in list(self.dist_df.columns):
                            new_level.remove(new_node)
                        continue
                levels.append(list(new_level))
                i += 1
            return levels

        leaves = []
        nodes = []
        self.potential_labels = []
        for node in list(self.graph.all_nodes.values()):
            if "#" in node.word or ":" in node.word:
                continue

            if node.term_id is not None:
                if self.terms_to_annotate is not None:
                    if node.word not in self.terms_to_annotate:
                        nodes.append(node.word)
                        continue
                leaves.append(node.word)
            else:
                nodes.append(node.word)

            if node.hyponyms is not None:
                self.potential_labels.append(node.word)

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

    def _build_init_table(self, label_dict_short):
        label_dict_dist = {}

        for k in list(label_dict_short):
            if k not in list(self.dist_df.columns):
                continue
            arr = []
            for w in label_dict_short[k]:
                if self.dist_df.loc[w, k] > 0:
                    arr.append(self.dist_df.loc[w, k])
            label_dict_dist[k] = arr

        sum_df = pd.DataFrame(columns=[SIZE, LENGTH, MEAN_SIM, NAME_SIM, SUM_SIM, QUALITY_SCORE, WORDS])
        for node, words in label_dict_short.items():
            vector_df = self.vector_df.loc[words]
            if len(words):
                try:
                    mean_ = np.mean(cs(vector_df))
                    name_ = cs([np.mean(vector_df.values, axis=0)], [self.vector_df.loc[node].values])[0][0]
                    length_ = np.mean(label_dict_dist[node]) if len(label_dict_dist[node]) else 1
                    sum_df = sum_df.append(pd.DataFrame({
                        SIZE: len(words),
                        LENGTH: length_,
                        MEAN_SIM: mean_,
                        NAME_SIM: name_,
                        SUM_SIM: mean_ + name_,
                        QUALITY_SCORE: max(math.log(len(words), 2), 1) * mean_ * name_ * (mean_ + name_) * length_,
                        WORDS: ", ".join(words)
                    }, index=[node]))
                except ValueError:
                    continue
            else:
                sum_df = sum_df.append(pd.DataFrame({
                    SIZE: 0,
                    LENGTH: 1,
                    MEAN_SIM: 0,
                    NAME_SIM: 0,
                    SUM_SIM:0,
                    QUALITY_SCORE: 0,
                    WORDS: ""
                }, index=[node]))

        sum_df.sort_values([SIZE, WORDS, NAME_SIM], ascending=[False, False, False], inplace=True)
        groups = sum_df.groupby(WORDS, as_index="False")
        repr_dict = {v[0]: list(v[1:]) for k, v in groups.groups.items() if len(v) > 1}
        if len(self.repr_dict) == 0:
            self.repr_dict = repr_dict
        else:
            self.repr_dict.update(repr_dict)
        sum_df.sort_values([QUALITY_SCORE], ascending=[False], inplace=True)
        sum_short_df = sum_df[sum_df[WORDS] != ""].drop_duplicates(subset=[WORDS]).append(sum_df[sum_df[WORDS] == ""])
        return sum_df.sort_values(SIZE, ascending=False), sum_short_df.sort_values(SIZE, ascending=False)

    def _larger_reprs(self, prev_df):
        # find better representatives with better quality score
        prev_df.sort_values([QUALITY_SCORE], ascending=[False], inplace=True)
        output_df = prev_df[prev_df[WORDS] != ""].drop_duplicates(subset=[WORDS]).append(prev_df[prev_df[WORDS] == ""])
        return output_df.sort_values(QUALITY_SCORE, ascending=False)

    def _overlap_repr(self, prev_df):
        # find best representatives with significant overlap (>= 50%)
        prev_df.sort_values(QUALITY_SCORE, ascending=False, inplace=True)

        overlap_sim_df = pd.DataFrame(np.zeros((len(prev_df), len(prev_df))), index=list(prev_df.index),
                                      columns=list(prev_df.index))
        for index_1, rows_1 in prev_df.iterrows():
            overlaps = {}
            words_1 = set(rows_1[WORDS].split(", "))

            for index_2, row_2 in prev_df.iterrows():
                words_2 = set(row_2[WORDS].split(", "))
                overlap = words_1.intersection(words_2)
                if len(overlap):
                    overlaps[index_2] = len(overlap)

            overlaps = {k: v for k, v in sorted(overlaps.items(), reverse=True, key=lambda x: x[1])}
            overlap_df = prev_df.loc[[k for k, v in overlaps.items() if v >= len(words_1) / 2]]
            if not len(overlap_df):
                logger.info(index_1)
                continue
            overlap_sim_df.loc[index_1, str(pd.to_numeric(overlap_df[QUALITY_SCORE]).idxmax())] = 1

        to_remove = []
        for index in list(overlap_sim_df.index):
            destination = overlap_sim_df.loc[index].idxmax()
            if overlap_sim_df.loc[destination, destination] == 1 and index != destination:
                to_remove.append(index)
                if destination not in self.repr_dict:
                    self.repr_dict[destination] = []
                self.repr_dict[destination].append(index)

        output_df = prev_df.loc[list(set(overlap_sim_df.index) - set(to_remove))]
        return output_df

    def _repr_cleaning(self, prev_df):
        # resolve minor conflicting words that fall info multiple classes
        words_overlap_dict = {}
        repr_dict_clean = {index: set(row[WORDS].split(", ")) for index, row in prev_df.iterrows()}
        sim_words_df = pd.DataFrame(columns=list(prev_df.index), index=list(prev_df.index))

        for n, (index_1, row_1) in enumerate(prev_df.iterrows()):
            words_1 = set(row_1[WORDS].split(", "))
            for m, (index_2, row_2) in enumerate(prev_df.iterrows()):
                if m == n:
                    continue
                words_2 = set(row_2[WORDS].split(", "))
                intersect = set(words_1).intersection(set(words_2))
                for w in intersect:
                    if w not in words_overlap_dict:
                        words_overlap_dict[w] = set()
                    words_overlap_dict[w] = words_overlap_dict[w].union({index_1, index_2})

                if len(intersect):
                    sim_words_df.loc[index_1, index_2] = ", ".join(intersect)
                    sim_words_df.loc[index_2, index_1] = ", ".join(intersect)
                else:
                    sim_words_df.loc[index_1, index_2] = ""
                    sim_words_df.loc[index_2, index_1] = ""

                repr_dict_clean[index_1] = repr_dict_clean[index_1] - intersect
                repr_dict_clean[index_2] = repr_dict_clean[index_2] - intersect

                # if labels exist as terms in other classes, remove them and move to the one with the same label
                for ind in [index_1, index_2]:
                    for k, v in repr_dict_clean.items():
                        if ind in v:
                            repr_dict_clean[k] = repr_dict_clean[k] - {ind}
                            repr_dict_clean[ind] = repr_dict_clean[ind].union({ind})

        repr_dict_clean = {k:v for k,v in sorted(repr_dict_clean.items(), reverse=True, key=lambda x: len(x[1]))}
        clean_df, clean_df_short = self._build_init_table(repr_dict_clean)
        result_dict = copy.copy(repr_dict_clean)
        clean_df_short.sort_values(QUALITY_SCORE, ascending=False, inplace=True)
        sim_words_df.fillna("", inplace=True)
        all_resolved = set().union(*list(repr_dict_clean.values()))

        for n, (index, row) in enumerate(clean_df_short.iterrows()):
            conflicting_words = set().union(*[v.split(", ") for v in sim_words_df.loc[index].values])
            for w in conflicting_words:
                if w in all_resolved:
                    continue
                if w not in words_overlap_dict:
                    continue
                candidates = words_overlap_dict[w]
                cand_df = pd.DataFrame(columns=[MEAN_SIM, NAME_SIM, SUM_SIM, LENGTH])

                for cand in candidates:
                    words = clean_df_short.loc[cand, WORDS].split(", ") if clean_df_short.loc[cand, WORDS] != "" else []
                    mean_ = np.mean(cs(self.vector_df.loc[words],
                                       [self.vector_df.loc[w].values] if len(self.vector_df.loc[w].shape) == 1
                                       else self.vector_df.loc[w])) \
                                    if len(words) else 0
                    name_ = cs([self.vector_df.loc[w].values], [self.vector_df.loc[cand].values])[0][0]
                    cand_df = cand_df.append(pd.DataFrame({
                        LENGTH: self.dist_df.loc[w, cand] if self.dist_df.loc[w, cand] > 0 else 1,
                        MEAN_SIM: mean_,
                        NAME_SIM: name_,
                        SUM_SIM: mean_ + name_,
                        TERM_IMPACT_SCORE: mean_ + name_,
                        WORDS: ", ".join(words)
                    }, index=[cand]), sort=False)
                if cand_df[TERM_IMPACT_SCORE].max() > 0:
                    best_cand = cand_df[TERM_IMPACT_SCORE].idxmax()
                    result_dict[best_cand].add(w)
                del words_overlap_dict[w]

        repr_dict_short = {k: v for k, v in result_dict.items() if len(v) >= MIN_WORDS}
        output_full_df, output_short_df = self._build_init_table(repr_dict_short)
        return output_short_df

    def _add_border_terms(self, prev_df):
        # assign the remaining terms to the bins

        def __score_calc(words_):
            mean_ = np.mean(cs(self.vector_df.loc[words_])) if len(words_) else 0
            name_ = np.mean(cs(self.vector_df.loc[words_], [self.vector_df.loc[repr].values]))
            dist_arr = []
            for w in words_:
                if w in list(self.dist_df.index):
                    if self.dist_df.loc[w, repr] > 0:
                        dist_arr.append(self.dist_df.loc[w, repr])
            # dist_arr = [self.dist_df.loc[w, repr] for w in words_ if self.dist_df.loc[w, repr] > 0]
            length_ = np.mean(dist_arr) if len(dist_arr) > 0 else 1
            impact_score = max(math.log(len(words_), 2), 1) * mean_ * name_ * (mean_ + name_) * length_
            return impact_score

        prev_df.sort_values(QUALITY_SCORE, ascending=False, inplace=True)

        all_words = set().union(*[v.split(", ") for k, v in prev_df[WORDS].items()])
        # all_leaves = set(self.graph.terms)
        if self.terms_to_annotate is not None:
            all_leaves = set(self.terms_to_annotate)
        else:
            all_leaves = set([k for k, v in self.graph.all_nodes.items() if v.term_id is not None])
        not_sorted = all_leaves - all_words
        outliers = []
        result_dict = {index: row[WORDS].split(", ") for index, row in prev_df.iterrows()}

        # impact score with a border point to it's MIN_SIZE best terms should be >= of the impact score of the MIN_SIZE
        # most sim termsw/o this new term
        for new_term in not_sorted:
            if new_term in list(result_dict):
                result_dict[new_term] += [new_term]
                continue

            best_reprs = {}
            for repr, row in prev_df.iterrows():
                words = row[WORDS].split(", ")
                # reverse sort of the word ids by the word similarity to a new term
                if new_term not in list(self.vector_df.index):
                    self.vector_df = self.vector_df.append(pd.DataFrame([self.model.get_vector(new_term)],
                                                                        columns=["d" + str(i) for i in
                                                                                 range(self.model.vector_size)],
                                                                        index=[new_term]))
                word_ids_sorted = cs(self.vector_df.loc[words], [self.vector_df.loc[new_term].values] \
                        if len(self.vector_df.loc[new_term].shape) == 1
                         else self.vector_df.loc[new_term]).reshape(1,-1)[0].argsort()[-len(words):][::-1]
                best_words = list(np.array(words)[list(word_ids_sorted)[:MIN_WORDS]])
                term_impact_score_best = __score_calc(best_words)
                updated_words_ = best_words + [new_term]
                term_impact_score_upd = __score_calc(updated_words_)
                if term_impact_score_upd - term_impact_score_best >= 0.05:
                    best_reprs[repr] = term_impact_score_upd - term_impact_score_best

            if not len(best_reprs):
                outliers.append(new_term)
            else:
                best_repr = list({k:v for k,v in sorted(best_reprs.items(), reverse=True, key=lambda x: x[1])})[0]
                result_dict[best_repr] += [new_term]

        return result_dict, outliers
