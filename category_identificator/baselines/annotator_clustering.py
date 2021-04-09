import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity as cs
from typing import Dict, List
import math
import copy

from category_identificator.annotator import Annotator
from utils.wordvectors import get_model
from config import MIN_WORDS

MIN_CLUSTERS = 7
MAX_CLUSTERS = 20
MIN_MEAN_SIM = 0.2
CLUSTERS = "clusters"
SIM_SCORE = "sim_score"
AVG_LEN = "avg_length"
VAL = "val"
CL = "cl_"


class Cluster:
    def __init__(self, clusters, alg, **kwargs):
        self.clusters = clusters
        self.alg = alg
        self.outliers = []
        self.mean_size = None
        self.sum_size = None
        self.mean_sim = None
        self.weighted_sim = None
        for k,v in kwargs.items():
            setattr(self, k, v)


class Clustering(Annotator):

    def __init__(self, topic, date, potential_labels=None):
        self.model = get_model()
        self.potential_labels = potential_labels
        super().__init__(topic=topic, date=date)

    def _extract_labels(self) -> (Dict[str, List[str]], List[str]):
        columns = ["d" + str(i) for i in range(self.model.vector_size)]
        vector_df = pd.DataFrame(columns=columns)

        for term in list(self.terms_to_annotate):
            vector_df = vector_df.append(pd.DataFrame([self.model.get_vector(term)], columns=columns, index=[term]))

        results = []

        for val in [0.5, 0.6, 0.7, 0.8]:
            clust_ = AgglomerativeClustering(n_clusters=None, affinity="cosine", linkage="average",
                                             distance_threshold=val).fit(vector_df)
            clusters = {}
            for label, term in zip(clust_.labels_, list(vector_df.index)):
                label_ = CL + str(label)
                if label_ not in clusters:
                    clusters[label_] = []
                clusters[label_].append(term)
            results.append(Cluster(clusters, "aggl_clust", **{"thresh": val}))

        short_results = []
        for cluster_obj in results:
            sim_list = []
            len_list = []
            clusters_short = {}
            for cl, terms in cluster_obj.clusters.items():
                if len(terms) < MIN_WORDS:
                    cluster_obj.outliers.extend(terms) 
                    continue
                clusters_short.update({cl: terms})
                sim = np.mean(cs(vector_df.loc[terms], vector_df.loc[terms]))
                sim_list.append(sim)
                len_list.append(len(terms))
            clu_obj_new = copy.deepcopy(cluster_obj)
            clu_obj_new.clusters = clusters_short
            clu_obj_new.mean_sim = np.mean(sim_list)
            clu_obj_new.mean_size = np.mean(len_list)
            clu_obj_new.sum_size = np.sum(len_list)
            clu_obj_new.weighted_sim = np.sum(sim_list[i]*len_list[i] for i in range(len(len_list)))/np.sum(len_list)
            short_results.append(clu_obj_new)

        short_results = sorted(short_results, reverse=True, key=lambda x: x.weighted_sim * math.log(x.sum_size, 2))
        best_res = short_results[0]
        self.params = {"algorithm": best_res.alg, "threshold": getattr(best_res, "thresh", -1)}

        if self.potential_labels is not None:
            for label in self.potential_labels:
                if label in list(vector_df.index):
                    continue
                vector_df = vector_df.append(pd.DataFrame([self.model.get_vector(label)], columns=columns, index=[label]))

            clusters = {}
            for terms in list(best_res.clusters.values()):
                sim_matr = cs(vector_df.loc[terms], vector_df.loc[self.potential_labels])
                label_id = np.argmax(np.mean(sim_matr, axis=0))
                label = self.potential_labels[label_id]
                clusters[label] = terms
        else:
            clusters = best_res.clusters
        outliers = best_res.outliers.copy()

        if self.terms_to_annotate is not None and self.cluster_all:
            for outlier in outliers.copy():
                cl_sim = []
                for cl_label, cl_terms in clusters.items():
                    cl_sim.append(np.mean(cs(vector_df.loc[cl_terms], [vector_df.loc[outlier]])))
                if np.max(cl_sim) < best_res.thresh - 0.2:
                    continue
                max_id = int(np.argmax(cl_sim))
                clusters[list(clusters)[max_id]].append(outlier)
                outliers.pop(outliers.index(outlier))
        return clusters, outliers

