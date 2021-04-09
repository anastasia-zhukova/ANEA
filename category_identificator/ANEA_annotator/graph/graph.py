import progressbar
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cs
from typing import List
from config import *

from category_identificator.ANEA_annotator.graph.node import Node
from category_identificator.ANEA_annotator.graph.node_properties import WikiProperties, WikiLink, Areas
from utils.wordvectors import get_model
from config import TERM_GROUPS


class Graph:

    def __init__(self, terms, fraction=TERM_GROUPS, topN_terms=None, domain=None):
        self.graph_top_down = {}
        self.graph_down_top = {}
        self.words_processed = set()
        self.all_nodes = {}
        self.all_wiki_properties = {}
        self.areas_all = {}
        self.areas_selected = set()
        self.not_in_graph = {}
        self.terms = terms
        self.fraction = fraction
        self.domain = domain
        self.topN_terms = topN_terms
        self.terms_grouped = None

        self.model = None

        self._create_graph(terms)
        self._domain_types_identif()

        logger.info("Graph constructed.\n")

    def _create_graph(self, selected_terms):

        self.terms_grouped = {}
        terms_are_heads = {}
        for key, term in selected_terms.items():
            # add to a dict terms that were identified as heads and these terms were found in the text
            if term.wikt_page_head == term.wikt_page and term.wikt_page_head is not None:
                terms_are_heads[term.wikt_page] = term
            # group terms by heads
            if term.wikt_page_head not in self.terms_grouped:
                self.terms_grouped[term.wikt_page_head] = []
            self.terms_grouped[term.wikt_page_head].append(term)

        self.terms_grouped = {k:v for k,v in sorted(self.terms_grouped.items(), reverse=True,
                                               key=lambda x: (len(x[1]), sum([v.counter for v in x[1]])))}
        if self.topN_terms is None:
            topN_heads = int(len(self.terms_grouped) / self.fraction) if self.fraction is not None \
                                                                                else len(self.terms_grouped)
            self.topN_terms = 0
            for j, (head_link, terms) in enumerate(self.terms_grouped.items()):
                if j > topN_heads:
                    break
                self.topN_terms += len(terms)

        logger.info("All terms: {0}, selected for the graph: {1}".format(str(selected_terms),
                                                                             str(self.topN_terms)))

        widgets = [progressbar.FormatLabel(
            "PROGRESS: Processing %(value)d-th/%(max_value)d (%(percentage)d %%) groups of terms (in: %(elapsed)s).")]
        bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.terms_grouped)).start()

        added_terms = 0
        for j, (head_link, terms) in enumerate(self.terms_grouped.items()):
            # skip terms w/o wiki-head as candidates for core terms
            if head_link is None:
                continue

            if added_terms > self.topN_terms:
                break

            for term in terms:
                if term.wikt_page == head_link and head_link is not None:
                    # will create a node later
                    continue
                if head_link is None and term.wikt_page is None:
                    self.not_in_graph[term.word] = term
                    continue
                # add a leaf node for which head != leaf node
                self._add_node(word=term.word, leaf=True, url=term.wikt_page,
                                                 hypernyms=[head_link] if head_link is not None else None,
                                                 term_id=term.id)
            if head_link is None:
                continue

            if head_link in terms_are_heads:
                # head == term, i.e., word found in the text
                head_term = terms_are_heads[head_link]
                head_word = head_term.head
                # leaf == true a term itself is a recognized head and there are no other nodes with similar head
                self._add_node(word=head_word,leaf=False if len(set([t.wikt_page for t in terms])) > 1 else True,
                                                 url=head_link, hypernyms=None, term_id=head_term.id)
            else:
                # head is not directly found in the text, but is derived from the found terms
                head_word = head_link.split("/")[-1]
                self._add_node(word=head_word, leaf=False, url=head_link, hypernyms=None, term_id=None)

            if not self.all_nodes[head_word].is_leaf:
                for term in terms:
                    if term.wikt_page == head_link:
                        continue
                    self._add_edge(head_word, term.word)

            added_terms += len(terms)
            bar.update(j)

        bar.finish()
        self._add_missing_edges()

    def _add_node(self, word, leaf, url, hypernyms, term_id):
        wikiprop = WikiProperties.extract_wiki_properties(url, self.all_wiki_properties, hypernyms)
        self.all_nodes[word] = Node(word=word, leaf=leaf, url=url,
                                         content_dict=wikiprop,
                                         term_id=term_id)
        if word in self.all_wiki_properties:
            self.all_wiki_properties[word].node = self.all_nodes[word]

    def _add_edge(self, upper, lower):
        # add to a dict
        if upper not in self.graph_top_down:
            self.graph_top_down[upper] = []
        self.graph_top_down[upper].append(lower)

        # add to a dict
        if lower not in self.graph_down_top:
            self.graph_down_top[lower] = []
        self.graph_down_top[lower].append(upper)

        # link each other
        found = False
        if self.all_nodes[upper].hyponyms is None:
            self.all_nodes[upper].hyponyms = []
        for hyp in self.all_nodes[upper].hyponyms:
            if hyp.text == lower:
                hyp.node = self.all_nodes[lower]
                found = True
                break

        if not found:
            if not len(self.all_nodes[upper].hyponyms):
                self.all_nodes[upper].hyponyms = []
            self.all_nodes[upper].hyponyms.append(WikiProperties(text=lower,
                                                                 links={lower: WikiLink(lower, self.all_nodes[lower].url)},
                                                                 node=self.all_nodes[lower]))
        # link each other
        found = False
        if self.all_nodes[lower].hypernyms is None:
            self.all_nodes[lower].hypernyms = []
        for hyp in self.all_nodes[lower].hypernyms:
            if hyp.text == upper:
                hyp.node = self.all_nodes[upper]
                found = True
                break

        if not found:
            if not len(self.all_nodes[lower].hypernyms):
                self.all_nodes[lower].hypernyms = []
            self.all_nodes[lower].hypernyms.append(WikiProperties(text=upper,
                                                                  links={upper: WikiLink(upper, self.all_nodes[upper].url)},
                                                                  node=self.all_nodes[upper]))

    def _add_missing_edges(self):
        for node_name, node in self.all_nodes.items():
            if node.hypernyms is None:
                continue
            for hypernym in node.hypernyms:
                if hypernym.text not in self.all_nodes:
                    continue
                if hypernym.text not in self.graph_top_down:
                    self.graph_top_down[hypernym.text] = []
                if node_name not in self.graph_top_down[hypernym.text]:
                    self._add_edge(hypernym.text, node_name)
            if node_name in self.all_wiki_properties:
                if self.all_wiki_properties[node_name].node is None:
                    self.all_wiki_properties[node_name].node = node
        logger.info("Edges restored.\n")

    def grow_graph(self, iter_num=ITER_GROW_GRAPH):
        for i in range(iter_num):
            # create nodes from hypernyms

            widgets = [progressbar.FormatLabel(
                "PROGRESS: ROUND {0}: Processing %(value)d-th (%(percentage)d %%) doc/entry (in: %(elapsed)s).".format(str(i)))]
            bar = progressbar.ProgressBar(widgets=widgets, maxval=len(self.all_nodes)).start()

            for j, node in enumerate(list(self.all_nodes.values())):

                if node.word in self.words_processed:
                    continue

                if node.hypernyms is None:
                    node.hypernyms = []
                    self.words_processed.add(node.word)
                    continue

                # create new nodes
                for hypernym in node.hypernyms:
                    # take only aread with capital letters
                    areas = {v for v in hypernym.areas if v[0].isupper()}
                    if len(areas):
                        if len(areas.intersection(self.areas_selected)) == 0:
                            continue

                    if hypernym.text in self.all_nodes:
                        continue

                    if not len(hypernym.links):
                        continue

                    if ":" in hypernym.text or "#" in hypernym.text:
                        continue

                    self._add_node(word=hypernym.text, leaf=False, url=hypernym.links[hypernym.text].link,
                                   hypernyms=None, term_id=None)
                    self._add_edge(hypernym.text, node.word)
                self.words_processed.add(node.word)
                bar.update(j)
            bar.finish()
            self._add_missing_edges()
        logger.info("Graph expanded.\n")


    def _domain_types_identif(self):
        exceptions = []
        for node in list(self.all_nodes.values()):
            if node.definitions is None:
                continue
            for definition in node.definitions:
                for t in definition.areas:
                    if t == "none":
                        continue
                    if t[0].islower() or len(t.split(" ")) > 1:
                        continue
                    if t == "IPA":
                        exceptions.append(node.word)
                    self.areas_all[t] = self.areas_all.get(t, 0) + 1
        self.areas_all = {k: v for k, v in sorted(self.areas_all.items(), reverse=True, key=lambda x: x[1])}

        if self.model is None:
            self.model = get_model()
        df = pd.DataFrame(columns=["d" + str(i) for i in range(self.model.vector_size)])

        for t in list(self.areas_all.keys()):
            df = df.append(
                    pd.DataFrame([self.model.get_vector(t)],
                                 columns=["d" + str(i) for i in range(self.model.vector_size)],
                                 index=[t]))

        # optimization of the areas selection: maximize the cross-similarity score across the areas and their quantity
        best_areas = []
        for cl_num in range(CLUSTER_AREAS_MIN, CLUSTER_AREAS_MAX, 2):
            agl = AgglomerativeClustering(n_clusters=cl_num).fit(df)
            clusters = {}
            for l, w in zip(agl.labels_, list(df.index)):
                if l not in clusters:
                    clusters[l] = Areas()
                clusters[l].areas.append(w)
                clusters[l].count += self.areas_all[w]

            for c, param in clusters.items():
                clusters[c].score = np.mean(cs(df.loc[param.areas], df.loc[param.areas]))

            areas = set()
            for i in set([np.argmax([v.score for v in list(clusters.values())]),
                          np.argmax([v.count for v in list(clusters.values())])]):
                areas = areas.union(set(clusters[list(clusters)[int(i)]].areas))
            areas = areas.union(set(list(self.areas_all)[:TOP_FREQ_AREAS]))
            best_areas.append(Areas(areas=areas,
                                    score=np.mean(cs(df.loc[areas], df.loc[areas])) * np.sum([self.areas_all[v] for v in areas]),
                                    count=np.sum([self.areas_all[v] for v in areas])))
        # chose the best areas with the max score
        self.areas_selected = best_areas[int(np.argmax([v.score for v in best_areas]))].areas
