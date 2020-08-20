from reading.reader_sc_sample import SCSampleReader
from preprocessing.preprocesor import Preprocessor
from graph.graph import Graph
from nea.semantic_graph import NEASemanticGraph


reader = SCSampleReader()
raw_text = reader.text_collection

preprocessor = Preprocessor()
selected_terms = preprocessor.preprocess(raw_text)

graph = Graph(selected_terms)
graph.grow_graph()

semgraph = NEASemanticGraph(graph=graph)
labels = semgraph.extract_labels()
# TODO resolve terms w/o nodes in a graph
a =1

