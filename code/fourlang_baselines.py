import json
import requests

from networkx.readwrite import json_graph
from nltk.corpus import stopwords

from fourlang_utils import Utils
from misc import words_from_id

DEBUG = True
DEF_ENDPOINT = "http://hlt.bme.hu/4lang/definition"
utils = Utils()
STOPS = set(stopwords.words('english'))


def is_stop(n):
    return n in STOPS or len(n) == 0 or n[0] == '=' or n.isupper()


def get_def_graph(word):
    data = {'word': word}
    data_json = json.dumps(data)
    # payload = {'json_payload': data_json}
    headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}
    r = requests.post(DEF_ENDPOINT, data=data_json, headers=headers)
    w_def = r.json()['word']
    graph = json_graph.adjacency.adjacency_graph(w_def)
    return graph


def concepts_from_lemma(lemma):
    def_graph = get_def_graph(lemma)
    return nodes(def_graph)


def nodes(graph, expand=0):
    ns = [node.split('_')[0] for node in graph.nodes]
    ns = set([n for n in ns if not is_stop(n)])
    if expand == 0:
        return ns
    return ns.union(*(nodes(get_def_graph(n), expand-1) for n in ns))

def node_support(prem, hyp, expand=0):
    nodes_prem = nodes(prem, expand)
    nodes_hyp = nodes(hyp, expand)
    if DEBUG:
        print(f'pr nodes: {nodes_prem}, hy nodes: {nodes_hyp}')
    sim = utils.asim_jac(nodes_prem, nodes_hyp)
    return sim


def edge_support(prem, hyp):
    edges_prem = utils.get_edges(prem)
    edges_hyp = utils.get_edges(hyp)
    if DEBUG:
        print(f'pr edges: {edges_prem}, hy edges: {edges_hyp}')
    sim = utils.asim_jac(edges_prem, edges_hyp)
    return sim


def fourlang(
        premise, hypothesis, type_prem, id_prem, type_hypo, id_hypo,
        is_premise_reversed, is_hypothesis_reversed):
    # stop_words = stopwords.words('english')
    pr_lemmata = words_from_id(id_prem)
    hy_lemmata = words_from_id(id_hypo)

    pr_graphs = {lemma: get_def_graph(lemma) for lemma in pr_lemmata}
    hy_graphs = {lemma: get_def_graph(lemma) for lemma in hy_lemmata}
    pr_pred = pr_lemmata[-1] if is_premise_reversed else pr_lemmata[0]
    hy_pred = hy_lemmata[-1] if is_hypothesis_reversed else hy_lemmata[0]

    if DEBUG:
        print(f'pr pred: {pr_pred}, hy pred: {hy_pred}')

    pr_pred_graph = pr_graphs[pr_pred]
    hy_pred_graph = hy_graphs[hy_pred]

    return node_support(pr_pred_graph, hy_pred_graph, expand=1)

    """
    # 3. Criterion: is voice and inversement the same?
    voice_pr = voice_of_id(id_prem)
    voice_hy = voice_of_id(id_hypo)
    same_voice = voice_pr == voice_hy
    same_inversement = is_premise_reversed == is_hypothesis_reversed
    third_criterion = same_voice == same_inversement
    """
