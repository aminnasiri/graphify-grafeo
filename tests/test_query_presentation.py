"""Readable output must retain direction, confidence and source citations."""
import networkx as nx

from graphify.query_presentation import describe_subgraph


def test_incoming_edge_is_not_described_as_outgoing():
    graph = nx.DiGraph()
    graph.add_node('caller', label='caller()', source_file='main.py')
    graph.add_node('callee', label='callee()', source_file='lib.py', source_location='L3')
    graph.add_edge('caller', 'callee', relation='calls', confidence='INFERRED',
                   source_file='main.py', source_location='L9')
    out = describe_subgraph(graph, graph.nodes, graph.edges, ['callee'], 500)
    assert 'caller() calls callee()' in out
    assert 'callee() calls caller()' not in out
    assert '[INFERRED; main.py:L9]' in out
    assert 'lib.py:L3' in out


def test_empty_neighborhood_does_not_invent_behavior():
    graph = nx.DiGraph()
    graph.add_node('isolated', label='isolated()')
    out = describe_subgraph(graph, graph.nodes, [], ['isolated'], 500)
    assert 'No matching relationships' in out
