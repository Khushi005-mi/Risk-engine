import pandas as pd

from engine.graph.graph_builder import GraphBuilder
from engine.graph.risk_propagation import RiskPropagation


def test_graph_creation():

    df = pd.DataFrame({
        "customer_id": [1, 2],
        "merchant_id": ["A", "A"]
    })

    builder = GraphBuilder()

    G = builder.build(df)

    assert len(G.nodes) > 0
    assert len(G.edges) > 0


def test_risk_propagation():

    rp = RiskPropagation()

    result = rp.propagate(
        graph=None,
        risk_scores={"A": 0.9}
    )

    assert isinstance(result, dict)
    