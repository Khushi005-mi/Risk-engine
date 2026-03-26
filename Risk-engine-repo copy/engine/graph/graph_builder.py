# graph/graph_builder.py

"""
Graph Builder Module
--------------------

Constructs a transaction graph from a batch of transactions.

Nodes:
    - Accounts

Edges:
    - Transactions between accounts

Supports:
    - Directed graphs
    - Edge aggregation
    - Feature extraction
"""

import pandas as pd
import networkx as nx
import logging
from typing import Optional


class GraphBuilder:
    """
    Builds a directed transaction graph from transaction data.
    """

    def __init__(
        self,
        source_col: str = "sender_account",
        target_col: str = "receiver_account",
        amount_col: str = "amount",
        timestamp_col: str = "timestamp"
    ):
        self.source_col = source_col
        self.target_col = target_col
        self.amount_col = amount_col
        self.timestamp_col = timestamp_col

        self.graph = nx.DiGraph()

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def build_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build a directed graph from transaction dataframe.
        """

        self.logger.info("Building transaction graph")

        for _, row in df.iterrows():

            src = row[self.source_col]
            dst = row[self.target_col]

            edge_data = {
                "amount": row[self.amount_col],
                "timestamp": row[self.timestamp_col]
            }

            self.graph.add_node(src)
            self.graph.add_node(dst)

            self.graph.add_edge(src, dst, **edge_data)

        self.logger.info(
            f"Graph built with {self.graph.number_of_nodes()} nodes "
            f"and {self.graph.number_of_edges()} edges"
        )

        return self.graph


    def build_aggregated_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Build graph with aggregated edge features
        (multiple transactions between accounts).
        """

        self.logger.info("Building aggregated graph")

        grouped = (
            df.groupby([self.source_col, self.target_col])
            .agg(
                total_amount=(self.amount_col, "sum"),
                transaction_count=(self.amount_col, "count"),
                avg_amount=(self.amount_col, "mean"),
            )
            .reset_index()
        )

        graph = nx.DiGraph()

        for _, row in grouped.iterrows():

            graph.add_edge(
                row[self.source_col],
                row[self.target_col],
                total_amount=row["total_amount"],
                transaction_count=row["transaction_count"],
                avg_amount=row["avg_amount"],
            )

        return graph


    def add_node_features(
        self,
        graph: nx.DiGraph,
        features: pd.DataFrame,
        node_col: str = "account_id"
    ) -> nx.DiGraph:
        """
        Attach node features to graph nodes.
        """

        self.logger.info("Adding node features")

        feature_dict = features.set_index(node_col).to_dict("index")

        for node, feat in feature_dict.items():

            if node in graph:
                graph.nodes[node].update(feat)

        return graph


    def compute_graph_metrics(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Compute graph metrics useful for fraud detection.
        """

        self.logger.info("Computing graph metrics")

        degree = dict(graph.degree())
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())

        pagerank = nx.pagerank(graph)

        clustering = nx.clustering(graph.to_undirected())

        df_metrics = pd.DataFrame({
            "node": list(graph.nodes()),
            "degree": [degree[n] for n in graph.nodes()],
            "in_degree": [in_degree[n] for n in graph.nodes()],
            "out_degree": [out_degree[n] for n in graph.nodes()],
            "pagerank": [pagerank[n] for n in graph.nodes()],
            "clustering": [clustering[n] for n in graph.nodes()]
        })

        return df_metrics


    def export_graph(self, graph: nx.DiGraph, path: str) -> None:
        """
        Export graph to file for downstream processing.
        """

        self.logger.info(f"Saving graph to {path}")

        nx.write_gpickle(graph, path)