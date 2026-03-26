# This module extracts graph-based risk features from the transaction graph created in graph_builder.py. These features are extremely valuable for detecting:
# fraud rings
# money mule networks
# layered money laundering
# hub accounts
# suspicious intermediaries
# The module computes node-level structural signals that ML models can learn from.
"""
Graph Feature Extraction Module
-------------------------------

Extracts node-level graph features used in fraud and credit-risk models.

Features include:
- degree centrality
- in/out degree
- betweenness centrality
- closeness centrality
- clustering coefficient
- pagerank
- triangle counts

These features help detect:
- hubs
- intermediaries
- fraud rings
- suspicious transaction flows
"""
import logging ## Provides a standardized system to record runtime events (info, warnings, errors) so engineers can monitor and debug the application in production.
import networkx as nx##Imports the NetworkX library used to create, manipulate, and analyze graphs such as transaction networks or account relationships.
import pandas as pd ##Imports the Pandas library used for structured data handling, enabling tabular operations like filtering, aggregation, and feature generation.
from typing import Optional ## Imports the Optional type hint used in function signatures to indicate that a variable can either contain a value of a specified type or be None.
class GraphFeatureExtractor:
    def __init__(self):

        self.logger = logging.getLogger(__name__) # Creates a logger object specific to this module so that all logs generated in this file can be identified by the module’s name.
        logging.basicConfig(level=logging.INFO) # Configures the global logging system and sets the minimum log level to display.
        # Large ML systems may have 100+ modules running simultaneously. Logging helps engineers:
# trace failures
# monitor pipelines
# audit models
# debug distributed systems
# Without logging, debugging production pipelines becomes almost impossible.
    def compute_degree_features(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Compute basic degree features.
        """

        self.logger.info("Computing degree features")

        nodes = list(graph.nodes())

        degree = dict(graph.degree())
        in_degree = dict(graph.in_degree())
        out_degree = dict(graph.out_degree())

        df = pd.DataFrame({
            "node": nodes,
            "degree": [degree[n] for n in nodes],
            "in_degree": [in_degree[n] for n in nodes],
            "out_degree": [out_degree[n] for n in nodes],
        })

        return df
    def compute_clustering_features(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Compute clustering coefficients and triangle counts.
        """

        self.logger.info("Computing clustering features")

        undirected = graph.to_undirected()

        clustering = nx.clustering(undirected)
        triangles = nx.triangles(undirected)

        nodes = list(graph.nodes())

        df = pd.DataFrame({
            "node": nodes,
            "clustering_coeff": [clustering[n] for n in nodes],
            "triangle_count": [triangles[n] for n in nodes],
        })

        return df
    def compute_neighbor_features(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Compute neighbor risk statistics.
        """

        self.logger.info("Computing neighbor features")

        nodes = list(graph.nodes())

        avg_neighbor_degree = nx.average_neighbor_degree(graph)

        df = pd.DataFrame({
            "node": nodes,
            "avg_neighbor_degree": [avg_neighbor_degree[n] for n in nodes],
        })

        return df
    def extract_all_features(self, graph: nx.DiGraph) -> pd.DataFrame:
        """
        Combine all graph features into one dataframe.
        """

        self.logger.info("Extracting all graph features")

        degree_df = self.compute_degree_features(graph)
        centrality_df = self.compute_centrality_features(graph)
        clustering_df = self.compute_clustering_features(graph)
        neighbor_df = self.compute_neighbor_features(graph)

        df = degree_df.merge(centrality_df, on="node")
        df = df.merge(clustering_df, on="node")
        df = df.merge(neighbor_df, on="node")

        return df
      