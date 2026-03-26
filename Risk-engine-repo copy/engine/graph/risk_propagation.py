# graph/risk_propagation.py

"""
Risk Propagation Module
-----------------------

Propagates fraud risk scores through a transaction graph.

Purpose:
    Detect hidden fraud networks by spreading risk from known
    suspicious nodes to their neighbors.

Method:
    Iterative message passing with decay factor.

Use cases:
    - Fraud ring detection
    - Money mule discovery
    - Suspicious account ranking
"""

import logging
from typing import Dict
import networkx as nx
import pandas as pd


class RiskPropagator:
    """
    Propagate risk scores across graph nodes.
    """

    def __init__(
        self,
        decay: float = 0.85,
        max_iter: int = 20,
        tol: float = 1e-6
    ):
        """
        Parameters
        ----------
        decay : float
            Risk decay factor across edges
        max_iter : int
            Maximum propagation iterations
        tol : float
            Convergence tolerance
        """

        self.decay = decay
        self.max_iter = max_iter
        self.tol = tol

        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)


    def initialize_risk(
        self,
        graph: nx.DiGraph,
        seed_nodes: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Initialize node risk scores.

        Parameters
        ----------
        graph : nx.DiGraph
        seed_nodes : dict
            Known risky nodes with initial risk

        Returns
        -------
        risk_scores : dict
        """

        risk_scores = {node: 0.0 for node in graph.nodes()}

        for node, risk in seed_nodes.items():
            if node in graph:
                risk_scores[node] = risk

        return risk_scores


    def propagate(
        self,
        graph: nx.DiGraph,
        seed_nodes: Dict[str, float]
    ) -> pd.DataFrame:
        """
        Run risk propagation algorithm.
        """

        self.logger.info("Starting risk propagation")

        risk_scores = self.initialize_risk(graph, seed_nodes)

        for iteration in range(self.max_iter):

            new_scores = risk_scores.copy()

            for node in graph.nodes():

                incoming = graph.predecessors(node)

                propagated_risk = 0.0

                for neighbor in incoming:

                    weight = graph[neighbor][node].get("amount", 1.0)

                    propagated_risk += (
                        risk_scores[neighbor] * weight
                    )

                new_scores[node] = (
                    (1 - self.decay) * risk_scores[node]
                    + self.decay * propagated_risk
                )

            diff = sum(
                abs(new_scores[n] - risk_scores[n])
                for n in graph.nodes()
            )

            risk_scores = new_scores

            self.logger.info(
                f"Iteration {iteration} diff={diff:.6f}"
            )

            if diff < self.tol:
                self.logger.info("Risk propagation converged")
                break

        df_risk = pd.DataFrame({
            "node": list(risk_scores.keys()),
            "risk_score": list(risk_scores.values())
        })

        return df_risk


    def normalize_scores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize risk scores between 0 and 1.
        """

        min_val = df["risk_score"].min()
        max_val = df["risk_score"].max()

        df["risk_score_normalized"] = (
            (df["risk_score"] - min_val)
            / (max_val - min_val + 1e-9)
        )

        return df


    def rank_accounts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank accounts by risk.
        """

        df = df.sort_values(
            by="risk_score_normalized",
            ascending=False
        )

        df["risk_rank"] = range(1, len(df) + 1)

        return df