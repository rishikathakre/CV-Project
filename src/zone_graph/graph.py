"""
Zone-transition graph: maintains a NetworkX DiGraph where nodes are zone names
and edges are directed transitions between zones, weighted by transition count.
One graph is maintained globally across all persons; per-person queries are
supported by tracking the last-seen zone per person externally (in features.py).
"""

from typing import Dict, Optional, Tuple

import networkx as nx


class ZoneTransitionGraph:
    """Directed graph of zone-to-zone transitions observed in the video."""

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add_transition(self, from_zone: str, to_zone: str) -> None:
        """Record one transition from from_zone → to_zone.

        Adds nodes automatically; increments edge weight on each call.
        Self-loops (staying in the same zone) are ignored.
        """
        if from_zone == to_zone:
            return
        if self._graph.has_edge(from_zone, to_zone):
            self._graph[from_zone][to_zone]["weight"] += 1
        else:
            self._graph.add_edge(from_zone, to_zone, weight=1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def transition_count(self, from_zone: str, to_zone: str) -> int:
        """Return how many times from_zone → to_zone has been observed."""
        if self._graph.has_edge(from_zone, to_zone):
            return self._graph[from_zone][to_zone]["weight"]
        return 0

    def out_edges(self, zone: str) -> Dict[str, int]:
        """Return {neighbour_zone: count} for all edges leaving zone."""
        if zone not in self._graph:
            return {}
        return {
            nbr: self._graph[zone][nbr]["weight"]
            for nbr in self._graph.successors(zone)
        }

    def most_common_transition(self, from_zone: str) -> Optional[Tuple[str, int]]:
        """Return (to_zone, count) for the highest-weight edge from from_zone, or None."""
        edges = self.out_edges(from_zone)
        if not edges:
            return None
        best = max(edges, key=lambda z: edges[z])
        return best, edges[best]

    def all_edges(self) -> Dict[Tuple[str, str], int]:
        """Return {(from_zone, to_zone): count} for every edge in the graph."""
        return {
            (u, v): data["weight"]
            for u, v, data in self._graph.edges(data=True)
        }

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @property
    def graph(self) -> nx.DiGraph:
        """Direct access to the underlying NetworkX DiGraph (read-only intent)."""
        return self._graph
