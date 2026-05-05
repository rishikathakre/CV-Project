from typing import Dict, Optional, Tuple

import networkx as nx


class ZoneTransitionGraph:
    """Tracks how many times people move from one store zone to another.

    Each edge in the graph is a (from_zone, to_zone) pair with a count of
    how many times that transition happened across all tracked people.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()

    def add_transition(self, from_zone: str, to_zone: str) -> None:
        """Record one movement from from_zone to to_zone."""
        if from_zone == to_zone:
            return
        if self._graph.has_edge(from_zone, to_zone):
            self._graph[from_zone][to_zone]["weight"] += 1
        else:
            self._graph.add_edge(from_zone, to_zone, weight=1)

    def transition_count(self, from_zone: str, to_zone: str) -> int:
        """Return how many times people moved from from_zone to to_zone."""
        if self._graph.has_edge(from_zone, to_zone):
            return self._graph[from_zone][to_zone]["weight"]
        return 0

    def out_edges(self, zone: str) -> Dict[str, int]:
        """Return all zones reachable from zone and the count for each."""
        if zone not in self._graph:
            return {}
        return {nbr: self._graph[zone][nbr]["weight"] for nbr in self._graph.successors(zone)}

    def most_common_transition(self, from_zone: str) -> Optional[Tuple[str, int]]:
        """Return the most frequent destination from from_zone, or None if no transitions."""
        edges = self.out_edges(from_zone)
        if not edges:
            return None
        best = max(edges, key=lambda z: edges[z])
        return best, edges[best]

    def all_edges(self) -> Dict[Tuple[str, str], int]:
        """Return every edge in the graph as a dict of {(from, to): count}."""
        return {(u, v): data["weight"] for u, v, data in self._graph.edges(data=True)}

    def node_count(self) -> int:
        return self._graph.number_of_nodes()

    def edge_count(self) -> int:
        return self._graph.number_of_edges()

    @property
    def graph(self) -> nx.DiGraph:
        return self._graph
