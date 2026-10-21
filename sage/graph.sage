"""
Coxeter group and graph G_{5,4}.

Vertices are elements of W(A_5), stored as integer IDs.
Distinct u,v are adjacent when u^{-1}v belongs to a standard parabolic
of rank at most two. Internal simple-reflection indices are 0,...,4.
"""


# ----------------------------------------------------------------------
# Parameters and labels
# ----------------------------------------------------------------------

CARTAN_TYPE = ["A", 5]
MAX_PARABOLIC_RANK = 2


def support_label(J):
    """Convert internal support indices to the one-based labels used in output."""
    return tuple(i + 1 for i in sorted(J))


# ----------------------------------------------------------------------
# Coxeter group and graph
# ----------------------------------------------------------------------

class ParabolicGraph:
    def __init__(
        self,
        cartan_type=CARTAN_TYPE,
        max_parabolic_rank=MAX_PARABOLIC_RANK,
    ):
        self.cartan_type = cartan_type
        self.max_parabolic_rank = max_parabolic_rank
        # --------------------------------------------------------------
        # Group elements and their integer IDs
        # --------------------------------------------------------------
        self.W = CoxeterGroup(cartan_type)
        self.simple_reflections = self.W.simple_reflections()
        self.simple_indices = tuple(sorted(self.simple_reflections.keys()))
        self.rank = len(self.simple_indices)
        # elements[i] is the group element represented by ID i.
        # element_index[w] is the ID representing the group element w.
        self.elements = tuple(self.W)
        self.element_index = {
            w: i for i, w in enumerate(self.elements)
        }
        self.vertices = tuple(range(len(self.elements)))
        self.e = self.element_index[self.W.one()]
        # --------------------------------------------------------------
        # Group operations
        # --------------------------------------------------------------
        # inv[i] represents elements[i]^{-1}.
        self.inv = tuple(
            self.element_index[~w]
            for w in self.elements
        )
        # mul[i][j] represents elements[i] * elements[j].
        self.mul = tuple(
            tuple(
                self.element_index[u * v]
                for v in self.elements
            )
            for u in self.elements
        )
        # simple[j] is the ID of the j-th internally indexed simple reflection.
        self.simple = tuple(
            self.element_index[self.simple_reflections[i]]
            for i in self.simple_indices
        )
        # All subsets of the internally indexed simple reflections.
        self.supports = tuple(
            frozenset(J)
            for r in range(self.rank + 1)
            for J in Subsets(range(self.rank), r)
        )
        # --------------------------------------------------------------
        # Coxeter support
        # --------------------------------------------------------------
        # The support of w is the set of simple reflections occurring in
        # a reduced expression. This set is independent of the expression.
        # Sage uses labels 1,...,5; internally we use 0,...,4.
        self.element_support = tuple(
            frozenset(i - 1 for i in w.reduced_word())
            for w in self.elements
        )
        # Bit j is one precisely when j belongs to the support.
        self.element_mask = tuple(
            sum(1 << j for j in J)
            for J in self.element_support
        )
        # --------------------------------------------------------------
        # Standard parabolic subgroups
        # --------------------------------------------------------------
        # W_J = {w in W : supp(w) is contained in J}.
        # Each subgroup is stored as a set of integer IDs.
        self.parabolics = {
            J: frozenset(
                w
                for w in self.vertices
                if self.element_support[w] <= J
            )
            for J in self.supports
        }
        # --------------------------------------------------------------
        # Adjacency in G_{5,4}
        # --------------------------------------------------------------
        # The closed neighborhood of e is the union of all W_J with
        # |J| <= max_parabolic_rank.
        allowed = set()
        for J in self.supports:
            if len(J) <= self.max_parabolic_rank:
                allowed.update(self.parabolics[J])
        self.relative_neighbors = frozenset(allowed - {self.e})
        identity_closed = self.relative_neighbors | {self.e}
        # N[g] = g N[e]. Thus v belongs to N[g] precisely when
        # g^{-1}v belongs to one of the allowed standard parabolics.
        self.closed = tuple(
            frozenset(
                self.mul[g][h]
                for h in identity_closed
            )
            for g in self.vertices
        )
        self.neighbors = tuple(
            neighborhood - {g}
            for g, neighborhood in enumerate(self.closed)
        )
        # Each undirected edge is listed once.
        self.graph = Graph(
            [
                (u, v)
                for u in self.vertices
                for v in self.neighbors[u]
                if u < v
            ]
        )
        self._check_structure()

    # ------------------------------------------------------------------
    # Left cosets
    # ------------------------------------------------------------------

    def coset_representatives(self, J):
        """
        Return left-coset representatives of gW_J in deterministic order.
        Choose the smallest integer ID in each coset.
        """
        J = frozenset(J)
        WJ = self.parabolics[J]
        unused = set(self.vertices)
        representatives = []
        while unused:
            rep = min(unused)
            coset = {
                self.mul[rep][w]
                for w in WJ
            }
            representatives.append(rep)
            unused.difference_update(coset)
        assert len(representatives) * len(WJ) == len(self.vertices)
        return tuple(representatives)

    # ------------------------------------------------------------------
    # Checks of indexing and graph properties
    # ------------------------------------------------------------------

    def _check_structure(self):
        """Check support indexing, adjacency symmetry, vertex count, and connectivity/regularity."""
        assert self.element_support[self.e] == frozenset()
        # Internal simple-reflection index j corresponds to support {j}.
        for j, s in enumerate(self.simple):
            assert self.element_support[s] == frozenset({j})
        # Check symmetry in the neighborhood tables themselves.
        for u in self.vertices:
            for v in self.neighbors[u]:
                assert u in self.neighbors[v]
        # Include all group elements in the Sage graph.
        assert self.graph.num_verts() == self.W.cardinality()
        assert self.graph.is_connected()
        assert self.graph.is_regular()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        """Return graph parameters for the computation log."""
        return {
            "coxeter_type": str(self.cartan_type),
            "vertices": self.graph.num_verts(),
            "degree": self.graph.degree(self.e),
            "edges": self.graph.num_edges(),
            "closed_neighbors": len(self.closed[self.e]),
        }
