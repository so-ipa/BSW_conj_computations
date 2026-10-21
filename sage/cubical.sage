"""
Graph cubes, normalization, and parabolic support for G_{5,4}.

A q-cube is a tuple indexed by x_0 + 2*x_1 + ... + 2^{q-1}*x_{q-1}.
Square order is (00,10,01,11); coordinate zero is the lowest bit.
Normalization removes projection and adjacent-coordinate min/max
connection degeneracies. Symmetries and left translates are not identified.
"""

from collections import Counter
from functools import lru_cache
from itertools import zip_longest


# ----------------------------------------------------------------------
# Vertices, edges, and faces of the binary cube
# ----------------------------------------------------------------------

@lru_cache(None)
def cube_edges(q):
    """Return each edge of the binary q-cube once, as vertex-index pairs."""
    return tuple(
        (x, x | (1 << direction))
        for x in range(1 << q)
        for direction in range(q)
        if (x & (1 << direction)) == 0
    )


@lru_cache(None)
def face_indices(q, direction, epsilon):
    """
    Return indices for the face x_direction = epsilon.
    The remaining coordinates retain their original order.
    """
    assert q >= 1
    assert 0 <= direction < q
    assert epsilon in (0, 1)

    indices = []

    for x in range(1 << (q - 1)):
        # Insert epsilon at the selected coordinate position.
        lower_bits = x & ((1 << direction) - 1)
        fixed_bit = epsilon << direction
        upper_bits = (x >> direction) << (direction + 1)
        indices.append(lower_bits | fixed_bit | upper_bits)

    return tuple(indices)


def face_cube(cube, q, direction, epsilon):
    """Restrict the cube to the face x_direction = epsilon."""
    indices = face_indices(q, direction, epsilon)
    return tuple(cube[j] for j in indices)


@lru_cache(None)
def earlier_neighbors(position, q):
    """Return adjacent input vertices with smaller indices, already assigned in backtracking."""
    return tuple(
        position - (1 << direction)
        for direction in range(q)
        if position & (1 << direction)
    )


# ----------------------------------------------------------------------
# Graph cubes and normalization
# ----------------------------------------------------------------------

class CubicalModel:
    """
    Graph-cube operations and an inventory of based non-degenerate squares.

    Graph maps send edges to edges or points. Based cubes have initial value e.
    """

    def __init__(self, graph):
        self.graph = graph
        self.based_squares = ()
        self.square_groups = {}
        self._face_descriptors = {}

    def is_graph_map(self, cube, q):
        """Check vertex membership and the graph-map condition on every edge."""
        if len(cube) != (1 << q):
            return False
        if any(v not in self.graph.vertices for v in cube):
            return False
        return all(
            cube[y] in self.graph.closed[cube[x]]
            for x, y in cube_edges(q)
        )

    def normalized_reference(self, cube, q):
        """
        For a valid graph cube, test projection and connection degeneracy directly.
        Return True if the cube survives, False if it factors through such a map.
        """
        # Degeneracy: the value is independent of one coordinate.
        for direction in range(q):
            mask = 1 << direction
            independent_of_coordinate = all(
                cube[x] == cube[x | mask]
                for x in range(1 << q)
                if (x & mask) == 0
            )
            if independent_of_coordinate:
                return False
        # Positive/negative connection: merge two adjacent coordinates by min/max.
        for direction in range(q - 1):
            for use_max in (False, True):
                fiber_values = {}
                consistent = True
                for x in range(1 << q):
                    coordinates = [(x >> i) & 1 for i in range(q)]
                    adjacent = coordinates[direction:direction + 2]
                    if use_max:
                        merged = max(adjacent)
                    else:
                        merged = min(adjacent)
                    reduced_coordinates = (
                        coordinates[:direction]
                        + [merged]
                        + coordinates[direction + 2:]
                    )
                    reduced_index = sum(
                        value << i
                        for i, value in enumerate(reduced_coordinates)
                    )
                    # A factor map exists only if cube is constant on
                    # every fiber of the connection.
                    if reduced_index in fiber_values:
                        if fiber_values[reduced_index] != cube[x]:
                            consistent = False
                            break
                    fiber_values[reduced_index] = cube[x]
                if consistent:
                    reduced_cube = tuple(
                        fiber_values[i]
                        for i in range(1 << (q - 1))
                    )
                    if self.is_graph_map(reduced_cube, q - 1):
                        return False
        return True

    def normalized(self, cube, q):
        """
        Test whether a valid graph cube survives normalization.
        Use explicit equality conditions for q <= 3 and the direct test otherwise.
        """
        if q == 0:
            return True
        if q == 1:
            return cube[0] != cube[1]
        if q == 2:
            a, b, c, d = cube
            return not (
                # Coordinate projections.
                (a == b and c == d)
                or (a == c and b == d)
                # Min and max on coordinates 0 and 1.
                or (a == b == c)
                or (b == c == d)
            )
        if q == 3:
            a, b, c, d, e, f, g, h = cube
            return not (
                # Coordinate projections.
                (a == b and c == d and e == f and g == h)
                or (a == c and b == d and e == g and f == h)
                or (a == e and b == f and c == g and d == h)
                # Min and max on coordinates 0 and 1.
                or (a == b == c and e == f == g)
                or (b == c == d and f == g == h)
                # Min and max on coordinates 1 and 2.
                or (a == c == e and b == d == f)
                or (c == e == g and d == f == h)
            )
        return self.normalized_reference(cube, q)

    # ------------------------------------------------------------------
    # Left translation, based cubes, and exact support
    # ------------------------------------------------------------------

    def translate(self, g, cube):
        """Return the cube x -> g * cube[x]."""
        multiplication_row = self.graph.mul[g]
        return tuple(multiplication_row[v] for v in cube)

    def rebase(self, cube):
        """Return x -> cube[0]^{-1} * cube[x], with initial value e."""
        inverse_initial = self.graph.inv[cube[0]]
        return self.translate(inverse_initial, cube)

    def based_support(self, cube):
        """
        Return the union of vertex supports for a based cube.
        This is the smallest J such that the image lies in W_J.
        """
        assert cube[0] == self.graph.e
        support_mask = 0
        for vertex in cube[1:]:
            support_mask |= self.graph.element_mask[vertex]
        return frozenset(
            i
            for i in range(self.graph.rank)
            if support_mask & (1 << i)
        )

    def support(self, cube):
        """Return exact support after rebasing the cube at e."""
        return self.based_support(self.rebase(cube))

    def face_descriptor(self, face):
        """
        Cache None for a normalized-zero square, otherwise (g, based, J).
        The original face is g * based; J is its exact based support.
        """
        if face not in self._face_descriptors:
            if not self.normalized(face, 2):
                self._face_descriptors[face] = None
            else:
                based = self.rebase(face)
                self._face_descriptors[face] = (
                    face[0],
                    based,
                    self.based_support(based),
                )
        return self._face_descriptors[face]

    # ------------------------------------------------------------------
    # Enumeration of based graph cubes
    # ------------------------------------------------------------------

    def based_maps_reference(self, q, allowed=None):
        """
        Enumerate based graph cubes by backtracking, including degenerate cubes.
        Restrict vertex values to allowed, or use all graph vertices if omitted.
        """
        if allowed is None:
            allowed = self.graph.vertices
        allowed = frozenset(allowed)
        if self.graph.e not in allowed:
            return
        vertex_count = 1 << q
        assigned = [None] * vertex_count
        assigned[0] = self.graph.e

        def recurse(position):
            if position == vertex_count:
                yield tuple(assigned)
                return
            # Intersect the closed neighborhoods of all already assigned
            # neighbors of the current input vertex.
            candidates = None
            for previous in earlier_neighbors(position, q):
                neighborhood = self.graph.closed[assigned[previous]]
                if candidates is None:
                    candidates = neighborhood
                else:
                    candidates = candidates & neighborhood
            for value in sorted(candidates & allowed):
                assigned[position] = value
                yield from recurse(position + 1)
            assigned[position] = None
        yield from recurse(1)

    def based_maps(self, q, allowed=None):
        """
        Enumerate based graph cubes, including degenerate cubes.
        Use neighborhood intersections for q=3, otherwise reference backtracking.
        """
        if q != 3:
            yield from self.based_maps_reference(q, allowed)
            return
        if allowed is None:
            allowed = self.graph.vertices
        allowed = frozenset(allowed)
        e = self.graph.e
        if e not in allowed:
            return
        closed = {
            v: self.graph.closed[v] & allowed
            for v in allowed
        }
        first = tuple(sorted(closed[e]))

        @lru_cache(None)
        def pair(a, b):
            return tuple(sorted(closed[a] & closed[b]))

        @lru_cache(None)
        def triple(a, b, c):
            return tuple(sorted(closed[a] & closed[b] & closed[c]))
        # Bottom face: (e, x10, x01, x11).
        # Top face:    (y00, y10, y01, y11).
        for x10 in first:
            for x01 in first:
                for x11 in pair(x10, x01):
                    for y00 in first:
                        for y10 in pair(y00, x10):
                            for y01 in pair(y00, x01):
                                for y11 in triple(y01, y10, x11):
                                    yield (
                                        e, x10, x01, x11,
                                        y00, y10, y01, y11,
                                    )

    # ------------------------------------------------------------------
    # Inventory of squares surviving normalization
    # ------------------------------------------------------------------

    def inventory(self):
        """
        Store surviving based squares and group them by exact support.
        Return counts before/after normalization and by support cardinality.
        """
        squares = []
        count = 0
        for cube in self.based_maps(2):
            count += 1
            # Check enumeration validity and normalization consistency.
            assert self.is_graph_map(cube, 2)
            survives = self.normalized(cube, 2)
            assert survives == self.normalized_reference(cube, 2)
            if survives:
                squares.append(cube)
        self.based_squares = tuple(squares)
        assert len(set(squares)) == len(squares)
        self.square_groups = {
            J: tuple(
                cube
                for cube in squares
                if self.based_support(cube) == J
            )
            for J in self.graph.supports
        }
        distribution = Counter(
            len(self.based_support(cube))
            for cube in squares
        )
        return {
            "all_based_2": count,
            "normalized_based_2": len(squares),
            "support_distribution": dict(sorted(distribution.items())),
        }


# ----------------------------------------------------------------------
# Implementation verification and consistency checks
# ----------------------------------------------------------------------
# These auxiliary tests check the consistency of cube operations,
# normalization, enumeration, and support computations.
# They serve as implementation checks, not independent mathematical
# verification of the homology computations.

def equality_patterns(length):
    """
    Enumerate equality patterns with labels assigned by first appearance.
    For example, (0,0,1) means the first two values agree and the third differs.
    """
    if length == 0:
        yield ()
        return

    def extend(prefix, maximum):
        if len(prefix) == length:
            yield tuple(prefix)
            return
        # Reuse an existing label or introduce the next new label.
        for value in range(maximum + 2):
            yield from extend(
                prefix + [value],
                max(maximum, value),
            )

    yield from extend([0], 0)


def verify_cubical_infrastructure(cubes):
    """
    Verify cube operations, normalization, and 3-cube enumeration
    against reference implementations.
    """
    graph = cubes.graph

    # A binary q-cube has q * 2^{q-1} edges for q >= 1.
    for q in range(5):
        expected_edges = q * (1 << (q - 1)) if q else 0
        assert len(cube_edges(q)) == expected_edges

    assert face_cube(tuple(range(4)), 2, 0, 0) == (0, 2)
    assert face_cube(tuple(range(4)), 2, 1, 0) == (0, 1)
    # Check the cubical face identities in dimensions 2 through 4.
    for q in range(2, 5):
        cube = tuple(range(1 << q))
        for i in range(q):
            for j in range(i + 1, q):
                for a in (0, 1):
                    for b in (0, 1):
                        left = face_cube(
                            face_cube(cube, q, j, b), q - 1, i, a
                        )
                        right = face_cube(
                            face_cube(cube, q, i, a), q - 1, j - 1, b
                        )
                        assert left == right

    # Eight distinct labels realize every equality pattern on a 3-cube.
    clique = tuple(sorted(graph.parabolics[frozenset({0, 1})]))

    labels = (
        clique
        + tuple(v for v in graph.vertices if v not in clique)[:2]
    )

    normalization_count = 0

    for q in range(4):
        for pattern in equality_patterns(1 << q):
            cube = tuple(labels[i] for i in pattern)
            assert (
                cubes.normalized(cube, q)
                == cubes.normalized_reference(cube, q)
            )
            normalization_count += 1

    # Verify that optimized and reference 3-cube enumerators agree in order.
    generator_count = 0
    sentinel = object()

    test_supports = (
        frozenset({0}),
        frozenset({0, 2}),
        frozenset({0, 1}),
    )

    for J in test_supports:
        WJ = graph.parabolics[J]
        for generated, reference in zip_longest(
            cubes.based_maps(3, WJ),
            cubes.based_maps_reference(3, WJ),
            fillvalue=sentinel,
        ):
            assert generated is not sentinel
            assert reference is not sentinel
            assert generated == reference
            generator_count += 1

    return {
        "normalization_patterns": normalization_count,
        "generator_cubes": generator_count,
    }


def verify_square_supports(cubes):
    """
    Check image contained in W_I iff J contained in I for each based square.
    Also check that every rebased face has support contained in J.
    """
    graph = cubes.graph

    for cube in cubes.based_squares:
        J = cubes.based_support(cube)
        image = frozenset(cube)
        for I in graph.supports:
            assert (image <= graph.parabolics[I]) == (J <= I)
        for direction in range(2):
            for epsilon in (0, 1):
                face = face_cube(cube, 2, direction, epsilon)
                assert cubes.support(face) <= J

    return len(cubes.based_squares)
