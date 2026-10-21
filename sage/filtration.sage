"""
Support-filtration computation for DH_2(G_{5,4}; F_2).

F_r is spanned by non-degenerate cubes whose based support has size at most r.
The computation uses local exact-support blocks and the connecting maps
    delta_2 : H_2(F_2,F_1) -> H_1(F_1),
    delta_3 : H_3(F_3,F_2) -> H_2(F_2).

Chains are integer bit vectors; boundary maps are stored by columns.
Over F_2, repeated faces cancel by XOR and boundary signs disappear.
The auxiliary rank-two quotient is not itself a homology group.

Load after graph.sage and cubical.sage. Linear algebra is supplied by gf2.py.
"""

from dataclasses import dataclass
from operator import xor
from itertools import combinations

from gf2 import (
    GF2Basis,
    GF2KernelTracker,
    GF2Ranker,
    linear_image,
    set_bits,
)


# ----------------------------------------------------------------------
# Helpers for Sage cross-checks
# ----------------------------------------------------------------------

def bitset_matrix(columns, nrows):
    """Convert integer-encoded columns to a Sage matrix over F_2."""
    columns = tuple(columns)

    return Matrix(
        GF(2),
        nrows,
        len(columns),
        lambda i, j: (columns[j] >> i) & 1,
    )


def sage_rank(columns, nrows):
    """Compute the column rank using Sage's matrix routines."""
    return bitset_matrix(columns, nrows).rank()


# ----------------------------------------------------------------------
# Exact-support chain blocks
# ----------------------------------------------------------------------

class ExactSupportBlock:
    """
    Local relative chain block for a nonempty exact support J.

    A basis element (g, cube) represents g * cube, with g in W_J and cube
    based of exact support J. All translates are retained. Faces of exact
    support vanish in the relative boundary.
    """

    def __init__(self, cubes, J):
        self.cubes = cubes
        self.graph = cubes.graph
        self.J = frozenset(J)

        assert self.J
        assert self.J <= frozenset(range(self.graph.rank))

        self.W = tuple(sorted(self.graph.parabolics[self.J]))

        self.based_c1 = tuple(
            (self.graph.e, v)
            for v in sorted(self.graph.relative_neighbors)
            if self.graph.element_support[v] == self.J
        )

        self.based_c2 = cubes.square_groups[self.J]

        self.c1 = tuple(
            (g, cube)
            for g in self.W
            for cube in self.based_c1
        )

        self.c2 = tuple(
            (g, cube)
            for g in self.W
            for cube in self.based_c2
        )

        self.c1_index = {
            basis_element: i for i,
            basis_element in enumerate(self.c1)
        }

        self.c2_index = {
            basis_element: i for i,
            basis_element in enumerate(self.c2)
        }

        assert len(self.c1_index) == len(self.c1)
        assert len(self.c2_index) == len(self.c2)

        self.d2 = tuple(
            self.boundary(g, cube, 2)
            for g, cube in self.c2
        )

    def boundary(self, g, based_cube, q):
        """
        Compute the relative boundary of the translated cube g * based_cube.

        For q=2, express the boundary in the c1 basis; for q=3, in the c2 basis.
        Discard faces that vanish under normalization or have support
        strictly contained in J. Return the boundary as an F_2 bit vector.
        """
        assert q in (2, 3)

        actual = self.cubes.translate(g, based_cube)
        index = (self.c1_index if q == 2 else self.c2_index)

        out = 0

        for d in range(q):
            for epsilon in (0, 1):
                face = face_cube(actual, q, d, epsilon)

                if not self.cubes.normalized(face, q - 1):
                    continue

                based_face = self.cubes.rebase(face)
                support = self.cubes.based_support(based_face)

                assert support <= self.J

                if support == self.J:
                    bit = 1 << index[(face[0], based_face)]
                    out = xor(out, bit)

        return out

    def descriptors(self, cube):
        """
        Extract all non-degenerate nonzero square faces of a 3-cube.

        Return each face as (initial vertex, based face, exact support),
        including faces whose support is strictly smaller than J.
        Keep repeated faces for cancellation over F_2.
        """
        out = []

        for d in range(3):
            for epsilon in (0, 1):
                descriptor = self.cubes.face_descriptor(
                    face_cube(cube, 3, d, epsilon))

                if descriptor is not None:
                    assert descriptor[2] <= self.J
                    out.append(descriptor)

        return tuple(out)

    def d3_from_descriptors(self, g, descriptors):
        """Compute the relative d_3 column using stored face descriptors."""
        out = 0

        for h, based_face, support in descriptors:
            if support == self.J:
                bit = 1 << self.c2_index[(self.graph.mul[g][h], based_face)]
                out = xor(out, bit)

        return out


# ----------------------------------------------------------------------
# Local H_2 in ranks one and two
# ----------------------------------------------------------------------

@dataclass
class LocalH2:
    """
    Store the computed local relative H_2 data.

    block: The local relative chain complex.
    rank_d2: Rank of the boundary map d_2.
    image_d3: Basis of im(d_3), used for quotient reduction.
    representatives: Cycles representing a basis of H_2.
    certificates: Independent d_3 columns and their source 3-cubes.
    """
    block: ExactSupportBlock
    rank_d2: int
    image_d3: GF2Basis
    representatives: tuple
    certificates: tuple

    @property
    def h2_dimension(self):
        """Return dim C_2 - rank(d_2) - rank(d_3)."""
        return (len(self.block.c2) - self.rank_d2 - self.image_d3.rank)


def compute_low_rank_h2(cubes, J, progress_every=100000, sage_cross_check=True):
    """
    Compute local relative H_2 for |J|=1 or 2.

    Find ker(d_2), exhaustively process all non-degenerate exact-support-J
    3-cubes and their W_J-translates, then select homology representatives.
    Check d_2 d_3=0 and recompute retained columns from their source cubes.
    Optional Sage checks compare d_2 rank and retained d_3 column rank.
    """
    block = ExactSupportBlock(cubes, J)

    assert len(block.J) in (1, 2)

    # Kernel and rank of d_2

    kernel_tracker = GF2KernelTracker()
    kernel = []

    for j, column in enumerate(block.d2):
        relation = kernel_tracker.add_column(column, j)

        if relation is not None:
            kernel.append(relation)

    rank_d2 = len(block.c2) - len(kernel)

    # Image of d_3

    image = GF2Basis()
    certificates = []

    based_c3_count = 0
    column_count = 0

    for cube in cubes.based_maps(3, block.W):
        if not cubes.normalized(cube, 3):
            continue

        if cubes.based_support(cube) != block.J:
            continue

        based_c3_count += 1
        descriptors = block.descriptors(cube)

        for g in block.W:
            column = block.d3_from_descriptors(g, descriptors)

            column_count += 1

            # d_2 d_3 = 0.
            assert linear_image(column, block.d2) == 0

            if image.add(column):
                certificates.append((column, g, cube))

        if (progress_every and based_c3_count % progress_every == 0):
            print("support", support_label(J), "based C3",
                  based_c3_count, "rank d3", image.rank, flush=True)

    # Replay every independent d_3 certificate directly from the cube.
    for column, g, cube in certificates:
        assert column == block.boundary(g, cube, 3)

    # Quotient ker(d_2) / im(d_3): retain original cycle representatives

    quotient = GF2Basis()
    representatives = []

    for cycle in kernel:
        assert linear_image(cycle, block.d2) == 0

        reduced_cycle = image.reduce(cycle)

        if quotient.add(reduced_cycle):
            representatives.append(cycle)

    assert (len(representatives) == len(kernel) - image.rank)

    # Rank comparison using Sage matrices

    if sage_cross_check:
        sage_rank_d2 = sage_rank(block.d2, len(block.c1))
        assert sage_rank_d2 == rank_d2

        certificate_columns = tuple(column for column, _, _ in certificates)
        sage_rank_d3 = sage_rank(certificate_columns, len(block.c2))
        assert sage_rank_d3 == image.rank

    result = LocalH2(
        block=block,
        rank_d2=rank_d2,
        image_d3=image,
        representatives=tuple(representatives),
        certificates=tuple(certificates),
    )

    print(
        "J =", support_label(J), "|W_J| =", len(block.W), "C2 =", len(block.c2), "rank d2 =", rank_d2, "rank d3 =", image.rank, "H2 =", result.h2_dimension, "exhaustive columns =", column_count, flush=True)

    return result


# ----------------------------------------------------------------------
# The first filtration stage F_1
# ----------------------------------------------------------------------

@dataclass
class F1Data:
    """
    Store the chain data and boundary maps for F_1.

    c1 and c2 contain the bases of C_1(F_1) and C_2(F_1).
    d1 and d2 contain the corresponding boundary matrix columns.
    image_d2 stores a basis of im(d_2), used to compute H_1(F_1).
    """
    c1: tuple
    c2: tuple
    c1_index: dict
    d1: tuple
    d2: tuple
    image_d2: GF2Basis
    rank_d1: int

    @property
    def h1_dimension(self):
        """Return dim C_1(F_1) - rank(d_1) - rank(d_2)."""
        return (len(self.c1) - self.rank_d1 - self.image_d2.rank)


def compute_f1(cubes, sage_cross_check=True):
    """
    Construct C_2(F_1) -> C_1(F_1) -> C_0(F_1).

    Retain all group translates of non-degenerate cubes of support rank one.
    Check d_1 d_2=0 and optionally compare both ranks with Sage matrices.
    """
    G = cubes.graph

    based_c1 = tuple(
        (G.e, v)
        for v in sorted(G.relative_neighbors)
        if len(G.element_support[v]) == 1
    )

    based_c2 = tuple(
        cube
        for cube in cubes.based_squares
        if len(cubes.based_support(cube)) == 1
    )

    c1 = tuple(
        cubes.translate(g, cube)
        for g in G.vertices
        for cube in based_c1
    )

    c2 = tuple(
        cubes.translate(g, cube)
        for g in G.vertices
        for cube in based_c2
    )

    c1_index = {
        cube: i
        for i, cube in enumerate(c1)
    }

    assert len(c1_index) == len(c1)
    assert len(set(c2)) == len(c2)

    # d_1 sends an oriented edge (u,v) to u+v over F_2.
    d1 = tuple(xor(1 << u, 1 << v) for u, v in c1)

    d2 = []
    image_d2 = GF2Basis()

    for cube in c2:
        column = 0

        for d in range(2):
            for epsilon in (0, 1):
                face = face_cube(cube, 2, d, epsilon)

                if cubes.normalized(face, 1):
                    assert len(cubes.support(face)) == 1

                    bit = 1 << c1_index[face]
                    column = xor(column, bit)

        assert linear_image(column, d1) == 0

        image_d2.add(column)
        d2.append(column)

    ranker_d1 = GF2Ranker()

    for column in d1:
        ranker_d1.add(column)

    if sage_cross_check:
        assert sage_rank(d1, len(G.vertices)) == ranker_d1.rank

        assert sage_rank(d2, len(c1)) == image_d2.rank

    result = F1Data(c1=tuple(c1), c2=tuple(c2), c1_index=c1_index, d1=tuple(
        d1), d2=tuple(d2), image_d2=image_d2, rank_d1=ranker_d1.rank)

    print("F1:", "C1 =", len(result.c1), "C2 =", len(result.c2), "rank d1 =", result.rank_d1,
          "rank d2 =", result.image_d2.rank, "H1 =", result.h1_dimension, flush=True)

    return result


# ----------------------------------------------------------------------
# Direct non-degenerate boundaries and the connecting map delta_2
# ----------------------------------------------------------------------

def full_normalized_boundary(cubes, chain, basis, q, translation=None):
    """
    Compute the full non-degenerate boundary of a q-chain over F_2.

    The chain is encoded as a bit vector, with basis[j] = (g, based_cube).
    Optionally translate the entire chain before computing its boundary.
    Return the set of nonzero faces, including those of lower support.

    Use the reference normalization routine for independent verification.
    """
    out = set()

    if translation is None:
        translation = cubes.graph.e

    for j in set_bits(chain):
        g, cube = basis[j]

        actual = cubes.translate(cubes.graph.mul[translation][g], cube)

        for d in range(q):
            for epsilon in (0, 1):
                face = face_cube(actual, q, d, epsilon)

                if cubes.normalized_reference(face, q - 1):
                    if face in out:
                        out.remove(face)
                    else:
                        out.add(face)

    return out


def connecting_boundary(cubes, local, f1, representative):
    """
    Compute the boundary used in the connecting map delta_2.

    Take a representative 2-cycle of the local relative H_2,
    translate it by the given coset representative, and compute
    its full non-degenerate boundary in C_1(F_1).

    Return the boundary as an F_2 bit vector, before reduction
    modulo im(d_2) of F_1.
    """
    assert len(local.representatives) == 1

    boundary = full_normalized_boundary(
        cubes,
        local.representatives[0],
        local.block.c2,
        2,
        representative,
    )

    out = 0

    for edge in boundary:
        assert len(cubes.support(edge)) == 1

        bit = 1 << f1.c1_index[edge]
        out = xor(out, bit)

    assert linear_image(out, f1.d1) == 0

    return out


@dataclass
class Delta2Data:
    """
    Store the computed connecting map delta_2 and its kernel.

    labels: Basis elements of H_2(F_2,F_1), indexed by (J, coset representative).
    raw_columns: Boundaries representing delta_2 images in C_1(F_1).
    columns: The same boundaries reduced modulo im(d_2) of F_1.
    kernel: A basis of ker(delta_2), in domain coordinates.
    table: Rank contributions from each rank-two support.
    coset_reps: Left-coset representatives for each support J.
    rank: Rank of delta_2.
    """
    labels: tuple
    raw_columns: tuple
    columns: tuple
    kernel: tuple
    table: tuple
    coset_reps: dict
    rank: int

    @property
    def h2_f2(self):
        """
        Return dim H_2(F_2) = dim ker(delta_2), using H_2(F_1) = 0.
        """
        return len(self.columns) - self.rank


def compute_delta2(cubes, rank1, rank2, f1, sage_cross_check=True,):
    """
    Compute the connecting map delta_2 and a basis of its kernel.

    For each rank-two support J, translate a local relative H_2 generator
    over all left cosets of W_J. Compute its full boundary and reduce it
    modulo im(d_2) of F_1 to obtain its image in H_1(F_1).

    Compute the kernel and verify the ranks and kernel relations.
    Optionally cross-check the ranks using Sage matrices.
    """
    assert all(result.h2_dimension == 0 for result in rank1.values())

    G = cubes.graph

    image = GF2Basis()
    tracker = GF2KernelTracker()

    labels = []
    raw_columns = []
    columns = []
    kernel = []
    table = []
    coset_reps = {}

    domain_dimension = 0

    for jt in combinations(range(G.rank), 2):
        J = frozenset(jt)
        local = rank2[J]

        assert (local.h2_dimension == len(local.representatives) == 1)

        reps = G.coset_representatives(J)
        coset_reps[J] = reps

        domain_dimension += (len(reps) * local.h2_dimension)

        before = image.rank

        for rep in reps:
            boundary = connecting_boundary(cubes, local, f1, rep)

            # Work in H_1(F_1) by reducing modulo im d_2.
            reduced = f1.image_d2.reduce(boundary)

            assert linear_image(reduced, f1.d1) == 0

            relation = tracker.add_column(reduced, len(columns))

            if relation is not None:
                kernel.append(relation)

            image.add(reduced)

            labels.append((J, rep))
            raw_columns.append(boundary)
            columns.append(reduced)

        table.append((support_label(J), len(reps),
                     image.rank - before, image.rank))

        print("J =", support_label(J), "cosets =", len(reps), "new rank =",
              image.rank - before, "cumulative =", image.rank, flush=True)

    assert domain_dimension == len(columns)

    # --------------------------------------------------------------
    # Verify rank(delta_2) using raw boundaries
    # --------------------------------------------------------------

    direct_ranker = GF2Ranker()

    for column in f1.image_d2.pivots.values():
        direct_ranker.add(column)

    boundary_rank = direct_ranker.rank

    for column in raw_columns:
        direct_ranker.add(column)

    assert (direct_ranker.rank - boundary_rank == image.rank)

    # --------------------------------------------------------------
    # Kernel relations and independence
    # --------------------------------------------------------------

    for relation in kernel:
        assert relation
        assert linear_image(relation, columns) == 0

    kernel_ranker = GF2Ranker()

    for relation in kernel:
        kernel_ranker.add(relation)

    assert kernel_ranker.rank == len(kernel)

    assert (len(kernel) == domain_dimension - image.rank)

    # --------------------------------------------------------------
    # Rank comparison using Sage matrices
    # --------------------------------------------------------------

    if sage_cross_check:
        # rank(im d_2)
        sage_boundary_rank = sage_rank(f1.d2, len(f1.c1))
        assert (sage_boundary_rank == f1.image_d2.rank)

        # rank(im d_2 + raw connecting boundaries)
        combined_columns = (tuple(f1.d2) + tuple(raw_columns))
        sage_combined_rank = sage_rank(combined_columns, len(f1.c1))
        assert (sage_combined_rank - sage_boundary_rank == image.rank)

        # The reduced columns themselves must have the same rank.
        assert sage_rank(columns, len(f1.c1)) == image.rank

    result = Delta2Data(labels=tuple(labels), raw_columns=tuple(raw_columns), columns=tuple(
        columns), kernel=tuple(kernel), table=tuple(table), coset_reps=coset_reps, rank=image.rank)

    print("delta2:", "domain =", len(result.columns), "rank =", result.rank,
          "kernel =", len(result.kernel), "dim H2(F2) =", result.h2_f2, flush=True)

    return result


# ----------------------------------------------------------------------
# Relative H_2 vanishing in ranks three and four
# ----------------------------------------------------------------------

@dataclass
class RelativeSurjectivity:
    """
    Store certificates proving surjectivity of the relative d_3 map.

    Each certificate records an independent boundary column and the
    3-cube producing it, specified by (kind, square_index, top, g).
    The based 3-cube is reconstructed from its bottom square and top face.
    """
    block: ExactSupportBlock
    rank: int
    certificates: tuple


def prism_3cubes_over_square(cubes, square, J):
    """
    Enumerate non-degenerate 3-cubes with a fixed based bottom square.

    The square has exact support J. Top values lie in W_J and satisfy all
    edge conditions, so every resulting cube also has exact support J.
    """
    G = cubes.graph
    J = frozenset(J)
    allowed = G.parabolics[J]

    x00, x10, x01, x11 = square

    for y00 in sorted(G.closed[x00] & allowed):
        for y10 in sorted(G.closed[x10] & G.closed[y00] & allowed):
            for y01 in sorted(G.closed[x01] & G.closed[y00] & allowed):
                for y11 in sorted(G.closed[x11] & G.closed[y10] & G.closed[y01] & allowed):
                    cube = (x00, x10, x01, x11, y00, y10, y01, y11)

                    if cubes.normalized(cube, 3):
                        assert (cubes.based_support(cube) == J)
                        yield cube


def prove_relative_surjectivity(cubes, J):
    """
    Prove that the relative boundary d_3 is surjective for |J|=3 or 4.

    Since C_1 = 0, surjectivity implies H_2 = 0.
    First try 3-cubes with constant top faces, then general prisms
    if necessary. Stop once their boundaries span C_2.

    Verify the independent boundary columns by reconstructing
    their source 3-cubes and recomputing their boundaries.
    """
    block = ExactSupportBlock(cubes, J)

    assert len(block.J) in (3, 4)
    assert not block.c1
    assert not any(block.d2)

    image = GF2Ranker()
    certificates = []

    cones = 0
    prisms = 0
    columns = 0

    target = len(block.c2)

    def insert(cube, kind, square_index):
        nonlocal columns

        descriptors = block.descriptors(cube)

        for g in block.W:
            column = block.d3_from_descriptors(g, descriptors)

            columns += 1

            if image.add(column):
                certificates.append((column, kind, square_index, cube[4:], g))

    # --------------------------------------------------------------
    # Constant-top cones
    # --------------------------------------------------------------

    for j, square in enumerate(block.based_c2):
        apices = set(block.W)

        for vertex in square:
            apices.intersection_update(cubes.graph.closed[vertex])

        for apex in sorted(apices):
            cube = square + (apex,) * 4

            if not cubes.normalized(cube, 3):
                continue

            cones += 1
            insert(cube, "cone", j)

            if image.rank == target:
                break

        if image.rank == target:
            break

    # --------------------------------------------------------------
    # General prisms if cones do not span C_2
    # --------------------------------------------------------------

    if image.rank < target:
        for j, square in enumerate(block.based_c2):
            for cube in prism_3cubes_over_square(cubes, square, block.J):
                # Constant-top cubes were already processed above.
                if (cube[4] == cube[5] == cube[6] == cube[7]):
                    continue

                prisms += 1
                insert(cube, "prism", j)

                if image.rank == target:
                    break

            if image.rank == target:
                break

    assert (image.rank == len(certificates) ==
            target), ("surjectivity not established", support_label(J))

    # --------------------------------------------------------------
    # Replay every independent certificate directly
    # --------------------------------------------------------------

    replay_ranker = GF2Ranker()

    for (column, kind, square_index, top, g) in certificates:
        cube = (block.based_c2[square_index] + top)

        assert cubes.is_graph_map(cube, 3)
        assert cubes.normalized_reference(cube, 3)
        assert (cubes.based_support(cube) == block.J)
        assert column == block.boundary(g, cube, 3)

        replay_ranker.add(column)

    assert replay_ranker.rank == target

    result = RelativeSurjectivity(
        block=block, rank=image.rank, certificates=tuple(certificates))

    print("J =", support_label(J), "C2 =", target, "rank d3 =", image.rank, "cones =", cones,
          "prisms =", prisms, "columns =", columns, "replayed =", len(certificates), flush=True)

    return result


# ----------------------------------------------------------------------
# Rank-two residual coordinates and the connecting map delta_3
# ----------------------------------------------------------------------

class RankTwoResiduals:
    """
    Represent the rank-two quotient space Q used in the delta_3 computation.

    Q is the direct sum of C_2^{g,J}/im(d_3^{g,J}) over rank-two blocks.
    Each block is represented by coordinates remaining after reduction
    modulo its relative 3-boundaries.

    Q is larger than H_2(F_2,F_1). A relative cycle represents either
    zero or the unique nonzero local H_2 class in each block.
    Membership in ker(delta_2) is checked separately.
    """

    def __init__(self, cubes, rank2, delta2):
        self.cubes, self.graph, self.delta2 = cubes, cubes.graph, delta2
        self.labels = delta2.labels
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.reducers, self.vertex_to_rep, self.blocks = {}, {}, {}
        total = 0
        for jt in combinations(range(self.graph.rank), 2):
            J = frozenset(jt)
            local = rank2[J]
            image = local.image_d3
            # Non-pivot coordinates represent C_2 / im(d_3).
            free = tuple(j for j in range(len(local.block.c2))
                         if j not in image.pivots)
            position = {j: i for i, j in enumerate(free)}

            def compact(v):
                return sum(1 << position[j] for j in set_bits(v))
            units = tuple(compact(image.reduce(1 << j))
                          for j in range(len(local.block.c2)))
            # The cycle subspace is the line spanned by seed.
            seed = compact(image.reduce(local.representatives[0]))
            assert seed and len(free) == local.rank_d2 + local.h2_dimension
            for col, _, _ in local.certificates:
                assert linear_image(col, units) == 0
            self.reducers[J] = dict(
                block=local.block, width=len(free), units=units, seed=seed)
            vmap = {}
            for rep in delta2.coset_reps[J]:
                label = (J, rep)
                for w in local.block.W:
                    v = self.graph.mul[rep][w]
                    assert v not in vmap
                    vmap[v] = rep
                self.blocks[label] = dict(start=total, width=len(free), seed=seed,
                                          class_index=self.index[label])
                total += len(free)
            assert len(vmap) == len(self.graph.vertices)
            self.vertex_to_rep[J] = vmap
        self.dimension = total

    def encode_face(self, h, bf, J):
        """
        Convert a rank-two square h * bf into coordinates in Q.

        Find the left coset hW_J and write h = rep * lg with lg in W_J.
        Compute the class of lg * bf in C_2^{e,J}/im(d_3),
        then place its coordinates in the corresponding summand of Q.
        """
        rep = self.vertex_to_rep[J][h]
        lg = self.graph.mul[self.graph.inv[rep]][h]
        R = self.reducers[J]
        j = R['block'].c2_index[(lg, bf)]
        return R['units'][j] << self.blocks[(J, rep)]['start']

    def decode_cycle(self, vector):
        """
        Convert a relative H_2 class represented in Q into H_2(F_2,F_1) coordinates.

        In each rank-two block, check that the component is either zero
        or the unique nonzero local H_2 class (seed).
        Return the corresponding bit vector in the basis of H_2(F_2,F_1).

        Membership in ker(delta_2) is checked separately.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label in self.labels:
            B = self.blocks[label]
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            assert piece in (
                0, B['seed']), ('not a local H2 class', label, piece)
            if piece:
                out = xor(out, 1 << B['class_index'])
        return out

    def translate_class(self, vector, r):
        """
        Compute the left action of r on H_2(F_2,F_1).

        Each basis element is indexed by a pair (J, rep), representing
        a left coset rep * W_J. Left multiplication by r sends this
        basis element to the one indexed by the coset (r * rep) * W_J.

        Return the resulting class as a bit vector.
        """
        out = 0
        for i in set_bits(vector):
            J, rep = self.labels[i]
            new_rep = self.vertex_to_rep[J][self.graph.mul[r][rep]]
            out = xor(out, 1 << self.index[(J, new_rep)])
        return out

    def encode_boundary(self, faces):
        """
        Convert the rank-two part of a 2-chain into coordinates in Q.

        faces contains the squares with coefficient one over F_2.
        Discard squares of support rank less than two, and map each
        remaining square to its class in the corresponding summand of Q.
        """
        out = 0
        for f in faces:
            bf = self.cubes.rebase(f)
            J = self.cubes.based_support(bf)
            assert len(J) <= 2
            if len(J) == 2:
                out = xor(out, self.encode_face(f[0], bf, J))
        return out


class DepTracker:
    """
    Track linear dependencies among rank-three relative boundaries.

    For each source 3-cube, store:
    a: its rank-three relative boundary,
    b: its rank-two boundary component represented in Q.

    Apply the same elimination operations to a, b, and the source indices.
    When a reduces to zero, obtain a relative 3-cycle and its
    rank-two boundary component in Q.
    """

    def __init__(self):
        self.pivots = {}
        self.raw_sources = []
        self.unit_lower = None
        self.unit_witness = None

    def add(self, a, b, source):
        """
        Process a source 3-cube and its boundary components (a, b).

        Reduce the rank-three boundary a using stored pivots, applying
        the same operations to b and the source witness.

        If a is independent, store a new pivot and return None.
        Otherwise, return (b, witness), where b is the lower boundary
        of a relative 3-cycle formed from the current source and the
        stored sources specified by witness.
        """
        witness = 0
        while a:
            k = a.bit_length() - 1
            if k not in self.pivots:
                witness = xor(witness, 1 << len(self.raw_sources))
                self.raw_sources.append(source)
                self.pivots[k] = (a, b, witness)
                return None
            old_a, old_b, old_witness = self.pivots[k]
            a = xor(a, old_a)
            b = xor(b, old_b)
            witness = xor(witness, old_witness)
        return b, witness

    @property
    def rank(self):
        """Return the dimension spanned by the stored upper vectors."""
        return len(self.pivots)

    def freeze(self, dimension):
        """
        Precompute source chains whose rank-three boundaries are basis vectors.

        Assume the stored rank-three boundaries span the entire target space.
        For each basis vector e_k, construct a source chain u_k with
        rank-three boundary e_k, and store its lower boundary and witness.

        Use these data to eliminate future rank-three boundaries directly.
        """
        assert set(self.pivots) == set(range(dimension))
        lower, witness = [], []
        for k in range(dimension):
            a, b, w = self.pivots[k]
            a = xor(a, 1 << k)
            lower.append(xor(b, linear_image(a, lower)))
            witness.append(xor(w, linear_image(a, witness)))
        self.unit_lower, self.unit_witness = tuple(lower), tuple(witness)
        # Check lookup evaluation against ordinary pivot elimination on every unit.
        for k in range(dimension):
            a, b = 1 << k, 0
            while a:
                p = a.bit_length() - 1
                a = xor(a, self.pivots[p][0])
                b = xor(b, self.pivots[p][1])
            assert b == self.unit_lower[k]


class LocalResidualView:
    """
    Represent the rank-two summands of Q contained in a fixed W_J.

    Select blocks (I, rep) with I contained in J and rep in W_J.
    Assign consecutive coordinates to these blocks, preserving their
    original quotient coordinates.

    No further quotient or homology computation is performed.
    """

    def __init__(self, residuals, J):
        self.residuals, self.J = residuals, frozenset(J)
        self.W = residuals.graph.parabolics[self.J]
        self.blocks = {}
        self.dimension = 0
        for label in residuals.labels:
            I, rep = label
            if I <= self.J and rep in self.W:
                old = residuals.blocks[label]
                self.blocks[label] = dict(start=self.dimension, width=old['width'],
                                          seed=old['seed'], class_index=old['class_index'])
                self.dimension += old['width']

    def encode_face(self, h, bf, I):
        """
        Convert a rank-two square h * bf into coordinates in the local residual space.

        Find the left coset hW_I and write h = rep * lg with lg in W_I.
        Compute the class of lg * bf in C_2^{e,I}/im(d_3),
        then place its coordinates in the corresponding summand of the local residual space.
        """
        R = self.residuals.reducers[I]
        rep = self.residuals.vertex_to_rep[I][h]
        lg = self.residuals.graph.mul[self.residuals.graph.inv[rep]][h]
        index = R['block'].c2_index[(lg, bf)]
        return R['units'][index] << self.blocks[(I, rep)]['start']

    def embed(self, vector):
        """
        Convert local residual coordinates into coordinates in Q.

        Copy each local block into its corresponding block in Q.
        Set all other coordinates to zero.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label, B in self.blocks.items():
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            out = xor(out, piece << self.residuals.blocks[label]['start'])
        return out

    def decode(self, vector):
        """
        Convert a relative H_2 class represented in the local residual space
        into global H_2(F_2,F_1) coordinates.

        In each rank-two block, check that the component is either zero
        or the unique nonzero local H_2 class (seed).
        Return the corresponding bit vector in the basis of H_2(F_2,F_1).

        Membership in ker(delta_2) is checked separately.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label, B in self.blocks.items():
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            assert piece in (0, B['seed']), ('not a cycle', label, piece)
            if piece:
                out = xor(out, 1 << B['class_index'])
        return out


@dataclass
class Delta3SourceCycle:
    """
    Store a relative 3-cycle and its image under delta_3.

    support: Rank-three support J of the source cycle.
    candidate_number: Position in the enumeration of candidate based 3-cubes.
    terms: Summands (g, based_cube) representing the relative 3-cycle.
    class_vector: Coordinates of its delta_3 image in ker(delta_2).
    """
    support: frozenset
    candidate_number: int
    terms: tuple
    class_vector: int


@dataclass
class Delta3Data:
    """
    Store the computed rank of delta_3 and its certificates.

    rank: Rank of delta_3.
    table: Rank contributions from each rank-three support.
    source_cycles: Reconstructed relative 3-cycles and their images.
    certificates: Independent translated images, recorded as
        (source_index, translation, class_vector).
    """
    rank: int
    table: tuple
    source_cycles: tuple
    certificates: tuple


def verify_delta3_source(
    cubes, residuals, view, tracker,
    source, witness, dependency, candidate_number
):
    """
    Reconstruct and verify a relative 3-cycle used to compute delta_3.

    Combine the current source with the stored sources selected by witness.
    Recompute the full normalized boundary and verify that it lies in F_2.

    Compare its rank-two residual with the dependency obtained by elimination.
    Decode the resulting class in H_2(F_2,F_1) and check that it lies
    in ker(delta_2).
    """
    terms = tuple(tracker.raw_sources[i]
                  for i in set_bits(witness)) + (source,)
    for g, cube in terms:
        assert g in view.W and cubes.is_graph_map(cube, 3)
        assert cubes.normalized_reference(cube, 3)
        assert cubes.based_support(cube) == view.J
    faces = full_normalized_boundary(cubes, (1 << len(terms)) - 1, terms, 3)
    assert all(len(cubes.support(f)) <= 2 for f in faces)
    replay_residual = residuals.encode_boundary(faces)
    assert replay_residual == view.embed(dependency)
    q = residuals.decode_cycle(replay_residual)
    assert q == view.decode(dependency)
    assert linear_image(q, residuals.delta2.columns) == 0
    return Delta3SourceCycle(view.J, candidate_number, terms, q)


def compute_delta3(cubes, rank3_relative, residuals, progress_every=0):
    """
    Compute the connecting map delta_3 and prove its surjectivity.

    For each rank-three support J, find linear dependencies among
    the relative boundaries of 3-cubes to construct relative 3-cycles.
    Compute their connecting images in H_2(F_2) and their left translates.

    Retain independent images and verify their source 3-cycles and boundaries.
    Use precomputed elimination data once the relative d_3 has full rank.
    Stop when the image rank equals dim H_2(F_2).
    """
    G = cubes.graph
    image = GF2Ranker()
    table = []
    source_cycles = []
    certs = []

    target = residuals.delta2.h2_f2
    stop = False

    face_maps = tuple(
        face_indices(3, d, epsilon)
        for d in range(3)
        for epsilon in (0, 1)
    )

    for jt in combinations(range(G.rank), 3):
        if stop:
            break

        J = frozenset(jt)
        block = rank3_relative[J].block

        assert rank3_relative[J].rank == len(block.c2)

        view = LocalResidualView(residuals, J)
        tracker = DepTracker()
        reps = G.coset_representatives(J)

        before = image.rank
        nc = nd = columns = 0
        seen_dependencies = {0}
        effect_cache = {}
        frozen = False

        # Split each boundary into exact-support-J and rank-two components.
        def column_pair(g, descriptors):
            a = b = 0

            for h, bf, I in descriptors:
                gh = G.mul[g][h]

                if I == J:
                    a = xor(a, 1 << block.c2_index[(gh, bf)])
                elif len(I) == 2:
                    b = xor(b, view.encode_face(gh, bf, I))
                else:
                    assert I < J and len(I) <= 1

            return a, b

        # Precompute and cache each face's contribution after eliminating the rank-three boundary.
        def frozen_effect(desc):
            if desc not in effect_cache:
                h, bf, I = desc

                if I == J:
                    effect_cache[desc] = tuple(
                        tracker.unit_lower[
                            block.c2_index[(G.mul[g][h], bf)]
                        ]
                        for g in block.W
                    )

                elif len(I) == 2:
                    effect_cache[desc] = tuple(
                        view.encode_face(G.mul[g][h], bf, I)
                        for g in block.W
                    )

                else:
                    assert I < J and len(I) <= 1
                    effect_cache[desc] = (0,) * len(block.W)

            return effect_cache[desc]

        # Decode, translate, and verify new independent connecting images.

        def process_dependency(dep, g, bc, descriptors, witness=None):
            nonlocal stop

            if dep in seen_dependencies:
                return

            seen_dependencies.add(dep)

            # Decode the residual as a class in H_2(F_2,F_1)
            # and verify that it belongs to ker(delta_2).
            q = view.decode(dep)

            assert q == residuals.decode_cycle(view.embed(dep))
            assert linear_image(q, residuals.delta2.columns) == 0

            if not q:
                return

            # Generate left translates and retain linearly independent
            # classes in H_2(F_2) = ker(delta_2).
            pending = []

            for r in reps:
                tq = residuals.translate_class(q, r)

                assert linear_image(tq, residuals.delta2.columns) == 0

                if image.add(tq):
                    pending.append((r, tq))

                assert image.rank <= target

                if image.rank == target:
                    stop = True
                    break

            if not pending:
                return

            # Recover the source 3-chain if only its residual was computed
            # using the frozen elimination tables.
            if witness is None:
                a, b = column_pair(g, descriptors)

                assert dep == xor(b, linear_image(a, tracker.unit_lower))

                witness = linear_image(a, tracker.unit_witness)

            # Reconstruct the relative 3-cycle and verify its full boundary.
            original = verify_delta3_source(
                cubes, residuals, view, tracker, (g, bc), witness, dep, nc
            )

            source_index = len(source_cycles)
            source_cycles.append(original)

            for r, tq in pending:
                full = full_normalized_boundary(
                    cubes, (1 << len(original.terms)) - 1, original.terms, 3, r)

                assert all(len(cubes.support(f)) <= 2 for f in full)

                assert residuals.decode_cycle(
                    residuals.encode_boundary(full)) == tq

                certs.append((source_index, r, tq))

        # Enumerate non-degenerate based 3-cubes of exact support J.
        for bc in cubes.based_maps(3, block.W):
            if (
                not cubes.normalized(bc, 3)
                or cubes.based_support(bc) != J
            ):
                continue

            nc += 1
            descriptors = []

            # Extract the non-degenerate square faces and their supports.
            for indices in face_maps:
                face = tuple(bc[i] for i in indices)
                desc = cubes.face_descriptor(face)

                if desc is not None:
                    assert desc[2] <= J
                    descriptors.append(desc)

            descriptors = tuple(descriptors)

            # Find relative 3-cycles by Gaussian elimination until d_3 has full rank.
            if not frozen:
                for g in block.W:
                    a, b = column_pair(g, descriptors)
                    result = tracker.add(a, b, (g, bc))
                    columns += 1

                    if result is not None:
                        nd += 1
                        dep, witness = result

                        process_dependency(dep, g, bc, descriptors, witness)

                    if stop:
                        break

                # Precompute inverse images of the C_2 basis under the relative d_3.
                if not stop and tracker.rank == len(block.c2):
                    tracker.freeze(len(block.c2))
                    frozen = True

            # After freezing, compute dependencies directly from cached face contributions.
            else:
                effects = [
                    frozen_effect(desc)
                    for desc in descriptors
                ]

                dependencies = [0] * len(block.W)

                for row in effects:
                    dependencies = [
                        xor(a, b)
                        for a, b in zip(dependencies, row)
                    ]

                for g, dep in zip(block.W, dependencies):
                    columns += 1
                    nd += 1

                    process_dependency(dep, g, bc, descriptors)

                    if stop:
                        break

            if stop:
                break

            if progress_every and nc % progress_every == 0:
                print('J =', support_label(J), 'candidates =', nc, 'upper rank =',
                      tracker.rank, 'image rank =', image.rank, '/', target, flush=True)

        table.append(dict(J=support_label(J), C2=len(block.c2), candidates=nc, columns=columns,
                     upper_rank=tracker.rank, dependencies=nd, new_rank=image.rank - before, cumulative_rank=image.rank))

        print('completed J =', support_label(J), 'columns =', columns, 'upper rank =', tracker.rank,
              'new rank =', image.rank - before, 'cumulative rank =', image.rank, flush=True)

    assert image.rank == target, (
        'Selected connecting images did not span H2(F2).')

    certificate_ranker = GF2Ranker()

    for _, _, q in certs:
        certificate_ranker.add(q)

    assert len(certs) == image.rank
    assert certificate_ranker.rank == image.rank

    return Delta3Data(image.rank, tuple(table), tuple(source_cycles), tuple(certs))
