"""
Linear algebra over F_2 using nonnegative integer bit vectors.

Bit j is the coefficient of e_j; addition is XOR (^).
For example, (1,0,1) is encoded by 5. The ambient dimension is external.
Import as Python: Sage's .sage preparser interprets ^ as exponentiation.
"""


def set_bits(vector):
    """Yield nonzero coefficient indices in increasing order; input is nonnegative."""
    while vector:
        # Isolate the lowest nonzero bit.
        lowest_bit = vector & -vector
        index = lowest_bit.bit_length() - 1
        yield index
        vector ^= lowest_bit


def linear_image(vector, columns):
    """Return sum_j vector_j * columns[j] over F_2, where columns[j] = A(e_j)."""
    image = 0

    for index in set_bits(vector):
        image ^= columns[index]

    return image


class GF2Basis:
    """
    Store a subspace U and reduce vectors to representatives modulo U.

    pivots[p] has highest nonzero coefficient p. Reduction clears all
    stored pivot coordinates; its remainder is zero exactly on U.
    """

    def __init__(self):
        self.pivots = {}

    def reduce(self, vector):
        """Return the representative with pivot coordinates zero, without changing U."""
        remainder = 0
        while vector:
            pivot = vector.bit_length() - 1
            basis_vector = self.pivots.get(pivot)
            if basis_vector is None:
                bit = 1 << pivot
                remainder ^= bit
                vector ^= bit
            else:
                vector ^= basis_vector
        return remainder

    def add(self, vector):
        """Add an independent vector to U; return whether the rank increases."""
        remainder = self.reduce(vector)
        if remainder == 0:
            return False
        pivot = remainder.bit_length() - 1
        self.pivots[pivot] = remainder
        return True

    @property
    def rank(self):
        """Return dim U."""
        return len(self.pivots)


class GF2Ranker:
    """
    Compute rank by incremental elimination with highest-coordinate pivots.

    Unlike GF2Basis, stop at the first unused pivot; no quotient
    representative is needed.
    """

    def __init__(self):
        self.pivots = {}

    def add(self, vector):
        """Process one vector; return whether it increases the rank."""
        while vector:
            pivot = vector.bit_length() - 1
            basis_vector = self.pivots.get(pivot)
            if basis_vector is None:
                self.pivots[pivot] = vector
                return True
            vector ^= basis_vector
        return False

    @property
    def rank(self):
        """Return the dimension spanned by the processed vectors."""
        return len(self.pivots)


class GF2KernelTracker:
    """
    Track column relations for a linear map A.

    Each stored pair satisfies vector = A(witness). Apply the same XOR
    operations to both entries; a zero image yields a kernel witness.
    """

    def __init__(self):
        self.pivots = {}

    def add_column(self, vector, index):
        """
        Process A(e_index), using distinct nonnegative indices for successive columns.
        Return None for an independent column, otherwise a nonzero relation
        in column coordinates. A zero column returns e_index.
        """
        witness = 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            stored_pair = self.pivots.get(pivot)
            if stored_pair is None:
                self.pivots[pivot] = (vector, witness)
                return None
            basis_vector, basis_witness = stored_pair
            vector ^= basis_vector
            witness ^= basis_witness
        return witness
