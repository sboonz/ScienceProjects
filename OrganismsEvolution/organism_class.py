"""
Attributes an organism has:

Genetic code, with individual sections responsible for different cell types.

Genes are of type, G, C, T, A

Cell types:
Sensor
Muscle
Fur
Genital
Pigmentation
Enzymes (not cells per se)

Behaviour:
Avoiding predators
Attracted to prey and mate (during mating season)
Genes need to be 95% compatible for reproduction
"""

import numpy as np

MUTATION_PROBABILITY = 0.01
GENE_TYPES = ["A", "C", "G", "T"]


class Organism:
    def __init__(self, position, genetic_sequence):
        self.position = position
        self.genetic_sequence = genetic_sequence

    def reproduce(self, other_parent):
        offspring_gene = ""
        for gene_a, gene_b in zip(
            self.genetic_sequence,
            other_parent.genetic_sequence
        ):
            random_number = np.random.random()
            if random_number < MUTATION_PROBABILITY:
                return np.random.choice(GENE_TYPES)
            else:
                return np.random.choice(a, b)

