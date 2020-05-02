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
        def gene_allocation(gene1, gene2):
            random_number = np.random.random()
            if random_number < MUTATION_PROBABILITY:
                return np.random.choice(GENE_TYPES)
            else:
                return np.random.choice(gene1, gene2)
        offspring_genes = "".join(
            [
                gene_allocation(gene1, gene2)
                for gene1, gene2 in zip(
                    self.genetic_sequence,
                    other_parent.genetic_sequence
                )
            ]
        )
        return Organism(self.position, offspring_genes)

