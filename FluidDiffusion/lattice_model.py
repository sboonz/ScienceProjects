import numpy as np
from PIL import Image
import time

IMAGE_MAXIMUM_INTENSITY = 255
ROOM_TEMPERATURE = 1


def get_total(field):
    def get_total_charge(cell):
        return cell.population

    vectorized_get_total_charge = np.vectorize(get_total_charge)
    return vectorized_get_total_charge(field)


class ChargeStack:
    def __init__(self, occupancy):
        if occupancy > 0:
            self.charges = list(
                np.random.choice([1, -1], int(occupancy))
            )
        else:
            self.charges = []

    def add_charge(self, charge):
        self.charges.append(charge)

    def total_charge(self):
        return sum(self.charges)

    def population(self):
        return len(self.charges)


class Medium:
    def __init__(self, particle_field):
        """
        The particle distribution is of the form of a 3D staggered array, where
        the first two dimensions represent the x and y respectively, and the
        third is the particle's charge.
        """
        vectorized_charge_stack = np.vectorize(ChargeStack)
        self.particle_field = vectorized_charge_stack(particle_field)

    def update_charge_distribution(self, temperature=ROOM_TEMPERATURE):
        self_field = self.particle_field
        width, height = self_field.shape
        mesh_x, mesh_y = np.meshgrid(
            np.arange(0, width, 1),
            np.arange(0, height, 1)
        )
        vectorized_charge_stack = np.vectorize(ChargeStack)
        template = vectorized_charge_stack(np.zeros((width, height)))

        def move_particle(x, y, charge):
            new_x = x + np.random.choice([-1, 1])
            new_y = y + np.random.choice([-1, 1])
            if new_x not in [-1, width] and new_y not in [-1, height]:
                energy = self_field[new_x, new_y].total_charge() * charge
                boltzmann_factor = np.exp(- energy / temperature)
                # similar implementation to the Metropolis-Hastings Algorithm
                random_number = np.random.random()
                if boltzmann_factor > 1 or random_number > boltzmann_factor:
                    template[new_x, new_y].add_charge(charge)
            template[x, y].add_charge(charge)

        def move_cell(x, y):
            stack_array = np.array(self_field[x, y].charges.copy())
            vectorized_move_particle = np.vectorize(move_particle)
            if len(stack_array):
                vectorized_move_particle(x, y, stack_array)

        vectorized_move_cell = np.vectorize(move_cell)
        vectorized_move_cell(mesh_x, mesh_y)
        self.particle_field = template

    def get_charge_field(self):
        def get_total_charge(cell):
            return cell.total_charge()
        vectorized_get_total_charge = np.vectorize(get_total_charge)
        return vectorized_get_total_charge(self.particle_field)

    def get_population_field(self):
        def get_total_charge(cell):
            return cell.population()
        vectorized_get_total_charge = np.vectorize(get_total_charge)
        return vectorized_get_total_charge(self.particle_field)


if __name__ == "__main__":
    # population = np.random.randint(0, 50, 10000).reshape((100, 100))
    population = np.zeros((100, 100))
    population[10: 20, 10: 20] = 50
    population.astype("int32")
    print(population)
    time_steps = 50
    medium = Medium(population)
    Image.fromarray(medium.get_population_field()).show()
    for _ in range(time_steps):
        medium.update_charge_distribution()
    Image.fromarray(medium.get_population_field()).show()
