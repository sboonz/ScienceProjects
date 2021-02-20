import numpy as np
import matplotlib.pyplot as plt


def fertility_function(age, time):
    peak_fertility_age = 200
    standard_deviation = 50
    return np.exp(
        -(age - peak_fertility_age) ** 2 / (2 * standard_deviation ** 2)
    )


def death_function(age, time):
    coefficients = []
    return sum(
        [
            coefficient * age ** order
            for coefficient, order in enumerate(coefficients)
        ]
    )


class AgeGroup:
    def __init__(self, _tally, _age=0):
        self.tally = _tally
        self.age = _age

    def deaths(self, death_rate_at_age):
        self.tally -= death_rate_at_age * self.tally

    def gets_older(self):
        self.age += 1

    def get_tally(self):
        return self.tally

    def get_age(self):
        return self.age


class Population:
    def __init__(self, _age_groups):
        self.age_groups = _age_groups

    def changes(
        self,
        time,
        _fertility_function=fertility_function,
        _death_function=death_function
    ):
        offspring_population = 0
        for age_group in self.age_groups:
            offspring_population += _fertility_function(
                age_group.get_age(),
                time
            ) * age_group.get_tally()
            age_group.deaths(_death_function(age_group.age), time)
            age_group.gets_older()
        self.age_groups.append(AgeGroup(offspring_population))

    def plot(self, save_path):
        plt.plot(
            [age_group.get_age() for age_group in self.age_groups],
            [age_group.get_tally() for age_group in self.age_groups]
        )
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def get_total(self):
        sum([age_group.get_tally() for age_group in self.age_groups])


def population_pattern(time_steps=500):
    population = Population([AgeGroup(age) for age in range(20)])
    population_at = []
    for time_step in range(time_steps):
        population.changes(time_step)
        population.plot(f"Population distribution at time {time_step}.png")
        population_at.append(population.get_total())
    plt.plot(population_at)
    plt.show()


