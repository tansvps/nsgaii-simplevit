import random
import numpy as np
from copy import deepcopy

class Mutation():
    def do_mutation(self, genotype):
        self.mutation_rate = 0.1
        genotype = deepcopy(genotype)
        if np.random.uniform() < self.mutation_rate:
            random_point = random.randint(0, len(genotype) - 1)
            selected_point = list(genotype[random_point])
            selected_point[0] = random.choice([1, 2, 4, 8, 16])
            selected_point[1] = random.choice([128, 256, 512, 1024, 2048, 4096])
            selected_tuple = tuple(selected_point)
            genotype[random_point] = selected_tuple #****************#
        else:
            print("No Mutation")

        return genotype

    def do_mutation_(self, genotype):
        genotype = deepcopy(genotype)
        random_point = random.randint(0, len(genotype) - 1)
        selected_point = list(genotype[random_point])
        selected_point[0] = random.choice([1, 2, 4, 8, 16])
        selected_point[1] = random.choice([128, 256, 512, 1024, 2048, 4096])
        selected_tuple = tuple(selected_point)
        genotype[random_point] = selected_tuple #****************#


        return genotype

