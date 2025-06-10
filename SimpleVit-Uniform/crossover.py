import random
import numpy as np


class Crossover:
    def __init__(self):
        self.crossover_rate = 0.8

    def pair(self, chrom_tuple):
        pair_members = []
        child = []
        latest_pops = chrom_tuple
        for i in range(0, len(latest_pops), 2):
            pair = (latest_pops[i], latest_pops[i + 1])
            pair_members.append(pair)
        for i, p in enumerate(pair_members):
            crossover = Crossover()
     
            c1, c2 = crossover.crossover_uniform(p[0], p[1])
            child.append(c1)
            child.append(c2)
        return child

    def crossover_uniform(self, parent1, parent2):
        if len(parent2) > len(parent1):
            p1 = parent2
            p2 = parent1
        else:
            p1 = parent1
            p2 = parent2


        start_pos = random.randint(0, len(p1) - len(p2))
        # print("start_pos", start_pos)

        parent_uni1 = p1[start_pos:start_pos + len(p2)]

        c1 = p1[:]
        c2 = p2[:]

        if np.random.uniform() < self.crossover_rate:
            c1 = p1[:start_pos]
            c2 = []

            for i in range(len(parent_uni1)):
                # print(parent_uni1[i], "|||", p2[i])
     
                if np.random.rand() < 0.5:
                    c1.append(parent_uni1[i])
                    c2.append(p2[i])
                else:
                    c1.append(p2[i])
                    c2.append(parent_uni1[i])

            c1.extend(p1[start_pos + len(p2):])
     

        else:
            c1 = parent1
            c2 = parent2
            # print("P1:", parent1)
            # print("P2:", parent2)
            # print("\nC1:",c1)
            # print("C2:", c2)
            print("NO CROSSOVER")

        return c1, c2
