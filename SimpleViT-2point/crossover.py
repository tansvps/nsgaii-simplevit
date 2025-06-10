import random
import numpy as np

class Crossover:
    def __init__(self):
        self.crossover_rate = 0.8
    def pair(self,chrom_tuple):
        pair_members = []
        child = []
        latest_pops = chrom_tuple
        for i in range(0, len(latest_pops), 2):
            pair = (latest_pops[i], latest_pops[i + 1])
            pair_members.append(pair)
        for i, p in enumerate(pair_members):
            crossover = Crossover()
            c1,c2 = crossover.perform_crossover(p[0],p[1])
            child.append(c1)
            child.append(c2)
        return child

    def perform_crossover(self, parent1, parent2):
        len1 = len(parent1)
        len2 = len(parent2)
        if np.random.uniform() < self.crossover_rate:
            
            if len1 > 2 and len2 > 2:
                #crossover points1
                point1_1 = random.randint(1, len1 - 1)
                point2_1 = random.randint(1, len1 - 1)
                while point2_1 == point1_1:
                    point2_1 = random.randint(1, len1 - 1)
                point1_1, point2_1 = min(point1_1, point2_1), max(point1_1, point2_1)
    
                #crossover points2
                point1_2 = random.randint(1, len2 - 1)
                point2_2 = random.randint(1, len2 - 1)
                while point2_2 == point1_2:
                    point2_2 = random.randint(1, len2 - 1)
                point1_2, point2_2 = min(point1_2, point2_2), max(point1_2, point2_2)
    
                c1 = parent1[:point1_1] + parent2[point1_2:point2_2] + parent1[point2_1:]
                c2 = parent2[:point1_2] + parent1[point1_1:point2_1] + parent2[point2_2:]
            else:
                 #one-point-crossover
                point1_1 = random.randint(1, len1 - 1)
                point1_2 = random.randint(1, len2 - 1)
                # print("parent1",parent1)
                # print("parent2",parent2)
                c1 = parent1[:point1_1] + parent2[point1_2:]
                c2 = parent2[:point1_2] + parent1[point1_1:]
                # print("child1",c1)
                # print("child2",c2)


            
            if len(c1) > 10:
                c1 = c1[:10]
                print("CUT_C1")
            if len(c2) > 10:
                c2 = c2[:10]
                print("CUT_C2")

        else:
            c1 = parent1
            c2 = parent2
            print("NO CROSSOVER")

        return c1, c2
    # def perform_crossover(self, parent1, parent2):
    #     len1 = len(parent1)
    #     len2 = len(parent2)
    #     if np.random.uniform() < self.crossover_rate:
    #         if len1 == 3 and len2 == 3 :
    #             point1_1 = random.randint(1, len1 - 1)
    #             point1_2 = random.randint(1, len2 - 1)
    #             c1 = parent1[:point1_1] + parent2[point1_2:]
    #             c2 = parent2[:point1_2] + parent1[point1_1:]
  
    #         elif len1 > 2 and len2 > 2:
    #             # Two-point crossover for both parents
    #             point1_1 = random.randint(1, len1 - 1)
    #             point2_1 = random.randint(1, len1 - 1)
    #             while point2_1 == point1_1:
    #                 point2_1 = random.randint(1, len1 - 1)
    #             point1_1, point2_1 = min(point1_1, point2_1), max(point1_1, point2_1)

    #             point1_2 = random.randint(1, len2 - 1)
    #             point2_2 = random.randint(1, len2 - 1)
    #             while point2_2 == point1_2:
    #                 point2_2 = random.randint(1, len2 - 1)
    #             point1_2, point2_2 = min(point1_2, point2_2), max(point1_2, point2_2)

    #             c1 = parent1[:point1_1] + parent2[point1_2:point2_2] + parent1[point2_1:]
    #             c2 = parent2[:point1_2] + parent1[point1_1:point2_1] + parent2[point2_2:]
       
    #         else:
    #             # One-point crossover
    #             point1_1 = random.randint(1, len1 - 1)
    #             point1_2 = random.randint(1, len2 - 1)
    #             c1 = parent1[:point1_1] + parent2[point1_2:]
    #             c2 = parent2[:point1_2] + parent1[point1_1:]
         

    #         if len(c1) > 10:
    #             c1 = c1[:10]
    #             print("CUT_C1")
    #         if len(c2) > 10:
    #             c2 = c2[:10]
    #             print("CUT_C2")
    #     else:
    #         c1 = parent1
    #         c2 = parent2
    #         print("NO CROSSOVER")

    #     return c1, c2
