import numpy as np
import functools

class Selection_NSGAII:
    def __init__(self, genotype, acc_test, parameters):
        self.genotype = genotype
        self.acc_test = acc_test
        self.parameters = parameters
        self.acc_column = []
        self.param_column = []
        self.fitnesses = []
        self.chromosome_nodes = {}
        self.Transfomer = {}
        self.genotype_str = []
        self.all_fitnesses = []
        self.keys = []
        self.fronts = []
        self.Amoebanet_sorted = []
        self.fitnesses_keep_acc = []
        self.fitnesses_keep_param = []
    def normolize(self):
        acc_param = [list(row) for row in zip(self.acc_test, self.parameters)]
        # print("acc_param",acc_param)
        acc_param = np.array(acc_param)
        fitnesses = np.zeros_like(acc_param)
        min_val = np.min(acc_param[:, 0])
        max_val = np.max(acc_param[:, 0])
        val_range = max_val - min_val
        self.acc_column.append((acc_param[:, 0] - min_val) / val_range)
        # print("self.acc_column",self.acc_column)
        # fitnesses[:, 0] = (acc_param[:, 0] - min_val) / val_range
        fitnesses[:, 0] = 1-((acc_param[:, 0] - min_val) / val_range)
        ###################################################
        min_val2 = np.min(acc_param[:, 1])
        max_val2 = np.max(acc_param[:, 1])
        val_range2 = max_val2 - min_val2
        self.param_column.append((acc_param[:, 1] - min_val2) / val_range2)
        # print("self.param_column",self.param_column)
        # fitnesses[:, 1] = 1 - ((acc_param[:, 1] - min_val2) / val_range2)
        fitnesses[:, 1] = (acc_param[:, 1] - min_val2) / val_range2
        self.fitnesses = fitnesses
        # print("self.fitnesses",self.fitnesses)

        o_acc = []
        for i in acc_param:
            o_acc.append(i[0])
        o_param = []
        for i in acc_param:
            o_param.append(i[1])

        n_acc = []
        for i in fitnesses:
            n_acc.append(i[0])
        n_param = []
        for i in fitnesses:
            n_param.append(i[1])

        for i, o_fitness in zip(o_acc, n_acc):
            self.fitnesses_keep_acc.append((i, o_fitness))

        for o, n in zip(o_param, n_param):
            self.fitnesses_keep_param.append((o, n))

        return acc_param,self.fitnesses_keep_acc, self.fitnesses_keep_param


    def param(self):
        for i, chromosome in enumerate(self.genotype):
            node = 'Transformer' + str(i)
            self.chromosome_nodes[node] = chromosome
        # print("chromosome_nodes",self.chromosome_nodes)

        for i, node in enumerate(self.chromosome_nodes):
            # print("node_name",node)
            self.Transfomer[node] = self.fitnesses[i]
            # print("Transfomer",self.Transfomer)

        self.chromosome_nodes.values()
        for i in self.chromosome_nodes.values():
            self.genotype_str.append(i)
        for i in self.Transfomer.values():
            # print(i)
            self.all_fitnesses.append(i)

        self.Transfomer.keys()
        for k in self.Transfomer.keys():
                self.keys.append(k)
        # print("keys",self.keys)

        return self.chromosome_nodes , self.Transfomer ,self.genotype_str,self.all_fitnesses

    def dominates(self, fitnesses_1, fitnesses_2):
        larger_or_equal = fitnesses_1 <= fitnesses_2
        larger = fitnesses_1 < fitnesses_2
        if np.all(larger_or_equal) and np.any(larger):
            return True
        else:
            return False

    def calculate_pareto_fronts(self):
        Sp_ = []
        Np = []
        for fitnesses_1 in self.all_fitnesses:
            Sp = set()
            Np.append(0)

            for i, fitnesses_2 in enumerate(self.all_fitnesses):
                if self.dominates(fitnesses_1, fitnesses_2):
                    Sp.add(self.keys[i])
                    # print("Sp : ", Sp)
                elif self.dominates(fitnesses_2, fitnesses_1):
                    Np[-1] += 1
            # print("\ndomination_counts(Np) : ", Np)
            Sp_.append(Sp)
            # print("domination_sets(Sp) : ", Sp_)

        Np = np.array(Np)
        # print("domination_counts(Np) : ", Np)

        # ------------------------------------------------------------------------#


        while True:

            current_front = np.where(Np == 0)[0]

            if len(current_front) == 0:
                break
            self.fronts.append(current_front)

            for individual in current_front:
                Np[individual] = -1

                dominated_by_current_set = Sp_[individual]
                # print("dominated_by_current_set", dominated_by_current_set)

                for i, dominated_by_current in enumerate(dominated_by_current_set):
                    # print(dominated_by_current)
                    # print(keys.index(dominated_by_current))
                    indx = self.keys.index(dominated_by_current)
                    Np[indx] -= 1
        return self.fronts

    def calculate_crowding_metrics(self, fronts):
        self.all_fitnesses = np.array(self.all_fitnesses)
        num_objectives = self.all_fitnesses.shape[1]
        num_individuals = self.all_fitnesses.shape[0]

        normalized_fitnesses = np.zeros_like(self.all_fitnesses)
        for objective_i in range(num_objectives):
            min_val = np.min(self.all_fitnesses[:, objective_i])
            max_val = np.max(self.all_fitnesses[:, objective_i])

            val_range = max_val - min_val
            normalized_fitnesses[:, objective_i] = (self.all_fitnesses[:,objective_i] - min_val) / val_range  # normalize each obj

        fitnesses = normalized_fitnesses
        crowding_metrics = np.zeros(num_individuals)

        for front in fronts:
            # print(front)
            for objective_i in range(num_objectives):
                sorted_front = sorted(front, key=lambda x: fitnesses[x, objective_i])
                # print("\nsorted_front", sorted_front)
                crowding_metrics[sorted_front[0]] = np.inf
                crowding_metrics[sorted_front[-1]] = np.inf
                # print("crowding_metrics", crowding_metrics)
                if len(sorted_front) > 2:
                    for i in range(1, len(sorted_front) - 1):
                        crowding_metrics[sorted_front[i]] += fitnesses[sorted_front[i + 1], objective_i] - fitnesses[sorted_front[i - 1], objective_i]

        return crowding_metrics, sorted_front, self.all_fitnesses

    def fronts_to_nondomination_rank(self):
        nondomination_rank_dict = {}
        nondomination_rank_dict_ind = {}

        for i, front in enumerate(self.fronts):  # i = rank
            for x in front:
                nondomination_rank_dict_ind[x] = i
                nondomination_rank_dict[self.keys[x]] = i
        # print(nondomination_rank_dict)

        return nondomination_rank_dict_ind

    def nondominated_sort(self,nondomination_rank_dict, crowding):

        num_individuals = len(crowding)
        indicies = list(range(num_individuals))

        def nondominated_compare(a, b):

            # returns 1 if a dominates b, or if they equal, but a is less crowded
            # return -1 if b dominates a, or if they equal, but b is less crowded
            # returns 0 if they are equal in every sense

            if nondomination_rank_dict[a] > nondomination_rank_dict[b]:  # domination rank, smaller better
                return -1
            elif nondomination_rank_dict[a] < nondomination_rank_dict[b]:
                return 1
            else:
                if crowding[a] < crowding[b]:  # crowding metrics, larger better
                    return -1
                elif crowding[a] > crowding[b]:
                    return 1
                else:
                    return 0

        self.non_domiated_sorted_indicies = sorted(indicies, key=functools.cmp_to_key(nondominated_compare),
                                              reverse=True)  # decreasing order, the best is the first
        # print(self.non_domiated_sorted_indicies)

        return self.non_domiated_sorted_indicies

    def Selected(self):
        for items in self.non_domiated_sorted_indicies:
            self.Amoebanet_sorted.append(self.keys[items])
        Amoebanet_selected = self.Amoebanet_sorted[:len(self.Amoebanet_sorted) // 2]

        selected_geno_str = []
        # selected_geno = []
        selected_fitness = []
        for i in Amoebanet_selected:
            strnodes = self.chromosome_nodes.get(i)
            # numnodes = self.num_chromosome_nodes.get(i)
            accnodes = self.Transfomer.get(i)
            # selected_geno.append(numnodes)
            selected_fitness.append(accnodes)
            selected_geno_str.append(strnodes)
        return selected_fitness, selected_geno_str



