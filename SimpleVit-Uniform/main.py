import os
import sys
import random
import string

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from population import Population
from mutation import Mutation
from crossover import Crossover
from train_eval import Train
from nsgaii import Selection_NSGAII
from save_population import Savefile
from utils import get_dataloaders, get_transformer_type,save_generation_data,save_generation
from save_plotdata import Savefileplot
from utils import save_fronts_plot


latest_pops_ = []
num_generations = 65
init_pops = 8
latest_pops = []
acc_test = []
parameter = []
all_acc = []
all_param = []
transformer_types = []

train_loader, valid_loader, test_loader = get_dataloaders(batch_size=64)

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Running the program without command line arguments")
        file_path = 1
        start_generation = 1
        pass

    else:
        file_path = sys.argv[1]
        start_generation = int(sys.argv[2])
        generation = start_generation
        with open(file_path, 'r') as file:
            lines = file.readlines()
        latest_pops = eval(lines[0].strip())
        acc_test = eval(lines[1].strip())
        parameter = eval(lines[2].strip())

        transformers_types = []
        for geno_m in latest_pops:
            type_transfm = get_transformer_type(geno_m)
            transformers_types.append(type_transfm)

            type_transfm.clear()
        print("transformers_types",transformers_types)

for generation in range(start_generation, num_generations):
    if file_path == f'output_generation{generation - 1}.txt':
        print("continue...")
    else:
        pass

    print(f"\nGeneration {generation}\n-------------------------------")
    for generation in range(start_generation, num_generations):
        if file_path == f'output_generation{generation - 1}.txt':
            print("continue...")
        else:
            pass

        print(f"\nGeneration {generation}\n-------------------------------")

        if generation == 1:
            genotype = []
            chrom_tuple = []
            list_genotype = []

            for k in range(init_pops):
                print(f"\nIndividual {str(k + 1)} / {str(init_pops)}")
                while True:
                    try:
                        population_instance = Population()
                        init_chromosome, type_transfm, depth = population_instance.random_pop()
                        print("Chromosome", init_chromosome)
                        chrom_tuple.append(init_chromosome)
                        list_genotype.append(init_chromosome)
                        model, param = population_instance.transfer_genotype(genotype=init_chromosome,type_transfm=type_transfm, depth=depth)
                        model_train = Train()
                        acc = model_train.traintest(model, train_loader, valid_loader)
                        acc = round(acc, 4)
                        print("Accuracy:", acc)
                        print("Parameter:", param)
                        parameter.append(param)
                        acc_test.append(acc)
                        break

                    except Exception as e:
                        print(f"ERROR: {e}. Retrying...")
                        continue

        print("\n--Crossover and Mutation Process--")
        crossover_instance = Crossover()
        child = crossover_instance.pair(chrom_tuple)

        for k, i in enumerate(child):
            print(f"\nIndividual {str(k + 1)} / {str(len(child))}")
            while True:
                try:
                    print("Chromosome:", i)
                    mutate = Mutation()
                    genotype_mutated = mutate.do_mutation(genotype=i)
                    print("Chromosome_mutated:", genotype_mutated)
                    depth_ = len(genotype_mutated)
                    type_transfm.clear()

                    transformers_types = []
                    for geno_m in latest_pops:
                        type_transfm = get_transformer_type(geno_m)
                        transformers_types.append(type_transfm)

                    population_instance = Population()
                    model, param = population_instance.transfer_genotype(genotype=genotype_mutated,type_transfm=type_transfm, depth=depth_)
                    model_train = Train()
                    acc = model_train.traintest(model, train_loader, valid_loader)
                    acc = round(acc, 4)
                    print("accuracy:", acc)
                    print("parameter:", param)
                    acc_test.append(acc)
                    parameter.append(param)
                    list_genotype.append(genotype_mutated)
                    break

                except Exception as e:
                    print(f"ERROR:{e}. Retrying...")
                    continue

        ##-----------NSGAII----------####
        selected = Selection_NSGAII(list_genotype, acc_test, parameter)
        acc_param, fitnesses_keep_acc, fitnesses_keep_param = selected.normolize()
        chromosome_nodes, Transfomer, genotype_list, all_fitnesses = selected.param()

        fronts = selected.calculate_pareto_fronts()
        crowding_metrics, sorted_front, all_fitnesses = selected.calculate_crowding_metrics(fronts)
        nondomination_rank_dict = selected.fronts_to_nondomination_rank()
        sorted_indicies = selected.nondominated_sort(nondomination_rank_dict, crowding_metrics)
        fitness, geno_nextgen = selected.Selected()
        print("\nChromosome_nextgeneration:", geno_nextgen, len(geno_nextgen))
        ##-----------NSGAII----------####

        acc_test.clear()
        parameter.clear()

        acc = [row[0] for row in fitness]
        para = [row[1] for row in fitness]
        fitness_map = {b: a for a, b in fitnesses_keep_acc}

        for ac in acc:
            if ac in fitness_map:
                acc_test.append(fitness_map[ac])

        file_ac = acc_test

        fitness_param_map = {j: i for i, j in fitnesses_keep_param}
        parameter.extend([fitness_param_map[pr] for pr in para if pr in fitness_param_map])
        file_pr = parameter

        latest_pops = geno_nextgen
        file_latest_pops = latest_pops
        print("Accuracy_nextgen:", file_ac)
        print("Parameter_nextgen:", file_pr)

        print(f"Saving files for Generation {generation}...")
        save_generation(generation, latest_pops, file_ac, file_pr, acc_param, fronts)

    else:

        crossover_instance = Crossover()
        child = crossover_instance.pair(latest_pops)
        list_mutate = []
        latestpops_check = latest_pops.copy()

        for k, i in enumerate(child):
            print(f"\nIndividual {str(k + 1)} / {str(len(child))}")
            mutate = Mutation()
            MAX_ATTEMPTS = 3
            while True:
                try:
                    type_transfm.clear()
                    print("Chromosome:", i)
                    genotype_mutated = mutate.do_mutation(genotype=i)
                    print("Chromosome_mutated: ", genotype_mutated)
                    attempt = 0
                    while True:
                        attempt += 1
                        if attempt > MAX_ATTEMPTS:
                            print(f"Max{i}")
                            break
                        if genotype_mutated in latestpops_check:
                            print("Duplicate, retry...")
                            genotype_mutated = mutate.do_mutation_(genotype=i)
                            print("Chromosome_mutated_: ", genotype_mutated)
                        else:
                            latestpops_check.append(genotype_mutated)
                            break

                    transformers_types = []
                    for geno_m in latest_pops:
                        type_transfm = get_transformer_type(geno_m)
                        transformers_types.append(type_transfm)

                    depth_ = len(genotype_mutated)
                    population_instance = Population()
                    model, param = population_instance.transfer_genotype(genotype=genotype_mutated,type_transfm=type_transfm, depth=depth_)
                    model_train = Train()
                    acc = model_train.traintest(model, train_loader, valid_loader)
                    acc = round(acc, 4)
                    print("Accuracy:", acc)
                    print("Parameter:", param)
                    acc_test.append(acc)
                    parameter.append(param)
                    list_mutate.append(genotype_mutated)
                    break

                except(IndexError, RuntimeError):
                    print("IndexError, retrying...")
                    continue  # retry mutation

        list_genotype = latest_pops + list_mutate

        ##-----------------NSGAII------------------####
        selected = Selection_NSGAII(list_genotype, acc_test, parameter)

        acc_param, fitnesses_keep_acc, fitnesses_keep_param = selected.normolize()
        chromosome_nodes, Transfomer, genotype_list, all_fitnesses = selected.param()
        fronts = selected.calculate_pareto_fronts()
        crowding_metrics, sorted_front, all_fitnesses = selected.calculate_crowding_metrics(fronts)
        nondomination_rank_dict = selected.fronts_to_nondomination_rank()
        sorted_indicies = selected.nondominated_sort(nondomination_rank_dict, crowding_metrics)
        # print("sorted_indicies", sorted_indicies)
        fitness, geno_nextgen = selected.Selected()
        print("\nChromosome_nextgeneration:", geno_nextgen, len(geno_nextgen))
        ##-----------------NSGAII------------------####

        acc_test.clear()
        parameter.clear()

        acc = [row[0] for row in fitness]
        para = [row[1] for row in fitness]
        fitness_map = {b: a for a, b in fitnesses_keep_acc}

        for ac in acc:
            if ac in fitness_map:
                acc_test.append(fitness_map[ac])

        file_ac = acc_test
        fitness_param_map = {j: i for i, j in fitnesses_keep_param}
        parameter.extend([fitness_param_map[pr] for pr in para if pr in fitness_param_map])
        file_pr = parameter

        latest_pops = geno_nextgen
        file_latest_pops = latest_pops
        print("Accuracy_nextgen:", file_ac)
        print("Parameter_nextgen:", file_pr)

        if generation % 8 == 0:
            print(f"Saving files for Generation {generation}...")
            save_generation(generation, latest_pops, file_ac, file_pr, acc_param, fronts)



