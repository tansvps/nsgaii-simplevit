import random
import string
from mutation_simplevit_V2 import Mutation
from nsgaii_simplevit_V2 import Selection_NSGAII
# from torchsummary import summary
from poplulation_simplevit_V2_dropout import Population
import sys
import torch
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
# from crossover_simplevit_V5 import Crossover
from crossover_uniform import Crossover
from file_simplevit import Savefile
from fileplot_simplevit import Savefileplot
from train_test_V8 import Train
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs('data', exist_ok=True)
########################################
batch_size = 128
print("bz_:",batch_size)

transform_train = transforms.Compose([transforms.Resize((32,32)),  #resises the image so it can be perfect for our model.
                                      transforms.RandomHorizontalFlip(), # FLips the image w.r.t horizontal axis
                                      transforms.RandomRotation(10),     #Rotates the image to a specified angel
                                      transforms.RandomAffine(0, shear=10, scale=(0.8,1.2)), #Performs actions like zooms, change shear angles.
                                      transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Set the color params
                                      transforms.ToTensor(), # comvert the image to tensor so that it can work with torch
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) #Normalize all the images
                               ])
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = datasets.CIFAR10(root='data', train=True, download=True, transform=transform_train)
valid_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)
# test_dataset = datasets.CIFAR10(root='data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=2)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=2)
# test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False,pin_memory=True,num_workers=2)
######################################
latest_pops_ = []
num_generations = 65
init_pops = 8
latest_pops = []
acc_test = []
parameter = []
all_acc = []
all_param = []
transformer_types = []
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

        type_transfm = []
        transformers_types = []
        for geno_m in latest_pops:
            for i in geno_m:
                if i[0] in [1, 2, 4, 8, 16]:
                    if i[1] in [1, 2, 4, 8, 16]:
                        type_tr = 'Transformer_AA'
                        type_transfm.append(type_tr)
                    else:
                        type_tr = 'Transformer_AF'
                        type_transfm.append(type_tr)
                else:
                    if i[1] in [128, 256, 512, 1024, 2048, 4096]:
                        type_tr = 'Transformer_FF'
                        type_transfm.append(type_tr)
                    else:
                        type_tr = 'Transformer_FA'
                        type_transfm.append(type_tr)

            # print("chormosome:", geno_m)
            # print("type_transfm:", type_transfm, len(type_transfm))
            # transformers_types.append(type_transfm[:])
            transformers_types.append(type_transfm.copy())
            type_transfm.clear()

# for generation in range(num_generations):
for generation in range(start_generation,num_generations):
    if file_path == f'output_generation{generation-1}.txt':
        print("continue...")
    else:
        pass

    print(f"\nGeneration {generation}\n-------------------------------")
    if generation == 1:
        genotype = []
        chrom_tuple = []
        list_genotype = []
        d = [3, 4, 5, 6, 7, 8, 9, 10]
        random.shuffle(d)
        for k in range(init_pops):
            print(f"\nIndividual {str(k + 1)} / {str(init_pops)}")
            while True:
                try:
                    population_instance = Population()
                    init_chromosome,type_transfm,depth = population_instance.random_pop(d[k])
                    print("Chromosome",init_chromosome)
                    # print("type_transfm", type_transfm)
                    chrom_tuple.append(init_chromosome)
                    list_genotype.append(init_chromosome)
                    model,param = population_instance.transfer_genotype(genotype=init_chromosome,type_transfm=type_transfm,depth=depth)
                    model_train = Train()
                    acc = model_train.traintest(model, train_loader, valid_loader)
                    # acc = acc.numpy()
                    acc = round(acc, 4)
                    print("Accuracy:",acc)
                    print("Parameter:",param)
                    parameter.append(param)
                    acc_test.append(acc)
                    break
                except Exception as e:
                    print(f"ERROR: {e}. Retrying...")
                    continue

        # print("\nAccuracy:", acc_test, len(acc_test))
        # print("Parameter:", parameter, len(parameter))

        print("\n--Crossover and Mutation Process--")
        crossover_instance = Crossover()
        child = crossover_instance.pair(chrom_tuple)
        # print("Child:",child)

        for k, i in enumerate(child):
            print(f"\nIndividual {str(k + 1)} / {str(len(child))}")
            while True:
                try:
                    print("Chromosome:",i)
                    mutate = Mutation()
                    genotype_mutated = mutate.do_mutation(genotype=i)
                    print("Chromosome_mutated:",genotype_mutated)
                    depth_ = len(genotype_mutated)
                    # print("len",depth_)
                    type_transfm.clear()
                    for geno_m in genotype_mutated:
                        if geno_m[0] in [1, 2, 4, 8, 16]:
                            if geno_m[1] in [1, 2, 4, 8, 16]:
                                type_tr = 'Transformer_AA'
                                type_transfm.append(type_tr)
                            else:
                                type_tr = 'Transformer_AF'
                                type_transfm.append(type_tr)
                        else:
                            if geno_m[1] in [128, 256, 512, 1024, 2048, 4096]:
                                type_tr = 'Transformer_FF'
                                type_transfm.append(type_tr)
                            else:
                                type_tr = 'Transformer_FA'
                                type_transfm.append(type_tr)
                    # modelpytorch
                    population_instance = Population()
                    model, param = population_instance.transfer_genotype(genotype=genotype_mutated,type_transfm=type_transfm,depth=depth_)
                    # model = model.to(device)
                    model_train = Train()
                    acc = model_train.traintest(model, train_loader, valid_loader)
                    # acc = acc.numpy()
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


        ##NSGAII####
        selected = Selection_NSGAII(list_genotype, acc_test, parameter)
        acc_param, fitnesses_keep_acc, fitnesses_keep_param = selected.normolize()
        chromosome_nodes, Transfomer, genotype_list, all_fitnesses = selected.param()

        fronts = selected.calculate_pareto_fronts()
        crowding_metrics, sorted_front, all_fitnesses = selected.calculate_crowding_metrics(fronts)
        nondomination_rank_dict = selected.fronts_to_nondomination_rank()
        sorted_indicies = selected.nondominated_sort(nondomination_rank_dict, crowding_metrics)
        # print("sorted_indicies:",sorted_indicies)
        fitness, geno_nextgen = selected.Selected()
        print("\nChromosome_nextgeneration:",geno_nextgen, len(geno_nextgen))

        for i in range(len(fronts)):
            if i==0:
                sorted_front = sorted(fronts[i], key=lambda x: acc_param[x, 0])
                first_f = sorted_front
                first_value = acc_param[sorted_front]
            if i==1:
                sorted_front = sorted(fronts[i], key=lambda x: acc_param[x, 0])
                second_f = sorted_front
                second_f_value = acc_param[sorted_front]
        
        acc_test.clear()
        parameter.clear()
        acc = []
        para = []
        for row in fitness:
            acc.append(row[0])
            para.append(row[1])

        file_ac = []
        for ac in acc:
            for a, b in fitnesses_keep_acc:
                if ac == b:
                    acc_test.append(a)
                    break

        file_ac = acc_test

        file_pr = []
        for pr in para:
            for i, j in fitnesses_keep_param:
                if pr == j:
                    parameter.append(i)
                    break

        file_pr = parameter
        latest_pops = geno_nextgen
        file_latest_pops = latest_pops
        print("Accuracy_nextgen:",file_ac)
        print("Parameter_nextgen:",file_pr)
        # #
    
        print(f"Saving files for Generation {generation}...")
        with open(f'output_generation{generation}.txt', 'w') as file:
            file.write(str(latest_pops) + '\n')
            file.write(str(file_ac) + '\n')
            file.write(str(file_pr))
        saver = Savefile(file_latest_pops, file_pr, file_ac, generation)
        saver.save_data()

        save_p = Savefileplot(generation, acc_param, fronts, first_f, first_value, second_f, second_f_value)
        save_p.save_data_plot()



    else:

        crossover_instance = Crossover()
        child = crossover_instance.pair(latest_pops)
        # print("Child:", child,len(child))
        list_mutate = []
        latestpops_check = latest_pops.copy()
    
        
        for k, i in enumerate(child):
            print(f"\nIndividual {str(k + 1)} / {str(len(child))}")
            # print("Chromosome:",i)
            mutate = Mutation()
            MAX_ATTEMPTS = 4
            while True:
                try:
                    type_transfm.clear()
                    print("Chromosome:", i)
                    genotype_mutated = mutate.do_mutation(genotype=i)
                    print("Chromosome_mutated: ", genotype_mutated)
                    #check
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
                            # print("Crossover and mutation process successful!")
                            latestpops_check.append(genotype_mutated)
                            break
                            
                    
                    for geno_m in genotype_mutated:
                        if geno_m[0] in [1, 2, 4, 8, 16]:
                            if geno_m[1] in [1, 2, 4, 8, 16]:
                                type_tr = 'Transformer_AA'
                                type_transfm.append(type_tr)
                            else:
                                type_tr = 'Transformer_AF'
                                type_transfm.append(type_tr)
                        else:
                            if geno_m[1] in [128, 256, 512, 1024, 2048, 4096]:
                                type_tr = 'Transformer_FF'
                                type_transfm.append(type_tr)
                            else:
                                type_tr = 'Transformer_FA'
                                type_transfm.append(type_tr)

                    depth_ = len(genotype_mutated)
                    # print("len", depth_)

                    
                    population_instance = Population()
                    model, param = population_instance.transfer_genotype(genotype=genotype_mutated,type_transfm=type_transfm,depth=depth_)
                    model_train = Train()
                    # model = model.to(device)
                    acc = model_train.traintest(model, train_loader, valid_loader)
                    # acc = acc.numpy()
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

        ##########################################################

        # print("\nAccuracy:", acc_test, len(acc_test))
        # print("Parameter:", parameter, len(parameter))
        list_genotype = latest_pops + list_mutate
        # print("list_genotype:", list_genotype) #old+new
        # print(len(list_genotype))

        ##NSGAII####
        selected = Selection_NSGAII(list_genotype, acc_test, parameter)

        acc_param, fitnesses_keep_acc, fitnesses_keep_param = selected.normolize()
        chromosome_nodes, Transfomer, genotype_list, all_fitnesses = selected.param()
        # print("\nchromosome_nodes:", chromosome_nodes)
        # print("Transfomer:", Transfomer)
        # print("genotype_str:", genotype_list)
        # print("all_fitnesses:", all_fitnesses)

        fronts = selected.calculate_pareto_fronts()
        crowding_metrics, sorted_front, all_fitnesses = selected.calculate_crowding_metrics(fronts)
        nondomination_rank_dict = selected.fronts_to_nondomination_rank()
        sorted_indicies = selected.nondominated_sort(nondomination_rank_dict, crowding_metrics)
        # print("sorted_indicies", sorted_indicies)
        fitness, geno_nextgen = selected.Selected()
        print("\nChromosome_nextgeneration:", geno_nextgen, len(geno_nextgen))

        for i in range(len(fronts)):
            if i==0:
                sorted_front = sorted(fronts[i], key=lambda x: acc_param[x, 0])
                first_f = sorted_front
                first_value = acc_param[sorted_front]
            if i==1:
                sorted_front = sorted(fronts[i], key=lambda x: acc_param[x, 0])
                second_f = sorted_front
                second_f_value = acc_param[sorted_front]
        
        acc_test.clear()
        parameter.clear()
        acc = []
        para = []
        for row in fitness:
            acc.append(row[0])
            para.append(row[1])

        file_ac = []
        for ac in acc:
            for a, b in fitnesses_keep_acc:
                if ac == b:
                    acc_test.append(a)
                    break

        file_ac = acc_test

        file_pr = []
        for pr in para:
            for i, j in fitnesses_keep_param:
                if pr == j:
                    parameter.append(i)
                    break

        latest_pops = geno_nextgen
        file_pr = parameter
        file_latest_pops = latest_pops
        print("Accuracy_nextgen:",file_ac)
        print("Parameter_nextgen:",file_pr)

        # with open('latest_generation.txt', 'w') as file:
        #     file.write(f"Generation: {generation}\n")
        #     file.write(str(latest_pops) + '\n')
        #     file.write(str(file_ac) + '\n')
        #     file.write(str(file_pr) + '\n')

        if generation % 8 == 0:
            print(f"Saving files for Generation {generation}...")
            # with open(f'output_generation{generation}.txt', 'w') as file:
            #     file.write(str(latest_pops))
            with open(f'output_generation{generation}.txt', 'w') as file:
                file.write(str(latest_pops) + '\n')
                file.write(str(file_ac) + '\n')
                file.write(str(file_pr))
        
        if generation % 8 == 0:
            saver = Savefile(file_latest_pops, file_pr, file_ac, generation)
            saver.save_data()

            save_p = Savefileplot(generation, acc_param, fronts, first_f, first_value, second_f, second_f_value)
            save_p.save_data_plot()


