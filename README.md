# NSGA-II for Vision Transformer Architecture Search

This project applies a multi-objective genetic algorithm (NSGA-II) to evolve Vision Transformer (ViT) architectures on image classification tasks. The goal is to discover architectures that balance **model accuracy** and **model size** (number of parameters).

---

## 🔍 Objectives

- Apply NSGA-II to search for optimal ViT configurations
- Use two-point crossover and mutation operations on architecture representations
- Evaluate each individual (chromosome) on CIFAR-10 using SimpleViT model
- Track the trade-off between accuracy and model complexity (Pareto front)

## 🗂 Project Structure

The project consists of two variants of ViT architecture search using NSGA-II:
- `SimpleViT-2point/` uses **two-point crossover**
- `SimpleViT-Uniform/` uses **uniform crossover**

Both folders share the same structure, differing only in `crossover.py`.

SimpleViT-*/ # Shared folder structure between both versions
├── main.py # Entry point for running the NSGA-II optimization loop
├── crossover.py # Implements the crossover strategy (differs per version)
├── mutation.py # Mutation logic for modifying chromosomes
├── nsgaii.py # Core NSGA-II workflow (sorting, selection, reproduction)
├── population.py # Population initialization and evolution logic
├── train_eval.py # Training and evaluation of generated ViT architectures
├── simplevit_model.py # Definition of the SimpleViT model architecture
├── save_population.py # Saves chromosome, accuracy, and model size for each generation
├── save_fronts.py # Extracts and stores Pareto fronts for analysis
├── Result/ # Output folder containing Generation files and result plots
