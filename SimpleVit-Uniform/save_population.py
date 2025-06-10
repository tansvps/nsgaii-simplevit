class Savefile:
    def __init__(self, file_latest_pops, file_pr, file_ac, generation):
        self.file_latest_pops = file_latest_pops
        self.file_pr = file_pr
        self.file_ac = file_ac
        self.all_generations = {}
        self.generation = generation

    def save_data(self):
        for i, (latest_pop, pr, ac) in enumerate(zip(self.file_latest_pops, self.file_pr, self.file_ac), start=1):
            node = f'Transformers{i}'

            self.all_generations[node] = {
                'Chromosome': latest_pop,
                'Parameters': pr,
                'Accuracy': ac
            }

            file_path = f'Generation{self.generation}.txt'
            self.write_to_file(file_path, self.all_generations[node])

    def write_to_file(self, file_path, data):
        with open(file_path, 'w') as txt_file:
            for member, data in self.all_generations.items():
                txt_file.write(f"{member}:\n")
                for key, value in data.items():
                    txt_file.write(f"  {key}: {value}\n")
                txt_file.write("\n")

