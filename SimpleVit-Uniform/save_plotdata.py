class Savefileplot:
    def __init__(self,generation,all_fitnesses,fronts,first_f,first_value,second_f,second_f_value):

        self.generation = generation
        self.all_fitnesses = all_fitnesses
        self.fronts = fronts
        self.all_data_plot  = {}

        self.first_f = first_f
        self.first_value = first_value
        self.second_f = second_f
        self.second_f_value = second_f_value

    def save_data_plot(self):
            node_gen = f'Fronts{self.generation}'

            self.all_data_plot[node_gen] = {
                'Fronts': self.fronts,
                'Fitnesses': self.all_fitnesses,
                'First_front' : self.first_f,
                'First_value' : self.first_value,
                'Second_front' : self.second_f,
                'Second_value' : self.second_f_value
            }

            file_path = f'Fronts{self.generation}.txt'
            self.write_to_file_plot(file_path, self.all_data_plot)


    def write_to_file_plot(self, file_path, data):
        with open(file_path, 'w') as txt_file:
            for member, data in self.all_data_plot.items():
                txt_file.write(f"{member}:\n")
                for key, value in data.items():
                    txt_file.write(f"  {key}= {value}\n")
                txt_file.write("\n")

