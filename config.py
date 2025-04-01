# config.py


iteration = 200
population_size = 200
pattern_count = 7
mutation_rate = 0.2

group = "group2" # group1, group2, group3
data_path = f"data/{group}" 
output_path = f"{population_size}_{mutation_rate}_{group}"

adaptation_rate = 0.05
bred_rate = 0.5
fresh_population_rate = 0.1

population_flag = False
mutation_flag = False
adaptation_flag = False