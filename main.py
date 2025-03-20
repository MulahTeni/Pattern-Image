import logging
import random
from functions import *
import config
import matplotlib.pyplot as plt

# Matplotlib interaktif mod
plt.ion()
fig, axes = plt.subplots(2, 3, figsize=(12, 8))

# Log ayarları
logging.basicConfig(filename='algorithm.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    images = convert_to_binary(filter_images_by_size(load_images_from_folder(config.data_path), config.target_size))
    
    population = [(generate_individual(config.pattern_count), 0) for _ in range(config.population_size)]
    for i in range(len(population)):
        population[i] = (population[i][0], fitness(population[i][0], images))
    
    best_fitness_history = []
    avg_fitness_history = []
    len_population = []
    
    for generation in range(config.iteration):
        population = natural_selection(
            sort_population(
                generate_new_generation(
                    population, images, config.pattern_shape, config.pattern_count, 
                    config.adaptation_rate, config.adaptation_cap, config.mutation_rate, 
                    int(config.bred_rate * config.population_size), int(config.fresh_population_rate * config.population_size)
                )
            ), config.population_size
        )
        
        best_fitness = population[0][1]
        avg_fitness = sum(ind[1] for ind in population) / len(population)
        
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        len_population.append(len(population) / config.population_cap)
        
        plot_everything(plt, axes, generation, len(population), len_population, best_fitness_history, avg_fitness_history, generation, [], [], reconstruct_images(population[0][0], images), config.mutation_rate)
        
        save_generation_results(len_population, best_fitness_history, avg_fitness_history, [], [], reconstruct_images(population[0][0], images), population[0][0], generation, config.output_path)
    
    plt.ioff()
    plt.close('all')
