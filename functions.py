import numpy as np
from PIL import Image
import random
import os
import cv2
import matplotlib.pyplot as plt

def save_generation_results(len_population, best_fitness_history, avg_fitness_history, mutation_updates, adaptation_updates, best_reconstructed_images, best_individual, generation, output_dir="result"):
    os.makedirs(output_dir, exist_ok=True)
    
    graph_path = os.path.join(output_dir, f"graph_{generation}_6.png")
    plt.savefig(graph_path)
    
    # Fitness grafiğini oluştur ve kaydet (Sadece axis[0,0])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(len_population, label='Space Fullness', color='black')
    ax.plot(avg_fitness_history, label='Avg Fitness', color='red')
    ax.plot(best_fitness_history, label='Best Fitness', color='blue')
    ax.set_title(f"Generation {generation}")
    
    for gen in mutation_updates:
        ax.axvline(x=gen, color='purple', linestyle='--', linewidth=1, label='Mutation Update' if gen == mutation_updates[0] else "")
    
    for gen in adaptation_updates:
        ax.axvline(x=gen, color='green', linestyle='--', linewidth=1, label='Adaptation Update' if gen == adaptation_updates[0] else "")
    
    ax.legend()
    
    graph_path = os.path.join(output_dir, f"graph_{generation}.png")
    plt.savefig(graph_path)
    plt.close(fig)
    #print(f"Saved graph: {graph_path}")
    
    if (generation + 1) % 50 == 0 or generation == 1:
        generation_dir = os.path.join(output_dir, f"generation_{generation}")
        os.makedirs(generation_dir, exist_ok=True)
    
        graph_path = os.path.join(generation_dir, f"graph.png")
        plt.savefig(graph_path)
        plt.close(fig)
        print(f"Saved graph: {graph_path}")
        
        # En iyi bireyin ürettiği resimleri kaydet
        for idx, img in enumerate(best_reconstructed_images):
            resized_img = cv2.resize(img, (960, 960), interpolation=cv2.INTER_NEAREST)
            img_path = os.path.join(generation_dir, f"reconstructed_{idx}.png")
            plt.imsave(img_path, resized_img, cmap='gray')
            print(f"Saved reconstructed image: {img_path}")
        
        # En iyi bireyin patternlerini kaydet
        for idx, pattern in enumerate(best_individual):
            resized_pattern = cv2.resize(pattern, (120, 120), interpolation=cv2.INTER_NEAREST)
            pattern_path = os.path.join(generation_dir, f"pattern_{idx}.png")
            plt.imsave(pattern_path, resized_pattern, cmap='gray')

            print(f"Saved pattern: {pattern_path}")

def plot_everything(pltt, axes, ite, pop_curr, len_population, best_fitness_history, avg_fitness_history, generation, mutation_updates, adaptation_updates, best_reconstructed_images, mutation_rate):
    axes[0, 0].clear()
    axes[0, 0].plot(len_population, label='Space Fullness', color='black')
    axes[0, 0].plot(avg_fitness_history, label='Avg Fitness', color='red')
    axes[0, 0].plot(best_fitness_history, label='Best Fitness', color='blue')
    axes[0, 0].set_title(f"Generation {generation}")
    
    for gen in mutation_updates:
        axes[0, 0].axvline(x=gen, color='purple', linestyle='--', linewidth=1, label='Mutation Update' if gen == mutation_updates[0] else "")
    
    for gen in adaptation_updates:
        axes[0, 0].axvline(x=gen, color='green', linestyle='--', linewidth=1, label='Adaptation Update' if gen == adaptation_updates[0] else "")
    
    axes[0, 0].legend()
        
    max_images = axes.shape[1]
    for i in range(1, 3):
        axes[0, i].clear()
        axes[0, i].imshow(best_reconstructed_images[i - 1], cmap='gray')
        axes[0, i].axis('off')

    for i in range(3):
        axes[1, i].clear()
        axes[1, i].imshow(best_reconstructed_images[2 + i], cmap='gray')
        axes[1, i].axis('off')
    
    
    axes[0, 1].set_title(f"Dynamic population and mutation")
    axes[0, 2].set_title(f"Adaptation allowed")
    
    axes[1, 0].set_title(f"Generation: {ite}")
    axes[1, 1].set_title(f"Population: {pop_curr}")
    axes[1, 2].set_title(f"Mutation chance: {mutation_rate}")
    
    pltt.pause(0.01)

def reconstruct_images(individual, original_images):
    if not individual:
        return [np.zeros_like(img) for img in original_images]

    reconstructed_images = []
    
    for img in original_images:
        new_img_array = np.zeros_like(img)

        for i in range(0, img.shape[0] - 2, 3):
            for j in range(0, img.shape[1] - 2, 3):
                block = img[i:i+3, j:j+3]
                
                best_pattern = find_best_pattern_for_block(block, individual)
                new_img_array[i:i+3, j:j+3] = best_pattern

        reconstructed_images.append(new_img_array)
    
    return reconstructed_images


def crossover(parent1, parent2):
    crossover_point = random.randint(1, len(parent1) - 1)
    
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    
    return child

def find_best_pattern_for_block(block, individual):
    best_pattern = None
    min_distance = float('inf')
    
    for pattern in individual:
        distance = hamming_distance(block, pattern)
        
        if distance < min_distance:
            min_distance = distance
            best_pattern = pattern
    
    return best_pattern
    
def adapt(individual, images, pattern_shape):
    if not individual:
        return individual  # Eğer birey boşsa direkt döndür

    pattern_usage = {tuple(pattern.flatten()): 0 for pattern in individual}

    for binary_img in images:
        height, width = binary_img.shape
        for i in range(0, height - 2, 3):
            for j in range(0, width - 2, 3):
                block = binary_img[i:i+3, j:j+3]
                
                best_pattern = find_best_pattern_for_block(block, individual)

                pattern_tuple = tuple(best_pattern.flatten())
                if pattern_tuple in pattern_usage:
                    pattern_usage[pattern_tuple] += 1

    least_used_pattern = min(pattern_usage, key=pattern_usage.get) if pattern_usage else None

    if least_used_pattern is not None and pattern_usage[least_used_pattern] > 0:
        new_pattern = generate_random_pattern(pattern_shape)
        
        for i in range(len(individual)):
            if np.array_equal(individual[i], np.array(least_used_pattern).reshape(3, 3)):
                individual[i] = new_pattern
                return individual
    else:
        random_index = random.randint(0, len(individual) - 1)
        individual[random_index] = generate_random_pattern(pattern_shape)

    return individual

def sort_population(population):
    population.sort(key=lambda x: x[1], reverse=True)
    return population

def natural_selection(population, population_cap):
    if not population:
        return []
    return population[:population_cap]
    
def generate_new_generation(population, images, pattern_shape, pattern_count, adaptation_rate, adaptation_cap, mutation_rate, bred_count, fresh_population_count):
    new_generation = []
    
    max_fitness = max(population, key=lambda x: x[1])[1]
    for individual, score in population:
        adaptation_probability = (max_fitness - score) / max_fitness
        final_adaptation_rate = min(adaptation_rate * adaptation_probability, adaptation_cap)
        
        if random.random() < final_adaptation_rate:
            adapted_individual = adapt(individual, images, pattern_shape)
            adapted_score = fitness(adapted_individual, images)
            if adapted_score > score:
                new_generation.append((adapted_individual, adapted_score))
            else:
                new_generation.append((individual, score))
        else:
            new_generation.append((individual, score))
    
    for _ in range(bred_count):
        parents = random.sample(population, 2)
        
        child = crossover(parents[0][0], parents[1][0])
        child = mutate(child, mutation_rate)
        child_score = fitness(child, images)
        new_generation.append((child, child_score))
    
    for _ in range(fresh_population_count):
        new_individual = generate_individual(pattern_count)
        new_score = fitness(new_individual, images)
        new_generation.append((new_individual, new_score))
    
    return new_generation

def mutate(individual, mutation_rate=0.1):
    mutated_individual = []
    for pattern in individual:
        if random.random() < mutation_rate:
            mutation_mask = np.random.rand(*pattern.shape) < mutation_rate
            mutated_pattern = np.where(mutation_mask, 1 - pattern, pattern)
            mutated_individual.append(mutated_pattern)
        else:
            mutated_individual.append(pattern)
    return mutated_individual

def hamming_distance(pattern1, pattern2):
    return np.sum(pattern1 != pattern2)

def fitness(individual, images):
    total_count = 0
    total_pixels = 0
    
    height, width = images[0].shape
    
    for image in images:
        for i in range(0, height - 2, 3):
            for j in range(0, width - 2, 3):
                block = image[i:i+3, j:j+3]
                
                best_pattern = find_best_pattern_for_block(block, individual)
                
                total_count += np.sum(block == best_pattern)
                total_pixels += block.size
                
    return total_count / total_pixels if total_pixels > 0 else 0  # Bölme hatasını önlemek için

def generate_random_pattern(pattern_shape):
    return np.random.randint(2, size=pattern_shape)

def generate_individual(pattern_count, pattern_shape=(3, 3)):
    return [generate_random_pattern(pattern_shape) for _ in range(pattern_count)]

def filter_images_by_size(images, target_size=(24, 24)):
    valid_images = []
    invalid_images = []
    
    for filename, img in images:
        if img.size == target_size:
            valid_images.append((filename, img))
        else:
            invalid_images.append(filename)
    
    if invalid_images:
        print(f"The following images were removed (not {target_size}):")
        for img_name in invalid_images:
            print(img_name)
    
    return [img for _, img in valid_images]

def convert_to_binary(images, threshold=128):
    binary_images = []
    
    for image in images:
        img_gray = image.convert("L")
        img_array = np.array(img_gray)
        
        binary_images.append(np.where(img_array > threshold, 1, 0))
    
    return binary_images

def load_images_from_folder(folder_path):
    images = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        if filename.lower().endswith((".png", ".jpg", ".jpeg")):
            img = Image.open(file_path)
            images.append((filename, img))
    
    return images
