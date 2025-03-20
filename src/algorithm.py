import numpy as np
import logging
from src.image_operations import find_best_pattern_for_block, calculate_loss, calculate_accuracy, convert_to_binary
import random
import matplotlib.pyplot as plt
import os

# Log ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_new_populations(populations, accuracies, population_size, mutation_rate):
    """
    Mevcut popülasyonlardan yeni popülasyonlar üretmek için elitizm, çaprazlama ve mutasyon uygular.
    """
    new_populations = []
    
    # En iyi %20'yi bul ve doğrudan yeni nesle geçir
    elite_count = max(1, int(0.2 * population_size))  # En az 1 birey garanti ediliyor
    sorted_indices = np.argsort(accuracies)[::-1]  # Accuracy'ye göre azalan sırayla indeksler
    elite_individuals = [populations[i] for i in sorted_indices[:elite_count]]

    new_populations.extend(elite_individuals)  # Elite bireyleri ekle

    # Çaprazlama ve mutasyon işlemleri ile kalan popülasyonu doldur
    while len(new_populations) < population_size:
        parent1, parent2 = random.sample(populations, 2)  # Rastgele iki ebeveyn seç
        child = crossover(parent1, parent2)  # Çaprazlama işlemi
        mutated_child = mutation(child, mutation_rate)  # Mutasyon işlemi
        new_populations.append(mutated_child)

    return new_populations


def crossover(parent1, parent2):
    """
    İki ebeveynden bir çocuk birey üretir. Çaprazlama işlemi ile yapılıyor.
    """
    crossover_point = random.randint(1, len(parent1) - 1)
    
    # Çocuğu oluştur: ilk kısım birinci ebeveynden, ikinci kısım ikinci ebeveynden
    child = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    
    return child

def mutation(individual, mutation_rate=0.1):
    """
    Mutasyon işlemi. Bireyin genetik yapısını değiştirir.
    """
    mutated_individual = individual.copy()  # Orijinal bireyi değiştirmemek için kopya al
    for i in range(len(mutated_individual)):
        if random.random() < mutation_rate:  # Mutasyon oranı kadar değişim
            mutated_individual[i] = 1 - mutated_individual[i]  # Bit değeri tersine çevir (0 -> 1, 1 -> 0)
    return mutated_individual


def algorithm(images, population, iterations):
    """
    Genetik algoritmanın bir nesli boyunca çalışacak ana fonksiyon.
    """
    total_loss = 0
    total_accuracy = 0
    
    # Resimleri gezerek her birinin pattern'lerini bul
    for img in images:
        # Yeni bir boş resim oluştur (pattern'lerle doldurulacak)
        new_img_array = np.zeros_like(img)

        # Resmin her 3x3'lük bloğuna gez
        for i in range(0, img.shape[0] - 2, 3):  # 24x24 boyutlu bir resim için
            for j in range(0, img.shape[1] - 2, 3):
                # 3x3'lük bloğu al
                block = img[i:i+3, j:j+3]
                
                # En uygun pattern'i bul
                best_pattern = find_best_pattern_for_block(block, population)
                
                # Yeni resme en iyi pattern'i yerleştir
                new_img_array[i:i+3, j:j+3] = best_pattern

        # Loss ve Accuracy hesaplama
        loss = calculate_loss(img, new_img_array)
        accuracy = calculate_accuracy(img, new_img_array)

        logger.info(f"Loss for image: {loss}, Accuracy for image: {accuracy}")

        total_loss += loss
        total_accuracy += accuracy
        
    return total_loss, total_accuracy

def save_pattern_images(images, population, ite_output_dir):
    """
    En iyi accuracy'yi veren popülasyonun pattern'leriyle 5 resim kaydeder.
    """

    for im, img in enumerate(images):
        # Yeni bir boş resim oluştur (pattern'lerle doldurulacak)
        new_img_array = np.zeros_like(img)

        # Resmin her 3x3'lük bloğuna gez
        for i in range(0, img.shape[0] - 2, 3):  # 24x24 boyutlu bir resim için
            for j in range(0, img.shape[1] - 2, 3):
                # 3x3'lük bloğu al
                block = img[i:i+3, j:j+3]
                
                # En uygun pattern'i bul
                best_pattern = find_best_pattern_for_block(block, population)
                
                # Yeni resme en iyi pattern'i yerleştir
                new_img_array[i:i+3, j:j+3] = best_pattern


        # Resmi kaydet
        img_path = os.path.join(ite_output_dir, f"generated_image_{im + 1}.png")
        plt.imsave(img_path, new_img_array, cmap='gray')
        logger.info(f"Saved generated image: {img_path}")

def save_best_population_patterns(population, output_dir):
    """
    En iyi popülasyonun 7 adet 3x3'lük pattern'lerini 24x24 boyutunda ayrı görüntüler olarak kaydeder.
    """
    os.makedirs(output_dir, exist_ok=True)  # Çıktı klasörü oluştur

    for idx, pattern in enumerate(population):  
        pattern_matrix = np.array(pattern).reshape(3, 3)  # 3x3 olarak düzenle

        # 3x3 pattern'i 24x24 olarak büyüt
        scaled_pattern = np.kron(pattern_matrix, np.ones((8, 8)))  # 3x3 → 24x24 büyütme

        # Görselleştirme ve kaydetme
        plt.figure(figsize=(3, 3))
        plt.imshow(scaled_pattern, cmap="gray")
        plt.axis("off")  # Eksenleri kaldır

        # Dosya kaydetme
        pattern_image_path = os.path.join(output_dir, f"best_pattern_{idx + 1}.png")
        plt.savefig(pattern_image_path, bbox_inches="tight", pad_inches=0)
        plt.close()

        logger.info(f"Saved pattern {idx + 1} as an image: {pattern_image_path}")


# İlk başta tek bir figür oluştur
plt.ion()  # Interaktif modu aç
fig, ax1 = plt.subplots()  # Tek bir figür oluştur
ax2 = ax1.twinx()  # İkinci y ekseni için

def plot_results(loss_per_generation, accuracy_per_generation):
    """
    Loss ve accuracy değerlerini tek bir figürde günceller.
    """
    ax1.clear()  # Önceki çizimi temizle
    ax2.clear()

    generations = range(1, len(loss_per_generation) + 1)

    # Loss grafiği
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color='tab:red')
    ax1.plot(generations, loss_per_generation, color='tab:red', label='Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Accuracy grafiği
    ax2.set_ylabel('Accuracy', color='tab:blue')
    ax2.plot(generations, accuracy_per_generation, color='tab:blue', label='Accuracy')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    plt.title("Accuracy and Loss Over Generations")
    plt.draw()  # Grafiği güncelle
    plt.pause(0.1)  # Küçük bir duraklama ekleyerek güncellemeleri işle

    return fig



def run_genetic_algorithm(images, populations, iterations, population_size, mutation_rate=0.1, output_dir="output"):
    """
    Genetik algoritmanın ana döngüsünü çalıştırır. Her nesil için popülasyonları günceller.
    """
    os.makedirs(output_dir, exist_ok=True)  # Çıktı klasörü oluşturuluyor
    loss_per_generation = []  # Jenerasyon başına loss değerlerini saklayacak liste
    accuracy_per_generation = []  # Jenerasyon başına accuracy değerlerini saklayacak liste
    binary_images = [convert_to_binary(img) for img in images]
    graph_path = os.path.join(output_dir, f"generation_graph.png")

    for generation in range(iterations):
        logger.info(f"\nStarting generation {generation + 1}/{iterations}")
        
        population_losses = []
        population_accuracys = []
        
        # Her bir popülasyonu işle
        for i, population in enumerate(populations):
            loss, accuracy = algorithm(binary_images, population, iterations)

            logger.info(f"Population {i}, Generation {generation + 1}.")
            logger.info(f"Loss: {loss}, Accuracy: {accuracy}")
            
            population_losses.append(loss)
            population_accuracys.append(accuracy)

        # Ortalama loss ve accuracy'yi kaydet
        avg_loss = sum(population_losses) / (len(populations) * 5)
        avg_accuracy = sum(population_accuracys) / (len(populations) * 5)

        logger.info(f"\nGeneration {generation + 1} Summary:")
        logger.info(f"Average Loss: {avg_loss}")
        logger.info(f"Average Accuracy: {avg_accuracy}")
        logger.info(f"Min Loss: {min(population_losses)}")
        logger.info(f"Max Accuracy: {max(population_accuracys)}")

        # Listeye jenerasyon başına loss ve accuracy ekle
        loss_per_generation.append(avg_loss)
        accuracy_per_generation.append(avg_accuracy)

        # En yüksek accuracy'yi veren popülasyonu bul
        best_population_index = population_accuracys.index(max(population_accuracys))
        best_population = populations[best_population_index]

        # 🔥 Dinamik olarak grafiği güncelle 🔥
        fig = plot_results(loss_per_generation, accuracy_per_generation)

        # 100 iterasyonda bir grafik kaydet, ve pattern'lerle resimler kaydet
        if (generation + 1) % 100 == 0:
            ite_output_dir = f"{output_dir}/{generation + 1}"
            os.makedirs(ite_output_dir, exist_ok=True)  # Çıktı klasörü oluşturuluyor
        
            fig.savefig(graph_path)
            logger.info(f"Saved graph for generation {generation + 1}: {graph_path}")

            save_pattern_images(binary_images, best_population, ite_output_dir)
            save_best_population_patterns(best_population, ite_output_dir)

        # Yeni popülasyonları oluştur
        populations = generate_new_populations(populations, population_accuracys, population_size, mutation_rate)
