import os
from PIL import Image
import random
import numpy as np

def convert_to_binary(img, threshold=128):
    """
    Verilen resmi binary (ikili) hale getirir.
    - Threshold: Piksel değeri eşik değeri. Bu değerin altındaki pikseller 0 (siyah), üstündekiler 1 (beyaz) olacak.
    """
    # Resmi gri tonlamaya çevir
    img_gray = img.convert("L")  # "L" modunda gri tonlama
    img_array = np.array(img_gray)

    # Eşik değerine göre binary hale getir
    binary_img = np.where(img_array > threshold, 1, 0)

    return binary_img

def hamming_distance(pattern1, pattern2):
    """
    Hamming mesafesini hesaplayan fonksiyon.
    Her iki pattern'in farklı olan bit sayısını döndürür.
    """
    return np.sum(pattern1 != pattern2)

def find_best_pattern_for_block(block, population):
    """
    Verilen bir 3x3'lük blok için popülasyondaki en benzer pattern'i bulur.
    """
    best_pattern = None
    min_distance = float('inf')
    
    for pattern in population:
        distance = hamming_distance(block, pattern)
        
        if distance < min_distance:
            min_distance = distance
            best_pattern = pattern
    
    return best_pattern

def calculate_loss(original_img, generated_img):
    """
    Loss hesaplama: Hamming mesafesi ya da pixel farkını kullanarak hesaplanabilir.
    """
    # İki resmi karşılaştırarak Hamming mesafesini hesapla
    loss = np.sum(original_img != generated_img)  # Pixel farkları toplamı
    return loss

def calculate_accuracy(original_img, generated_img):
    """
    Accuracy hesaplama: Orijinal resim ile üretilen resmin eşleşme oranı.
    """
    # Doğru eşleşen piksellerin oranını hesapla
    correct_pixels = np.sum(original_img == generated_img)
    total_pixels = original_img.size
    accuracy = correct_pixels / total_pixels
    return accuracy

def generate_random_pattern(pattern_size):
    return np.random.randint(2, size=(3, 3))

def generate_individual(pattern_count, pattern_size = (3, 3)):
    individual = []
    
    for _ in range(pattern_count):
        # Rastgele bir pattern üret
        pattern = generate_random_pattern(pattern_size)
        individual.append(pattern)
    
    return individual

def load_images_from_folder(folder_path):
    images = []
    # Klasördeki tüm dosyaların adlarını al
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Eğer dosya bir resimse (PNG, JPEG, vb.) resmi aç
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            img = Image.open(file_path)
            images.append((filename, img))  # Resimleri ve dosya adlarını tuple olarak saklıyoruz
    
    return images

def filter_images_by_size(images, target_size=(24, 24)):
    valid_images = []
    invalid_images = []
    
    for filename, img in images:
        # Boyut kontrolü
        if img.size == target_size:
            valid_images.append((filename, img))
        else:
            invalid_images.append(filename)
    
    # Boyutu 24x24 olmayanları bildir
    if invalid_images:
        print("The following images were removed (not 24x24):")
        for img_name in invalid_images:
            print(img_name)
    
    # Yalnızca 24x24 olanları döndür
    return [img for _, img in valid_images]
