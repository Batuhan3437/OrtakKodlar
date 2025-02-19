import os
import random
 
def reduce_images(folder_path, target_count=4000):
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
 
    if len(all_files) > target_count:
        files_to_delete = random.sample(all_files, len(all_files) - target_count)  
        for file in files_to_delete:
            os.remove(file)
        print(f"{len(files_to_delete)} dosya silindi. Kalan dosya sayısı: {target_count}")
    else:
        print(f"Klasörde zaten {len(all_files)} dosya var, işlem yapılmadı.")
 
folder_to_reduce = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\PlantMaster_Veriler\BIRLESTIRILMIS\VeriSetOrj224\VeriSetOrj224\ÜzümHastalıkları_GrapeDisease\black_rot"
 
reduce_images(folder_to_reduce)
 
 