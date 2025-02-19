import os
import shutil
 
old_base_directory = input("Base directory yolunu girin: ").strip().strip('"').strip("'")
 
new_base_directory = os.path.join(os.path.dirname(old_base_directory), 'Veri Seti SınıflandırılmışV2')
 
if not os.path.exists(new_base_directory):
    os.makedirs(new_base_directory)
 
for folder_name in os.listdir(old_base_directory):
    folder_path = os.path.join(old_base_directory, folder_name)
   
    if os.path.isdir(folder_path):
        for subfolder_name in os.listdir(folder_path):
            subfolder_path = os.path.join(folder_path, subfolder_name)
           
            if os.path.isdir(subfolder_path):
                new_subfolder_path = os.path.join(new_base_directory, subfolder_name)
                shutil.move(subfolder_path, new_subfolder_path)
                print(f"{subfolder_name} başarıyla taşındı.")
           
            elif os.path.isfile(subfolder_path):
                new_subfolder_path = os.path.join(new_base_directory, subfolder_name)
                shutil.move(subfolder_path, new_subfolder_path)
                print(f"{subfolder_name} başarıyla taşındı.")
 
 