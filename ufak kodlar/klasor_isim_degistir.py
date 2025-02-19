import os
 
def rename_folders(base_dir):
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
       
        if os.path.isdir(folder_path):
            for subfolder_name in os.listdir(folder_path):
                subfolder_path = os.path.join(folder_path, subfolder_name)
               
                if os.path.isdir(subfolder_path):
                    new_name = f"{folder_name}-{subfolder_name}"
                    new_subfolder_path = os.path.join(folder_path, new_name)
                   
                    os.rename(subfolder_path, new_subfolder_path)
                    print(f"{subfolder_name} ismi değiştirildi: {new_name}")
 
base_directory = r"D:\Projeler\UniversiteProjeler\PlantMasterAI\PlantMaster_Veriler\BIRLESTIRILMIS\VeriSetOrj224\VeriSetOrj224"  
rename_folders(base_directory)
 
 