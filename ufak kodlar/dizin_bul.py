import os

def count_files(directory):
    return sum(1 for entry in os.scandir(directory) if entry.is_file())

def display_tree(directory, indent="", output_file=None):
    entries = sorted(os.scandir(directory), key=lambda e: (not e.is_dir(), e.name.lower()))
    subdirs = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    
    for index, entry in enumerate(subdirs):
        is_last = index == len(subdirs) - 1 and not files
        prefix = "└── " if is_last else "├── "
        line = indent + prefix + entry.name
        print(line)
        if output_file:
            output_file.write(line + "\n")
        new_indent = indent + ("    " if is_last else "│   ")
        display_tree(entry.path, new_indent, output_file)
    
    if not subdirs:
        file_count = count_files(directory)
        line = indent + f"└── ({file_count} dosya) - {directory}"
        print(line)
        if output_file:
            output_file.write(line + "\n")

if __name__ == "__main__":
    root_dir = input("Ana dizini girin: ").strip()
    if os.path.isdir(root_dir):
        output_path = os.path.join(root_dir, "dizin_raporu.txt")
        with open(output_path, "w", encoding="utf-8") as file:
            file.write(root_dir + "\n")
            print(root_dir)
            display_tree(root_dir, output_file=file)
        print(f"Rapor kaydedildi: {output_path}")
    else:
        print("Geçerli bir dizin giriniz!")