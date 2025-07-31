import os
import csv

# === Ganti sesuai folder kamu ===
folder = r"Indonesian License Plate Recognition Dataset\labels\test"
output_csv = os.path.join(folder, "ground_truth.csv")

# === Mapping class_id ke karakter ===
label_map = {
    0: '0',
    1: '1',
    2: '2',
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'A',
    11: 'B',
    12: 'C',
    13: 'D',
    14: 'E',
    15: 'F',
    16: 'G',
    17: 'H',
    18: 'I',
    19: 'J',
    20: 'K',
    21: 'L',
    22: 'M',
    23: 'N',
    24: 'O',
    25: 'P',
    26: 'Q',
    27: 'R',
    28: 'S',
    29: 'T',
    30: 'U',
    31: 'V',
    32: 'W',
    33: 'X',
    34: 'Y',
    35: 'Z'
}

data = []

for file in os.listdir(folder):
    if file.endswith(".txt"):
        txt_path = os.path.join(folder, file)
        jpg_name = file.replace(".txt", ".jpg")

        with open(txt_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        boxes = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 2:
                class_id = int(parts[0])
                x_center = float(parts[1])
                char = label_map.get(class_id, '?')
                boxes.append((x_center, char))

        # Urutkan berdasarkan x_center (dari kiri ke kanan)
        boxes.sort(key=lambda x: x[0])
        plate_number = ''.join([char for _, char in boxes])
        data.append([jpg_name, plate_number])

# Simpan ke CSV
with open(output_csv, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image", "ground_truth"])
    writer.writerows(data)

print(f"✅ ground_truth.csv berhasil dibuat di: {output_csv}")
