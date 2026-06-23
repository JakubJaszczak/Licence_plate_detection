import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def analyze_yolo_dataset(images_dir, labels_dir):
    data_list = []

    # Pobranie listy plików graficznych
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(path_error_message := f"Nie znaleziono zdjęć w katalogu: {images_dir}")
        return None

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_file)
        img_path = os.path.join(images_dir, img_file)

        # Wczytanie wymiarów obrazu
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape

        # Sprawdzenie czy istnieje plik adnotacji
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    # Wartości w formacie YOLO są znormalizowane (0 do 1)
                    x_center_rel = float(parts[1])
                    y_center_rel = float(parts[2])
                    width_rel = float(parts[3])
                    height_rel = float(parts[4])

                    # Konwersja na wartości bezwzględne (w pikselach)
                    box_width_px = width_rel * img_w
                    box_height_px = height_rel * img_h

                    # Obliczenie procentu powierzchni obrazu zajmowanego przez obiekt
                    obj_area_px = box_width_px * box_height_px
                    img_area_px = img_w * img_h
                    area_percentage = (obj_area_px / img_area_px) * 100

                    # Stosunek proporcji (szerokość / wysokość)
                    aspect_ratio = box_width_px / box_height_px if box_height_px > 0 else 0

                    data_list.append(
                        {
                            "image_name": img_file,
                            "image_width": img_w,
                            "image_height": img_h,
                            "class_id": class_id,
                            "box_width_px": box_width_px,
                            "box_height_px": box_height_px,
                            "area_percentage": area_percentage,
                            "aspect_ratio": aspect_ratio,
                        }
                    )

    df = pd.DataFrame(data_list)
    return df


def generate_report(df, output_dir="stats_output"):
    if df is None or df.empty:
        print("Brak danych do analizy.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("--- PODSTAWOWE STATYSTYKI ZBIORU ---")
    print(f"Łączna liczba wykrytych obiektów (tablic): {len(df)}")
    print(f"Liczba unikalnych obrazów z adnotacjami: {df['image_name'].nunique()}\n")

    # 1. Analiza rozdzielczości obrazów
    print("1. Rozdzielczości obrazów (najczęstsze):")
    resolution_counts = df.groupby(["image_width", "image_height"]).size().reset_index(name="count")
    print(resolution_counts.sort_values(by="count", ascending=False).to_string(index=False))
    print("\n")

    # 2. Analiza wymiarów ramek otaczających
    print("2. Wymiary obiektów (w pikselach):")
    print(df[["box_width_px", "box_height_px"]].describe().loc[["mean", "min", "max"]])
    print("\n")

    # 3. Analiza zajętości powierzchni i proporcji
    print("3. Procentowa powierzchnia i współczynnik proporcji (Aspect Ratio):")
    print(df[["area_percentage", "aspect_ratio"]].describe().loc[["mean", "min", "max"]])

    # Generowanie i zapisywanie wykresów
    plt.figure(figsize=(12, 5))

    # Wykres 1: Rozkład procentowy powierzchni obiektów
    plt.subplot(1, 2, 1)
    plt.hist(df["area_percentage"], bins=30, color="skyblue", edgecolor="black")
    plt.title("Rozkład procentowy powierzchni tablic")
    plt.xlabel("Procent powierzchni obrazu (%)")
    plt.ylabel("Liczba obiektów")

    # Wykres 2: Rozkład stosunku proporcji (Aspect Ratio)
    plt.subplot(1, 2, 2)
    plt.hist(df["aspect_ratio"], bins=30, color="salmon", edgecolor="black")
    plt.title("Rozkład współczynnika proporcji")
    plt.xlabel("Współczynnik proporcji")
    plt.ylabel("Liczba obiektów")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "statystyki_geometryczne.png")
    plt.savefig(plot_path)
    plt.close()

    # Zapis surowych danych statystycznych do pliku CSV
    csv_path = os.path.join(output_dir, "surowe_statystyki.csv")
    df.to_csv(csv_path, index=False)

    print(f"\nWykresy zostały zapisane w: {plot_path}")
    print(f"Dane szczegółowe zostały zapisane w: {csv_path}")


# --- KONFIGURACJA ŚCIEŻEK ---
# Podaj ścieżki do folderów wybranego podzbioru (np. train lub test)
IMAGES_DIRECTORY = r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\train"
LABELS_DIRECTORY = r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\labels\train"

if __name__ == "__main__":
    # Uruchomienie analizy
    raw_data = analyze_yolo_dataset(IMAGES_DIRECTORY, LABELS_DIRECTORY)
    generate_report(raw_data)
