import os

import cv2
import matplotlib.pyplot as plt
import pandas as pd


def analyze_dataset_with_target_resolution(images_dir, labels_dir, target_w=640, target_h=640):
    data_list = []
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(image_extensions)]

    if not image_files:
        print(f"Nie znaleziono zdjęć w katalogu: {images_dir}")
        return None

    for img_file in image_files:
        base_name = os.path.splitext(img_file)[0]
        label_file = f"{base_name}.txt"
        label_path = os.path.join(labels_dir, label_file)
        img_path = os.path.join(images_dir, img_file)

        img = cv2.imread(img_path)
        if img is None:
            continue
        img_h, img_w, _ = img.shape

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center_rel = float(parts[1])
                    y_center_rel = float(parts[2])
                    width_rel = float(parts[3])
                    height_rel = float(parts[4])

                    # Wariant A: Wymiary oryginalne (surowe dane przed skalowaniem)
                    box_w_orig = width_rel * img_w
                    box_h_orig = height_rel * img_h
                    area_orig = box_w_orig * box_h_orig

                    # Wariant B: Wymiary po skalowaniu do rozdzielczości sieci (640x640)
                    box_w_scaled = width_rel * target_w
                    box_h_scaled = height_rel * target_h
                    area_scaled = box_w_scaled * box_h_scaled
                    img_area_scaled = target_w * target_h
                    area_perc_scaled = (area_scaled / img_area_scaled) * 100

                    aspect_ratio = box_w_scaled * box_h_scaled if box_h_scaled > 0 else 0

                    data_list.append(
                        {
                            "image_name": img_file,
                            "orig_width": img_w,
                            "orig_height": img_h,
                            "box_w_orig": box_w_orig,
                            "box_h_orig": box_h_orig,
                            "area_orig": area_orig,
                            "box_w_scaled": box_w_scaled,
                            "box_h_scaled": box_h_scaled,
                            "area_scaled": area_scaled,
                            "area_perc_scaled": area_perc_scaled,
                            "aspect_ratio": aspect_ratio,
                        }
                    )

    return pd.DataFrame(data_list)


def generate_comparative_report(df, output_dir="stats_output"):
    if df is None or df.empty:
        print("Brak danych do analizy.")
        return

    os.makedirs(output_dir, exist_ok=True)

    total_objects = len(df)
    print("--- PORÓWNANIE GEOMETRII OBIEKTÓW ---")
    print(f"Łączna liczba obiektów: {total_objects}\n")

    print("Średnie wymiary w pikselach przed i po skalowaniu do 640x640:")
    print(df[["box_w_orig", "box_h_orig", "box_w_scaled", "box_h_scaled"]].mean())
    print("\n")

    # 1. Klasyfikacja wielkości obiektów przed skalowaniem (oryginalna rozdzielczość)
    small_orig = df[df["area_orig"] < 1024]
    medium_orig = df[(df["area_orig"] >= 1024) & (df["area_orig"] <= 9216)]
    large_orig = df[df["area_orig"] > 9216]

    print("Rozkład wielkości obiektów PRZED SKALOWANIEM (Oryginalna rozdzielczość):")
    print(f"Obiekty małe (<32x32 px): {len(small_orig)} ({len(small_orig) / total_objects * 100:.2f}%)")
    print(f"Obiekty średnie (32x32 - 96x96 px): {len(medium_orig)} ({len(medium_orig) / total_objects * 100:.2f}%)")
    print(f"Obiekty duże (>96x96 px): {len(large_orig)} ({len(large_orig) / total_objects * 100:.2f}%)")
    print("\n")

    # 2. Klasyfikacja wielkości obiektów po skalowaniu (docelowa rozdzielczość sieci)
    small_scaled = df[df["area_scaled"] < 1024]
    medium_scaled = df[(df["area_scaled"] >= 1024) & (df["area_scaled"] <= 9216)]
    large_scaled = df[df["area_scaled"] > 9216]

    print("Rozkład wielkości obiektów PO SKALOWANIU (Docelowa rozdzielczość 640x640):")
    print(f"Obiekty małe (<32x32 px): {len(small_scaled)} ({len(small_scaled) / total_objects * 100:.2f}%)")
    print(f"Obiekty średnie (32x32 - 96x96 px): {len(medium_scaled)} ({len(medium_scaled) / total_objects * 100:.2f}%)")
    print(f"Obiekty duże (>96x96 px): {len(large_scaled)} ({len(large_scaled) / total_objects * 100:.2f}%)")
    print("\n")

    # Generowanie wykresu porównawczego powierzchni
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(df["area_orig"], bins=30, color="gray", alpha=0.6, label="Przed skalowaniem (Oryginalna)")
    plt.hist(df["area_scaled"], bins=30, color="blue", alpha=0.6, label="Po skalowaniu (640x640)")
    plt.title("Porównanie bezwzględnej powierzchni ramek (px^2)")
    plt.xlabel("Powierzchnia obiektu")
    plt.ylabel("Liczba obiektów")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(df["aspect_ratio"], bins=30, color="salmon", edgecolor="black")
    plt.title("Rozkład stosunku proporcji (Szerokość / Wysokość)")
    plt.xlabel("Współczynnik proporcji")
    plt.ylabel("Liczba obiektów")

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "porownanie_rozdzielczosci.png")
    plt.savefig(plot_path)
    plt.close()

    csv_path = os.path.join(output_dir, "statystyki_skalowania.csv")
    df.to_csv(csv_path, index=False)

    print(f"Wykresy zostały zapisane w: {plot_path}")
    print(f"Dane szczegółowe zostały zapisane w: {csv_path}")


IMAGES_DIRECTORY = r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\images\train"
LABELS_DIRECTORY = r"D:\praca_magisterska\project\Licence_plate_detection\datasets\licence_plate\labels\train"

if __name__ == "__main__":
    df_stats = analyze_dataset_with_target_resolution(IMAGES_DIRECTORY, LABELS_DIRECTORY)
    generate_comparative_report(df_stats)
