"""Descarga y empaquetado del dataset Colorectal Histology (Zenodo 53169).

Genera arrays compactos en uint8 para optimizar el consumo de RAM (Mac M2, 8 GB).
- images_150x150.npy: uint8 (5000, 150, 150, 3) ~337.5 MB
- labels_multiclass.npy: int64 (5000,) [0: TUMOR .. 7: EMPTY]
- labels_binary.npy: int64 (5000,) [1: TUMOR, 0: NO TUMOR] (convencion medica estandar)
- labels_binary_kather.npy: int64 (5000,) [0: TUMOR, 1: NO TUMOR] (mapeo del notebook del profesor)
- filenames.npy: str (5000,)
"""

import hashlib
import io
import os
import shutil
import urllib.request
import zipfile
from pathlib import Path
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Constantes y rutas
# ---------------------------------------------------------------------------
ZENODO_URL = "https://zenodo.org/api/records/53169/files/Kather_texture_2016_image_tiles_5000.zip/content"
EXPECTED_MD5 = "0ddbebfc56344752028fda72602aaade"
EXPECTED_TOTAL_IMAGES = 5000

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
ZIP_PATH = RAW_DIR / "Kather_texture_2016_image_tiles_5000.zip"

# Mapeo oficial de clases segun enunciado y paper Kather et al. (2016):
# 0: TUMOR, 1: STROMA, 2: COMPLEX, 3: LYMPHO, 4: DEBRIS, 5: MUCOSA, 6: ADIPOSE, 7: EMPTY
CLASS_FOLDER_MAPPING = {
    "01_TUMOR": 0,
    "02_STROMA": 1,
    "03_COMPLEX": 2,
    "04_LYMPHO": 3,
    "05_DEBRIS": 4,
    "06_MUCOSA": 5,
    "07_ADIPOSE": 6,
    "08_EMPTY": 7,
}


def compute_md5(file_path: Path, chunk_size: int = 8192 * 1024) -> str:
    """Calcula el hash MD5 de un fichero por bloques para no saturar la RAM."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_dataset() -> None:
    """Descarga el fichero ZIP oficial de Zenodo si no existe o si el MD5 no coincide."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if ZIP_PATH.exists():
        print(f"Fichero encontrado en {ZIP_PATH}. Verificando MD5...")
        file_md5 = compute_md5(ZIP_PATH)
        if file_md5 == EXPECTED_MD5:
            print("MD5 verificado correctamente.")
            return
        print(f"MD5 erroneo ({file_md5} != {EXPECTED_MD5}). Re-descargando...")

    print(f"Descargando dataset desde {ZENODO_URL}...")
    import subprocess
    cmd = ["curl", "-L", "--fail", "-o", str(ZIP_PATH), ZENODO_URL]
    res = subprocess.run(cmd, capture_output=False)
    if res.returncode != 0:
        raise RuntimeError(f"Fallo al descargar con curl (codigo {res.returncode})")

    print("Descarga completada. Verificando integridad...")
    file_md5 = compute_md5(ZIP_PATH)
    if file_md5 != EXPECTED_MD5:
        raise ValueError(f"Integridad fallida: MD5 obtenido {file_md5} != esperado {EXPECTED_MD5}")
    print("MD5 verificado con exito.")


def process_and_pack() -> None:
    """Descomprime y empaqueta las 5.000 teselas directamente a arrays .npy en uint8."""
    print("Leyendo teselas directamente desde el archivo ZIP...")
    images_list = []
    labels_multiclass = []
    labels_binary = []
    filenames_list = []

    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        # Filtrar solo imagenes tif / jpg / png ignorando metadatos de macOS (__MACOSX)
        file_list = [f for f in z.namelist() if not f.startswith("__MACOSX") and f.lower().endswith((".tif", ".tiff", ".jpg", ".jpeg", ".png"))]
        
        # Ordenar alfabeticamente para garantizar determinismo estricto
        file_list.sort()

        if len(file_list) != EXPECTED_TOTAL_IMAGES:
            raise ValueError(f"Se esperaban {EXPECTED_TOTAL_IMAGES} imagenes pero se encontraron {len(file_list)}")

        print(f"Procesando {len(file_list)} teselas (150x150x3) en formato uint8...")
        for idx, filename in enumerate(file_list):
            parts = filename.strip("/").split("/")
            # Formato esperado: 'Kather_texture_2016_image_tiles_5000/01_TUMOR/1A11_CRC-Prim-HE-07_001.tif'
            folder_name = parts[-2]
            base_name = parts[-1]

            if folder_name not in CLASS_FOLDER_MAPPING:
                raise ValueError(f"Carpeta de clase desconocida: {folder_name} en {filename}")

            c_idx = CLASS_FOLDER_MAPPING[folder_name]

            # Leer imagen con PIL directamente del stream
            img_bytes = z.read(filename)
            with Image.open(io.BytesIO(img_bytes)) as img:
                img_rgb = img.convert("RGB")
                img_arr = np.array(img_rgb, dtype=np.uint8)

            if img_arr.shape != (150, 150, 3):
                raise ValueError(f"Dimension inesperada en {filename}: {img_arr.shape}")

            images_list.append(img_arr)
            labels_multiclass.append(c_idx)
            # En binario clinico: 1 = TUMOR, 0 = NO TUMOR
            labels_binary.append(1 if c_idx == 0 else 0)
            filenames_list.append(f"{folder_name}/{base_name}")

            if (idx + 1) % 1000 == 0:
                print(f"  [{idx + 1}/{len(file_list)}] teselas cargadas...")

    # Convertir a ndarrays
    images_np = np.stack(images_list, axis=0) # shape (5000, 150, 150, 3), uint8
    labels_multi_np = np.array(labels_multiclass, dtype=np.int64)
    labels_bin_np = np.array(labels_binary, dtype=np.int64)
    # Mapeo del profesor: TUMOR = 0, NO TUMOR = 1
    labels_bin_kather_np = np.where(labels_multi_np == 0, 0, 1).astype(np.int64)
    filenames_np = np.array(filenames_list, dtype=object)

    # Guardar en disco
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    images_file = DATA_DIR / "images_150x150.npy"
    labels_multi_file = DATA_DIR / "labels_multiclass.npy"
    labels_bin_file = DATA_DIR / "labels_binary.npy"
    labels_bin_kather_file = DATA_DIR / "labels_binary_kather.npy"
    filenames_file = DATA_DIR / "filenames.npy"

    print("Guardando arrays .npy en data/...")
    np.save(images_file, images_np)
    np.save(labels_multi_file, labels_multi_np)
    np.save(labels_bin_file, labels_bin_np)
    np.save(labels_bin_kather_file, labels_bin_kather_np)
    np.save(filenames_file, filenames_np)

    # Tamanios en MB
    mb_images = images_file.stat().st_size / (1024 * 1024)
    print("\n" + "=" * 50)
    print("RESUMEN DE EMPAQUETADO:")
    print(f"Imagenes: {images_np.shape}, dtype={images_np.dtype}, tamanio archivo={mb_images:.2f} MB")
    print(f"Etiquetas multiclase: {labels_multi_np.shape}, clases 0..7 con 625 muestras c/u")
    print(f"Etiquetas binarias estandar (1=TUMOR): {np.bincount(labels_bin_np)} [0: 4375, 1: 625]")
    print(f"Etiquetas binarias profesor (0=TUMOR): {np.bincount(labels_bin_kather_np)} [0: 625, 1: 4375]")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    download_dataset()
    process_and_pack()
