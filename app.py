"""Comparaison de visages avec DeepFace, pilotée par le contenu du dossier /data.

Mode verify : exactement 2 images directement dans data/
Mode find   : une image nommée "temoin.*" dans data/ + un sous-dossier data/base/
"""
import math
import os
import sys
from pathlib import Path

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA = Path(os.environ.get("DATA_DIR", "/data"))
BASE_DIR = DATA / "base"
MODEL = os.environ.get("MODEL", "VGG-Face")
DETECTOR = os.environ.get("DETECTOR", "opencv")


def images_in(folder: Path):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXT)


def pct(distance: float, threshold: float) -> float:
    """Distance -> score 0-100 %. distance == seuil -> 50 %. Pas une probabilité."""
    return 100.0 / (1.0 + math.exp(6.0 * (distance / threshold - 1.0)))


def run_verify(a: Path, b: Path):
    from deepface import DeepFace
    print(f"[verify] {a.name}  <->  {b.name}  (modèle={MODEL}, détecteur={DETECTOR})\n")
    r = DeepFace.verify(img1_path=str(a), img2_path=str(b),
                        model_name=MODEL, detector_backend=DETECTOR)
    verdict = "MÊME personne" if r["verified"] else "personnes DIFFÉRENTES"
    print(f"Correspondance : {pct(r['distance'], r['threshold']):.1f} %")
    print(f"Verdict        : {verdict}")
    print(f"Distance       : {r['distance']:.4f} (seuil {r['threshold']:.4f})")


def run_find(ref: Path):
    from deepface import DeepFace
    n = len(images_in(BASE_DIR))
    print(f"[find] témoin={ref.name}  base=base/ ({n} images)  "
          f"(modèle={MODEL}, détecteur={DETECTOR})")
    print("La première fois, l'indexation de la base peut prendre du temps.\n")
    results = DeepFace.find(img_path=str(ref), db_path=str(BASE_DIR),
                            model_name=MODEL, detector_backend=DETECTOR,
                            silent=True)
    for i, df in enumerate(results, 1):
        if len(results) > 1:
            print(f"--- Visage n°{i} détecté sur la photo témoin ---")
        if df.empty:
            print("Aucune correspondance trouvée.")
            continue
        for _, row in df.sort_values("distance").iterrows():
            name = os.path.relpath(row["identity"], BASE_DIR)
            print(f"{pct(row['distance'], row['threshold']):5.1f} %   {name}")


def main():
    if not DATA.is_dir():
        sys.exit(f"Dossier {DATA} introuvable (volume non monté ?).")

    ref = next((p for p in images_in(DATA) if p.stem.lower() == "temoin"), None)
    top_images = images_in(DATA)

    try:
        if ref and BASE_DIR.is_dir():
            if not images_in(BASE_DIR) and not any(BASE_DIR.rglob("*")):
                sys.exit("data/base/ est vide.")
            run_find(ref)
        elif len(top_images) == 2:
            run_verify(*top_images)
        else:
            sys.exit(
                "Contenu de data/ non reconnu. Deux possibilités :\n"
                "  - verify : mets exactement 2 images dans data/\n"
                "  - find   : mets 'temoin.jpg' dans data/ + un dossier data/base/ rempli d'images\n"
                f"(trouvé : {len(top_images)} image(s) à la racine, "
                f"base/ {'présent' if BASE_DIR.is_dir() else 'absent'})"
            )
    except ValueError as e:
        sys.exit(f"Erreur : {e}\nAstuce : photo plus nette/de face, ou essaie DETECTOR=retinaface.")


if __name__ == "__main__":
    main()
