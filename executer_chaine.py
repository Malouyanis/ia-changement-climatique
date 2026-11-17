"""Exécute les notebooks sans interface (équivalent de « Run All »), dans l'ordre demandé.

Exemples :
    python executer_chaine.py 01 02
    python executer_chaine.py 03 --essai unet_sans_poids
    python executer_chaine.py tout          # 01, 02, 03 et 04 pour chaque essai, puis 05 (essai retenu)

Le notebook est réécrit avec ses sorties. Pour un essai autre que l'essai principal du notebook
(ESSAI_PRINCIPAL), une copie exécutée est enregistrée dans executions/ et le notebook d'origine
n'est pas modifié.
"""
import re
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient

DOSSIER = Path(__file__).resolve().parent
NOTEBOOKS = {
    "01": "01_telechargement.ipynb",
    "02": "02_preparation.ipynb",
    "03": "03_unet.ipynb",
    "04": "04_indicateurs.ipynb",
    "05": "05_validation_croisee.ipynb",
}
ESSAIS = ["unet_base", "unet_sans_poids", "unet_2021_seul"]
ESSAI_RETENU = "unet_sans_poids"     # meilleur mIoU de validation (notebook 03)
# Essai dont les sorties sont gardées dans le notebook d'origine ; les autres vont dans executions/
ESSAI_PRINCIPAL = {"03": "unet_base", "04": ESSAI_RETENU, "05": ESSAI_RETENU}


def executer(code, essai=None):
    chemin = DOSSIER / NOTEBOOKS[code]
    principal = ESSAI_PRINCIPAL.get(code)
    essai = essai or principal
    nb = nbformat.read(chemin, as_version=4)
    if code in ESSAI_PRINCIPAL:
        for cellule in nb.cells:
            if cellule.cell_type == "code" and re.search(r'^NOM_ESSAI = ".*"', cellule.source, re.M):
                cellule.source = re.sub(r'^NOM_ESSAI = ".*"', f'NOM_ESSAI = "{essai}"', cellule.source, flags=re.M)
                break
        else:
            raise RuntimeError(f"Ligne NOM_ESSAI introuvable dans {chemin.name}")

    sortie = chemin
    if principal and essai != principal:
        (DOSSIER / "executions").mkdir(exist_ok=True)
        sortie = DOSSIER / "executions" / f"{chemin.stem}_{essai}.ipynb"

    print(f"=== {chemin.name} ({essai}) ...", flush=True)
    t0 = time.time()
    client = NotebookClient(nb, timeout=None, kernel_name="python3",
                            resources={"metadata": {"path": str(DOSSIER)}})
    try:
        client.execute()
    finally:
        nbformat.write(nb, sortie)   # on garde les sorties même en cas d'erreur
    print(f"    terminé en {(time.time() - t0) / 60:.1f} min -> {sortie.relative_to(DOSSIER)}", flush=True)


if __name__ == "__main__":
    args = sys.argv[1:]
    essai = None
    if "--essai" in args:
        k = args.index("--essai")
        essai = args[k + 1]
        del args[k:k + 2]
    if args == ["tout"]:
        executer("01")
        executer("02")
        for e in ESSAIS:
            executer("03", e)
        for e in ESSAIS:
            executer("04", e)
        executer("05")
    else:
        for code in args:
            executer(code, essai)
