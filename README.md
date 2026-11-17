# Impact des mégafeux de 2022 sur la forêt de Gironde

Projet du cours **Intelligence artificielle et changement climatique** (OUAP, ESIEE Paris, 2026-2027), projet 2 : analyse de l'occupation du sol par apprentissage profond.

Un réseau **UNet** est entraîné sur des images **Sentinel-2** pour cartographier l'occupation du sol en 2021, 2022 et 2025 sur le massif des Landes de Gascogne (zone d'environ 77 × 46 km autour des feux de La Teste-de-Buch et de Landiras). Les cartes servent à mesurer la perte de forêt due aux feux de l'été 2022, son devenir trois ans plus tard et la fragmentation du massif.

## Données utilisées

| Source | Rôle |
|---|---|
| Sentinel-2 L2A (Microsoft Planetary Computer) | Images d'entrée : composites médians de fin d'été 2021, 2022 et 2025, 6 bandes (B02, B03, B04, B08, B11, B12) |
| ESA WorldCover 2021 v200 | Étiquettes d'occupation du sol, regroupées en 6 classes |
| Périmètres EFFIS 2022 (Copernicus) | Étiquettes « brûlé » sur l'image 2022 |
| Hansen Global Forest Change v1.12 | Exclusion des coupes rases de l'apprentissage, interprétation des pertes |

Toutes les données sont téléchargées automatiquement par le notebook 01. Elles ne sont **pas** incluses dans le dépôt (plusieurs Go).

## Organisation du dépôt

| Fichier / dossier | Contenu |
|---|---|
| `01_telechargement.ipynb` | Téléchargement des images Sentinel-2, de WorldCover, des périmètres EFFIS et de Hansen ; composites médians et reprojection en UTM 30N |
| `02_preparation.ipynb` | Regroupement des classes, construction des étiquettes 2021 et 2022, découpage en patchs 256 × 256 et répartition en 5 plis spatiaux |
| `03_unet.ipynb` | Entraînement du UNet pour un essai donné (`NOM_ESSAI`) |
| `04_indicateurs.ipynb` | Cartes d'occupation du sol des trois années, validation par le dNBR et indicateurs (perte de forêt, devenir en 2025, fragmentation) |
| `05_validation_croisee.ipynb` | Validation croisée spatiale à 5 plis de l'essai retenu |
| `unet.py` | Architecture du UNet, partagée par les notebooks 03 à 05 |
| `executer_chaine.py` | Exécute les notebooks sans interface, dans l'ordre |
| `executions/` | Copies exécutées des notebooks 03 et 04 pour les autres essais |
| `modeles/` | Historiques d'entraînement et résultats (métriques) de chaque essai |
| `figures/` | Toutes les figures produites par les notebooks |
| `figures_rapport/` | Les 9 figures du rapport, numérotées |

## Les trois essais

| Essai | Poids de classes | Images d'entraînement |
|---|---|---|
| `unet_base` | oui (1/√fréquence) | 2021 + 2022 |
| `unet_sans_poids` | non | 2021 + 2022 |
| `unet_2021_seul` | oui | 2021 |

L'essai retenu est **`unet_sans_poids`**, qui a la meilleure mIoU de validation.

## Installation

Python 3.11. Dans le dossier du projet :

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

L'entraînement utilise automatiquement le GPU disponible : Apple (MPS), NVIDIA (CUDA), ou à défaut le processeur.

## Reproduire les résultats

Ouvrir les notebooks dans l'ordre (01 → 05) et les exécuter entièrement, ou tout lancer depuis le terminal :

```bash
python executer_chaine.py tout
```

Cette commande exécute 01 et 02, puis 03 et 04 pour chacun des trois essais, puis 05 pour l'essai retenu. On peut aussi lancer une seule étape :

```bash
python executer_chaine.py 01 02
python executer_chaine.py 03 --essai unet_sans_poids
```

Les poids des modèles entraînés (`modeles/*.pt`) ne sont pas versionnés : ils sont recréés par le notebook 03. Un entraînement prend environ 20 minutes sur le GPU d'un Mac M5 Max.

## Principaux résultats

- Validation croisée spatiale à 5 plis : mIoU de 0,79 ± 0,02, et IoU de 0,94 ± 0,01 pour les surfaces brûlées en 2022.
- Environ 31 000 ha de forêt perdus entre 2021 et 2022, dont près des trois quarts à cause des feux.
- En 2025, 84 % de la forêt brûlée est couverte de végétation basse et 7 % seulement est redevenue forêt.
- La plus grande tache forestière passe de 63 % à 34 % de la surface boisée.

## Auteur

Yanis Malou — ESIEE Paris, E4 Data Science et Intelligence Artificielle.
