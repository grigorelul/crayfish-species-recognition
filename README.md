# Clasificarea speciilor de raci

Acest proiect folosește doar trei scripturi principale:

- [CropRaci.py](CropRaci.py) — generează crop-uri din imaginile etichetate cu LabelMe
- [Predict.py](Predict.py) — face predicția pentru o singură imagine folosind modelul salvat
- [TrainClassifier.py](TrainClassifier.py) — antrenează un clasificator AlexNet pe crop-uri

## Cum funcționează

1. Etichetează imaginile (daca nu ai deja fișierele JSON)
   - Pune imaginile în `BazaDeDateRaci`, separate pe subfoldere cu numele speciilor (vezi `CLASS_NAMES` din `CropRaci.py`).
   - Rulează LabelMe:
     ```bash
     labelme BazaDeDateRaci
     ```
   - Pentru fiecare imagine desenează un dreptunghi în jurul racului și pune-i eticheta `rac`, apoi salvează (Ctrl+S). Se va crea un `.json` cu același nume lângă imagine.

2. Generează crop-urile
   - Rulează:
     ```bash
     python CropRaci.py
     ```
   - Rezultatul va fi salvat în folderul `BazaDeDateRaciCropped`.

3. Antrenează modelul
   - Rulează:
     ```bash
     python TrainClassifier.py
     ```
   - Scriptul va crea modelul final în `alexnet_raci_best.pth`.

4. Fă predicții
   - Rulează:
     ```bash
     python Predict.py
     ```
   - Va cere o imagine și va afișa clasa prezisă.

## Cerințe

Instalează dependințele cu:

```bash
pip install -r requirements.txt
```