# Clasificarea speciilor de raci

Acest proiect folosește doar trei scripturi principale:

- [CropRaci.py](CropRaci.py) — generează crop-uri din imaginile etichetate cu LabelMe
- [Predict.py](Predict.py) — face predicția pentru o singură imagine folosind modelul salvat
- [TrainClassifier.py](TrainClassifier.py) — antrenează un clasificator AlexNet pe crop-uri

## Cum funcționează

1. Pregătește datele
   - Asigură-te că folderul `BazaDeDateRaci` conține imaginile și fișierele JSON cu etichete.
   - Rulează:
     ```bash
     python CropRaci.py
     ```
   - Rezultatul va fi salvat în folderul `BazaDeDateRaciCropped`.

2. Antrenează modelul
   - Rulează:
     ```bash
     python TrainClassifier.py
     ```
   - Scriptul va crea modelul final în `alexnet_raci_best.pth`.

3. Fă predicții
   - Rulează:
     ```bash
     python Predict.py
     ```
   - Va cere o imagine și va afișa clasa prezisă.

## Cerințe

Instalează dependențele cu:

```bash
pip install -r requirements.txt
```

## Notă

Acest README este bazat doar pe fluxul oferit de cele trei fișiere menționate mai sus.