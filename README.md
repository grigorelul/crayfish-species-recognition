# Identificarea automată a speciilor de raci din genul *Austropotamobius* prin tehnici de deep learning pe bază de imagini

Acest proiect propune o soluție de clasificare binară automată a două specii de raci de
apă dulce, morfologic foarte similare - *Austropotamobius bihariensis* și
*Austropotamobius torrentium* - pe baza unor fotografii capturate în habitatul natural.

Contribuția principală constă în adaptarea arhitecturii **AlexNet**, pre-antrenată pe
ImageNet, prin **transfer learning**, pentru un set de date propriu, de dimensiuni
reduse (318 imagini), colectat din teren. Straturile convoluționale ale rețelei
(`model.features`) sunt menținute complet îngheț­ate pe toată durata antrenării, pentru
a păstra reprezentările vizuale generale învățate pe ImageNet, în timp ce stratul final
de clasificare (`classifier[6]`) este reinițializat și antrenat pentru cele două clase
vizate. Modelul final obține o acuratețe de **95,92%** pe un set de testare independent
de 49 de imagini.

Fluxul complet de procesare cuprinde:

1. **Adnotarea manuală** a regiunii de interes (bounding box în jurul exemplarului),
   folosind instrumentul open-source [LabelMe](https://github.com/wkentaro/labelme);
2. **Extragerea automată** a regiunii adnotate și redimensionarea acesteia la
   rezoluția cerută de AlexNet (227×227 pixeli) - `CropRaci.py`;
3. **Antrenarea modelului** prin transfer learning, cu augmentare extinsă a datelor în
   timpul antrenării și împărțire stratificată a setului de date (70% / 15% / 15%) -
   `TrainClassifier.py`;
4. **Clasificarea** unei imagini noi, cu afișarea probabilităților pe fiecare clasă -
   `Predict.py`.

## Structura proiectului

```
proiect/
├── CropRaci.py          # Extragerea automata a regiunii de interes (ROI) din imaginile adnotate
├── TrainClassifier.py    # Antrenarea modelului AlexNet prin transfer learning
├── Predict.py            # Clasificarea unei imagini noi cu modelul antrenat
├── requirements.txt      # Dependințele Python necesare
└── README.md
```

## Cerințe

- Python ≥ 3.9
- [PyTorch](https://pytorch.org/) și `torchvision` (recomandat cu suport CUDA, pentru
  antrenare pe GPU - modelul a fost antrenat pe o placă video NVIDIA RTX 4060 Laptop)

Toate dependințele sunt listate în `requirements.txt`. Instalare rapidă:

```bash
pip install -r requirements.txt
```


## Setul de date

Setul de date propriu (318 imagini, colectate din teren și adnotate manual) **nu este
distribuit public** în acest repository, întrucât cele două specii vizate sunt specii
protejate, cu acces restricționat în habitatul natural. Structura de foldere așteptată
de scripturi, pentru cei care dispun de un set de date propriu adnotat similar, este
următoarea:

```
BazaDeDateRaci/                                # imagini brute + adnotari LabelMe (.json)
├── Austropotamobius bihariensis/
│   ├── imagine1.jpg
│   ├── imagine1.json
│   └── ...
└── Austropotamobius torrentium/
    ├── imagine1.jpg
    ├── imagine1.json
    └── ...
```

Adnotarea se face cu LabelMe (inclus în `requirements.txt`). Pentru a porni aplicația
din terminal, direct pe folderul cu imaginile unei specii:

```bash
python -m labelme "BazaDeDateRaci/Austropotamobius bihariensis/"
```

(similar pentru folderul celeilalte specii). Comanda deschide interfața grafică a
LabelMe, încărcând toate imaginile din folderul indicat, cu navigare între ele din
lista din panoul din dreapta.

Pentru fiecare imagine, în interfața LabelMe:

1. Click pe **Edit → Create Rectangle** (sau scurtătura `Ctrl+R`);
2. Click-and-drag pe imagine, pentru a desena un bounding box în jurul exemplarului;
3. La eliberarea click-ului, introduceți eticheta `rac` (sau selectați-o din lista de
   etichete deja folosite, dacă a fost introdusă anterior);
4. Salvați adnotarea cu `Ctrl+S`.

LabelMe salvează automat, lângă fiecare imagine, un fișier `.json` cu același nume
(de exemplu, pentru `imagine1.jpg` va rezulta `imagine1.json`), exact structura citită
de `CropRaci.py`. Dacă o imagine conține mai mulți exemplari, se pot desena mai multe
bounding box-uri cu eticheta `rac` pe aceeași imagine - `CropRaci.py` le va extrage pe
toate, separat.

## Instrucțiuni de utilizare

### 1. Extragerea regiunii de interes (ROI)

```bash
python CropRaci.py
```

Scriptul citește imaginile și adnotările din `BazaDeDateRaci/` și salvează exemplarele
decupate și redimensionate la 227×227 pixeli în `BazaDeDateRaciCropped/`, separate pe
clasă (`Austropotamobius_bihariensis` / `Austropotamobius_torrentium`).

Principalii parametri configurabili la începutul scriptului:

| Parametru | Descriere | Valoare implicită |
|---|---|---|
| `SRC_DIR` | Folderul cu imaginile brute și adnotările LabelMe | `BazaDeDateRaci` |
| `DST_DIR` | Folderul de destinație pentru imaginile decupate | `BazaDeDateRaciCropped` |
| `IMG_SIZE` | Rezoluția de redimensionare a crop-ului | `227` |
| `PADDING` | Marja (în pixeli) adăugată în jurul bounding box-ului adnotat | `20` |
| `PREVIEW` | Dacă `True`, nu salvează fișiere, ci afișează interactiv, pentru fiecare exemplar, o fereastră cu imaginea originală alături de crop-ul rezultat, pentru verificare vizuală înainte de procesarea efectivă | `False` |
| `FORCE` | Dacă `True`, suprascrie fișierele deja existente în `DST_DIR` | `False` |

### 2. Antrenarea modelului

```bash
python TrainClassifier.py
```

Scriptul citește imaginile decupate din `BazaDeDateRaciCropped/`, le împarte
stratificat în seturi de antrenare/validare/testare (70% / 15% / 15%, cu `seed=42`,
pentru ca împărțirea să fie reproductibilă la rulări ulterioare), aplică augmentarea
datelor în timpul antrenării și antrenează modelul AlexNet (cu straturile
convoluționale îngheț­ate) timp de 60 de epoci. Modelul cu cea mai bună acuratețe pe
setul de validare este salvat automat ca `alexnet_raci_best.pth`, iar la final este
raportată acuratețea pe setul de testare.

Principalii hiperparametri configurabili la începutul scriptului: `BATCH_SIZE`,
`LEARNING_RATE`, `NUM_EPOCHS`, `TRAIN_RATIO`, `VAL_RATIO`.

### 3. Clasificarea unei imagini noi

```bash
python Predict.py
```

La rulare fără argumente, scriptul deschide fereastra nativă a sistemului de operare
pentru selectarea unei imagini (de exemplu, File Explorer pe Windows), sau solicită
calea în terminal, dacă mediul nu poate afișa această fereastră. Este necesar ca
fișierul `alexnet_raci_best.pth`, generat la pasul anterior, să existe în același
folder cu scriptul.

Rezultatul afișat include clasa prezisă și, pentru fiecare dintre cele două specii,
procentul calculat de funcția softmax a rețelei (interpretabil ca nivel de încredere
al modelului în acea clasă), atât în terminal, cât și, dacă `matplotlib` este
disponibil, într-o fereastră grafică cu imaginea originală și un grafic cu aceste
procente.

Pentru a clasifica direct o imagine specifică, fără fereastra de selecție, se poate
seta calea în variabila `IMAGE_PATH` de la începutul scriptului.

## Reproducerea rezultatelor raportate

Întrucât setul de date propriu nu este distribuit în acest repository, reproducerea exactă a rezultatelor presupune deținerea unui set de imagini propriu, adnotat similar. Pentru a reproduce acuratețea de 95,92%
pe setul de testare, raportată în lucrare, este suficient să se ruleze, în ordine, cei
trei pași descriși mai sus (CropRaci.py → TrainClassifier.py → Predict.py),
folosind un astfel de set de date, fără modificarea hiperparametrilor impliciți.
Împărțirea antrenare/validare/testare este reproductibilă, fiind fixată prin
seed=42.