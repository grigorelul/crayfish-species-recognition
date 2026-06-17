# Sistem de Clasificare a Speciilor de Rac (Crayfish Species Classification System)

# Instalare dependințe trebuie run la requirements.txt

# 1. Dataset segmentare
python prepare_dataset.py --data_root BazaDeDateRaci

# 2. Antrenare segmentare
python train.py --n_classes 5 --amp --patience 20

# 3. Evaluare segmentare
python predict.py --split test

# 4. Dataset clasificare (folosește noul best_model.pth)
python prepare_dataset_cls.py --data_root BazaDeDateRaci

# 5. Antrenare clasificare
python train_cls.py --unfreeze_epoch 15

# 6. Evaluare clasificare
python predict_cls.py --split test