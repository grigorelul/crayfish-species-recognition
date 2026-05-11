# Sistem de Clasificare a Speciilor de Rac (Crayfish Species Classification System)

## Instalare dependințe trebuie run ala requirements.txt

# Trebuie instalat tot din requirement cu comanda: 

# 1. Pregătești datele (o singură dată)
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented

# 2. Antrenezi (poate dura ore)
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp

# 3a. Evaluezi pe test set (cât de bun e modelul)
python predict.py --model best_model.pth --split test

# 3b. Folosești pe imagini noi din teren
python predict.py --model best_model.pth --input imagine_noua.jpg


## Pentru clasificator
# 1. Pregătești datele
python prepare_dataset_cls.py

# 2. Antrenezi clasificatorul
python train_cls.py --epochs 50 --batch_size 16 --amp

# 3. Evaluezi pe test
python predict_cls.py --split test

# 4. Pe imagini noi (producție)
python pipeline.py --input poza.jpg