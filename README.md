# Sistem de Clasificare a Speciilor de Rac (Crayfish Species Classification System)

# Instalare dependințe trebuie run la requirements.txt


# 1. Pregătești datele 
python prepare_dataset.py --data_root BazaDeDateRaci_Augmented

# 2. Antrenezi 
python train.py --data_dir dataset --epochs 100 --batch_size 4 --amp

# 3a. Evaluez pe test set 
python predict.py --model best_model.pth --split test

# 3b. Folosesc pe imagini noi 
python predict.py --model best_model.pth --input imagine_noua.jpg


## Pentru clasificator
# 1. Pregătești datele
python prepare_dataset_cls.py

# 2. Antrenezi clasificatorul
python train_cls.py --epochs 50 --batch_size 16 --amp

# 3. Evaluezi pe test
python predict_cls.py --split test

# 4. Folosesti pe imagini noi
python pipeline.py --input poza.jpg