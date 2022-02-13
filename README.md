Ortam:

- Python 3.7.7
- Anaconda 
- PyTorch 1.6
- 4 NVIDIA Quadro RTX 6000 24GB 

# Kullanım:
## Python ortamınu kurun
```
conda create -n melanoma python=3.7 pip
pip install -r requirements.txt
```

## Veri setlerini nidirin
```
cd data/isic2019
bash download_isic2019.sh
unzip ISIC_2019_Training_Input.zip
cd ..
kaggle competitions download -c siim-isic-melanoma-classification
unzip siim-isic-melanoma-classification.zip 
```


## ISIC 2019 modelini eğitin
```
cd src/etl
python 0_create_isic2019_splits.py
cd ..
python run.py configs/isic2019/mk001.yaml train --gpu 0,1,2,3 --num-workers 4
```

## Run ISIC 2019 model on ISIC 2020 data
Eğer farklı bir isim kullanıyorsanız `src/configs/predict_isic2019.yaml` içindeki `model_checkpoints`'i kendi model isminize göre değiştirin
```
cd src
python run.py configs/predict/predict_isic2019.yaml
```

## 'nevus' etiketlerini ISIC 2020 veri setine uygulayın
```
cd src/eval
python nevi.py
cd ../etl
python 10_combine_cdeotte_nevi_with_isic2019.py
```

## Uyguladığınız 'nevus' etiketleriyle beraber ISIC 2019 ve ISIC 2020 veri setlerini kullanarak modelinizi eğitin.
```
cd src
bash train_kfold.sh
```

## Pseudo etiketlerinizi oluşturun
```
cd src
python run.py configs/predict/predict_bee_nometa.yaml predict_kfold \
    --gpu 0 --backbone tf_efficientnet_b6_ns \
    --model-config configs/bee/bee508.yaml \
    --checkpoint-dir ../checkpoints/bee508/tf_efficientnet_b6_ns/ \
    --save-file ../lb-predictions/bee508_5fold.pkl --num-workers 4
cd etl
python 12_make_pseudo_nometa.py
```

## Pseudo etiketli veri setini kullanarak modelinizi eğitin.
```
cd src
bash train_kfold_pseudolabel.sh
```

## Test veri setinde modelinizin performansını ölçün
```
cd src
bash inference.sh
```








