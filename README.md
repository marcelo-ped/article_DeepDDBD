## **How To Clone This Repository**
Install git lfs:
```bash
sudo apt install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.python.sh | bash
git clone lfs https://github.com/marcelo-ped/article_DeepDDBD.git
```
* Install the python package dependencies via pip.
```bash
pip install -r requirements.txt
```

* Download the datasets used on this work
```bash
gdown https://drive.google.com/drive/folders/1MzlQ9I1r3QXpOA03PkJ2jn3GQ7UhhFcM --folder
mv artigo_Marcelo_ITSC/*.zip .
rm -rf artigo_Marcelo_ITSC
unzip AUC_Dataset.zip
unzip Kaggle_Dataset.zip
unzip DV_Dataset.zip
```
    
## **Running**
```bash
python demo.py --help
python demo.py 
--dataset 'One from DV/Kaggle/AUC.'
--neural_network 'One from VOLO/efficientNet/resnet152.'
```
