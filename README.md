# Make Sure You Read all Instructions :

# Requirements 
#### You have to Installed all the requirments. Save all the below requirements in requirements.txt
#### Run this line in cmd/shell :  pip install -r requirements.txt

# ⭐Plant-Disease-Detection
* Plant Disease is necessary for every farmer so we are created Plant disease detection using Deep learning. In which we are using convolutional Neural Network for classifying Leaf images into 39 Different Categories. The Convolutional Neural Code build in Pytorch Framework. For Training we are using Plant village dataset. Dataset Link is in My Blog Section.
# ⭐Plant-Disease-Segmentation (Addon)
* Segmentation involves delineating and highlighting specific regions of interest within the plant images. This granular approach not only aids in accurate identification but also contributes significantly to targeted treatment strategies.

## Configure Project in your Machine
# Requirements:
* You must have python install in your machine.
* Create a Python Virtual Environment & Activate Virtual Environment [Link](https://docs.python.org/3/tutorial/venv.html)
* Install all the dependencies using below command
    `pip install -r requirements.txt`

## Train 2 model in total, one for detection and one for segmentation:
 * If you want swiftly to skip the trainning part (30` miniutes) to preview stage, download 2 trained project's model here:
[Detection Model](https://drive.google.com/file/d/1XpG04vKaUloLPgQqJgkJ2hk5S4fus_mP/view?usp=sharing)
[Segmentation Model](https://drive.google.com/file/d/1Ylk8gH3eLgiK7qDtW6g7D30Nj-5-L8wH/view?usp=sharing)
* Put each of download models to destination specifically accoording to final instruction of two model below
# Detection model:
* Download dataset for training from [here](https://drive.google.com/file/d/1K9UPT2ztU-Y22v8cVNh9JI7KDpRK8Hur/view?usp=sharing)
* Extract and place all dataset contains class directories(ex: Apple___Cedar_apple_rust, Cherry___healthy,...) in Dataset folder(create if itsn't exist one) to ipynb file from Plant-Disease-Detection\Model
* Run code inside Plant Disease DetectionCode.ipynb for trainning, model result ('plant_disease_model_1.pt') will be exported, place the model into plant-Disease-Detection\Flask Deployed App
# Segmentation model:
* Download dataset for training from [here](https://drive.google.com/file/d/1K9UPT2ztU-Y22v8cVNh9JI7KDpRK8Hur/view?usp=sharing)
* Extract and place all folders contain aug_data vs orig_data into input\leaf_disease_segmentation 
* Run code src\train.py for trainning, model result ('best_model_iou.pth') will be exported and placed in the outputs folder automatically, done!

## ⭐Run project
* Go to the `Flask Deployed App` folder.
* Run the Flask app using below command `python3 app.py`
# Tesing
* If you do not have leaf images then you can use test images located in input\inference_data\images folder
* Each Image have it's disease name so you can verify model is working perfact or not.
