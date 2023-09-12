<div align="center">
  <h1>YOLOv1: Building Object Detection Model from scratch</h1>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267430814-f1bbe296-e561-45ea-96c3-9afa4c34fdb8.png" width="700" height="300" />
</div>



In this project, we've meticulously developed YOLOv1 from the ground up, and we've gone a step further by conducting comprehensive training on both Kaggle and Colab platforms.


## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Dataset Preparation](#DataPreProcessing)
- [Training](#training)
- [Inference](#inference)
- [About Me](#AboutMe)
- [Contact Me](#ContactMe)
- [License](#license)

## Introduction

The You Only Look Once (YOLO) approach suggests using an end-to-end neural network to make simultaneous predictions of bounding boxes and class probabilities. This contrasts with previous object detection algorithms that adapted classifiers for detection purposes.





## Getting Started

To reuse this implementation, consider the following scenarios:

- If you are utilizing Google Colab, please upload your notebook and proceed to download the Mapillary dataset. Ensure that you save it to your Google Drive and verify that the file paths are correctly configured. Execute the CSV file generation process. It is advisable to follow these steps to ensure accuracy.

- For users on Kaggle, the process is similar to that of Google Colab. However, there is no need to allocate time for dataset construction since Kaggle already provides access to two Mapillary datasets, one of 12GB and another of 29GB.





### Prerequisites

The key requirement here is to utilize a GPU (CUDA). Without it, the process will continue indefinitely, yielding no meaningful results.



## Data Collection

<div align="center">
  <h5>Mapillary Dataset</h5>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267437869-fa4e92c4-e5b7-45e9-a3d0-528052492352.jpeg" width="480" height="220" />
</div>

  The Mapillary traffic sign dataset is the largest and most diverse publicly accessible dataset for teaching machines the detection and recognition of traffic signs. It consists of 100,000 images from around the world, featuring significant variability in factors like weather, time of day, camera sensors, and perspectives.

**Below are some key statistics about the dataset:**
- The dataset contains 300+ unique traffic sign classes with over 320,000 labeled signs, including 52,000 fully verified and human-annotated images alongside partially annotated ones.

- Extensive experiments were conducted to establish a solid foundation for detection and classification tasks. The dataset's diversity supports efficient transfer learning for larger traffic sign datasets.

- Transfer learning using this dataset, as well as previously published traffic sign datasets, was analyzed. Pretraining on this dataset boosts average precision in binary detection by around 6%, thanks to its comprehensive and diverse nature.

  

## Data PreProcessing

### building input-label pair


```
import os
import csv

def get_all_file_names(directory):
    file_names = os.listdir(directory)
    return file_names

def create_dataset(min_idx, max_idx, csv_name, file_names, dist_path):

    with open(f"{dist_path}/{csv_name}.csv", mode="w", newline="") as csv_file:
        for i, fl_name in enumerate(file_names[min_idx:max_idx+1]):
            name = fl_name.split(".")[0]
            img_name = name+".jpg"
            label_name = name+".txt"
            data = [img_name, label_name]
            writer = csv.writer(csv_file)
            writer.writerow(data)


directory_path = "/content/Yolo_Test_Data/annotations"
dist_path="/content/Yolo_Test_Data/"

file_names = get_all_file_names(directory_path)

train_size = [0, int(0.8 * len(file_names))+1]
test_size = [train_size[0], int(0.2 * len(file_names))+1]
val_size = [test_size[0], int(0.2 * len(file_names))+1]


create_dataset(min_idx=train_size[0], max_idx=train_size[1], csv_name="big_train",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=test_size[0], max_idx=test_size[1], csv_name="big_test",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=val_size[0], max_idx=val_size[1], csv_name="big_val",file_names=file_names, dist_path=dist_path)

# this is just when building the model
create_dataset(min_idx=0, max_idx=101, csv_name="tiny_train",file_names=file_names, dist_path=dist_path)
create_dataset(min_idx=101, max_idx=121, csv_name="tiny_test",file_names=file_names,dist_path=dist_path)
create_dataset(min_idx=121, max_idx=141, csv_name="tiny_val",file_names=file_names, dist_path=dist_path)
```




## Training

<div align="center">
  <h4>Loss Value and mAP after 2000 iter</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267448009-008d8cc3-f4ff-4faf-8bb3-c6ac85270bbb.jpeg" width="700" height="300"/>
</div>





## Inference

<div align="center">
  <h4>after 2000 iter mAP ~= 50%</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267453546-a30a0581-788e-49d6-a8dc-8607d7c9656a.jpeg" width="700" height="300"/>
</div>

<div align="center">
  <h4>after 4000 iter mAP ~= 80%</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267452962-b060ef01-78ba-4fa4-a245-9e85b4b15d29.jpg" width="700" height="300"/>
</div>

## About Me 🚀
🎓 I’m ismail ben alla, i love neural networks 😍. </br>
⛵ AI enthusiast </br>
✅ optimist and hardworker </br>
🌱 Learning advanced computer science technologies</br>
😄 I LOVE WHAT I AM DOING</br>


## Contact Me
<p align="left">
<a href="https://twitter.com/ismail_ben_alla" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/twitter.svg" alt="ismail_ben_alla" height="30" width="40" /></a>
<a href="https://linkedin.com/in/ismail-ben-alla-7144b5221/" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/linked-in-alt.svg" alt="ismail-ben-alla-7144b5221/" height="30" width="40" /></a>
<a href="https://instagram.com/ismail_ben_alla" target="blank"><img align="center" src="https://raw.githubusercontent.com/rahuldkjain/github-profile-readme-generator/master/src/images/icons/Social/instagram.svg" alt="ismail_ben_alla" height="30" width="40" /></a>
</p>


## License
MIT


<div align="center">
  <h4>I hope you have enjoyed YOLO</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267407911-fe4de299-15c5-4689-9ce5-500b82cd128c.gif" width="300" height="300"/>
</div>
