<div align="center">
  <h1>YOLOv1: Building Object Detection Model from scratch</h1>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267430814-f1bbe296-e561-45ea-96c3-9afa4c34fdb8.png" width="700" height="300" />
</div>



In this project, we've meticulously developed YOLOv1 from the ground up, and we've gone a step further by conducting comprehensive training on both Kaggle and Colab platforms.


## Table of Contents
- [Introduction](#introduction)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
- [Inference](#inference)
- [Evaluation](#evaluation)
- [Results](#results)
- [Customization](#customization)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The You Only Look Once (YOLO) approach suggests using an end-to-end neural network to make simultaneous predictions of bounding boxes and class probabilities. This contrasts with previous object detection algorithms that adapted classifiers for detection purposes.



## Getting Started

Explain how to get started with your project.

### Prerequisites

List the software and hardware prerequisites required to run your code. Include information about which versions of Python, PyTorch, and other dependencies are needed.

### Installation

Provide step-by-step instructions for installing your project and its dependencies.


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

### Extracting Context, Question and answer

##### Note

In our JSON object, the keys have been reversed for clarity and to avoid any ambiguities. 
Here are the key mappings:

- `txetnoc` maps to `context`
- `noitseuq` maps to `question`
- `rewsna` maps to `answer`



```
import json

big_file = []
for index in range(1, 15):
  print(index)
  with open(f'/content/dataset_{index}.json') as json_file:
      data = json.load(json_file)

  clean_data = []
  for para in data["data"]:
    for qas in para["paragraphs"]:
      context = qas["context"]
      for quests in qas["qas"]:
        question = quests["question"]
        for ans in quests["answers"]:
          answer = ans["text"]
          clean_data.append({"txetnoc": context,
                            "noitseuq": question,
                            "rewsna": answer})
  big_file = [*big_file, *clean_data]



with open(f'/content/big_data.json', "w") as json_file:
    json.dump(big_file, json_file)
```



## Training


## Inference


## Evaluation


## Results


## License
MIT


<div align="center">
  <h4>I hope you have enjoyed this little project</h4>
  <img src="https://github-production-user-asset-6210df.s3.amazonaws.com/89405673/267407911-fe4de299-15c5-4689-9ce5-500b82cd128c.gif" width="300" height="300"/>
</div>
