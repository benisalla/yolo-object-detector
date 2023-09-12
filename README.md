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

  The Stanford Question Answering Dataset (SQuAD) serves as a reading comprehension benchmark. It comprises questions formulated by crowdworkers, centered around a collection of Wikipedia articles. Each question is designed to be answered by a specific excerpt or segment of text from its corresponding reading passage. In some cases, the question could also be about content that cannot be answered directly from the provided passage.

**Below are some key statistics about the dataset:**

1. **Combined Diversity**: SQuAD2.0 combines the strengths of SQuAD1.1's 100,000 questions with an additional 50,000 unanswerable questions. This comprehensive compilation enhances the dataset's scope and complexity.

2. **Adversarial Craftsmanship**: A distinctive feature of SQuAD2.0 is the inclusion of unanswerable questions created through adversarial techniques. These questions, over 50,000 in number, closely resemble answerable queries, posing an advanced challenge for systems.

3. **Dual Evaluation Criterion**: SQuAD2.0 introduces a dual evaluation criterion. Systems are evaluated not only on their ability to answer questions but also on their capacity to recognize scenarios where no suitable answer can be derived from the provided information.

4. **Abstention Competence**: Noteworthy is the demand for systems to showcase their capability of abstaining from answering when appropriate. This feature adds a layer of sophistication to the evaluation process.

  

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
