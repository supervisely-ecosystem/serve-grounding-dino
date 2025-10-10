<div align="center" markdown>
<img src="https://github.com/supervisely-ecosystem/serve-grounding-dino/releases/download/v0.0.1/dino.png"/>  

# Serve Grounding DINO

<p align="center">
  <a href="#Overview">Overview</a> •
  <a href="#How-To-Run">How To Run</a> •
  <a href="#Acknowledgment">Acknowledgment</a>
</p>

[![](https://img.shields.io/badge/supervisely-ecosystem-brightgreen)](../../../../supervisely-ecosystem/serve-grounding-dino)
[![](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://supervisely.com/slack)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/supervisely-ecosystem/serve-grounding-dino)
[![views](https://app.supervisely.com/img/badges/views/supervisely-ecosystem/serve-grounding-dino.png)](https://supervisely.com)
[![runs](https://app.supervisely.com/img/badges/runs/supervisely-ecosystem/serve-grounding-dino.png)](https://supervisely.com)

</div>

# Overview

Grounding DINO extends a closed-set object detection model with a text encoder, enabling open-set object detection. The model achieves remarkable results, such as 52.5 AP on COCO zero-shot. Open-set object detection is trained using existing bounding box annotations and aims at detecting arbitrary classes with the help of language generalization.

Grounding DINO is built upon the DETR-like model DINO, which is an end-to-end Transformer-based detector. Grounding DINO is a dual-encoder-single-decoder architecture. It contains an image backbone for image feature extraction, a text backbone for text feature extraction, a feature enhancer for image and text feature fusion, a language-guided query selection module for query initialization, and a cross-modality decoder for box refinement.

![grounding dino](https://github.com/supervisely-ecosystem/serve-grounding-dino/releases/download/v0.0.1/grounding_dino_architecture.png)

For each (Image, Text) pair, Grounding DINO first extracts vanilla image features and vanilla text features using an image backbone and a text backbone, respectively. The two vanilla features are fed into a feature enhancer module for cross-modality feature fusion. After obtaining cross-modality text and image features, a language-guided query selection module is used to select cross-modality queries from
image features. Like the object queries in most DETR-like models, these crossmodality queries will be fed into a cross-modality decoder to probe desired features from the two modal features and update themselves. The output queries of the last decoder layer will be used to predict object boxes and extract corresponding phrases.

# How To Run

**Step 1.** Select pretrained model architecture and press the **Serve** button

![pretrained_models](https://github.com/supervisely-ecosystem/serve-grounding-dino/releases/download/v0.0.1/grounding_dino_deploy.png)

**Step 2.** Wait for the model to deploy

![deployed](https://github.com/supervisely-ecosystem/serve-grounding-dino/releases/download/v0.0.1/grounding_dino_deploy_2.png)

# Acknowledgment

This app is based on the great work [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO). ![GitHub Org's stars](https://img.shields.io/github/stars/IDEA-Research/GroundingDINO?style=social)
