# SIGKDD2023_Latent_Outlier_Softening
Code and dataset for our paper "Open-Set Semi-Supervised Text Classification with Latent Outlier Softening", SIGKDD2023

## Overview

In this project, we focus on the open-set semi-supervised text classification(OSTC) task in text classification.

As open-set semi-supervised learning has not been studied in text classification, we follow the pipeline approach in open-set semi-supervised image classification to combine the semi-supervised learning model with outlier detectors. Specifically, we train a pipeline OSTC model in two steps: (1) first, train existing outlier detectors in text classification on the labeled-text set D𝑎 and leverage them to filter out OOD examples in the unlabeled-text set D𝑢 ; (2) then, use the remaining unlabeled-text set to train the STC model. This approach reduces the negative impact of the OOD texts by excluding OOD examples by hard filtering. For outlier detection, we introduce the recent outlier detector used in text classification, including MSP [1], DOC [2], LMCL [3] and Softmax [4]. Meanwhile, we select the recent UDA [5] and MixText [6] as the STC models in the pipeline approach, which leverage BERT and consistency training with data augmentation techniques. We name these pipeline models combining the outlier detectors and STC models with a “+” symbol, e.g., UDA+MSP, UDA+DOC, etc.

is simple and has shown to be effective, its hard OOD filtering strategy may cause error recognition of ID or OOD texts that harm the OSTC training. To tackle this problem, we propose a Latent Outlier Softening (LOS) framework shown in following Figure to replace hard OOD filtering with OOD softening that adaptively assigns soft weights to objectives of text examples, where the lower weights are resorted to weaken the negative impact of OOD texts. LOS is naturally derived from latent variable modeling. The details of assumptions and derivation may be difficult to describe here clearly, so if you want to learn more about the model, try to read the paper.

![微信截图_20230530132520](C:\Users\Lenovo\Desktop\微信截图_20230530132520.png)

## Datasets

We create three benchmarks from existing text classification datasets to evaluate OSTC: AGNews, DBPedia and Yahoo. The statistics of the datasets is below. # lab, #unl., #val and #test respectively denote the labelled, unlabeled, validation and test texts for each class. And #ID (|Y|), #OOD denote the number of ID classes and OOD classes, respectively.

![微信截图_20230530133604](C:\Users\Lenovo\Desktop\微信截图_20230530133604.png)

We provide the processed data as well as the raw data for these three datasets. The datasets can be downloaded from the [Baidu Cloud](https://pan.baidu.com/s/1qd9XhU_1N3GHfq95-d-_OA) (access code: hkw6).
