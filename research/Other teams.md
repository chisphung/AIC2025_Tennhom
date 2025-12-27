---
share_link: https://share.note.sx/oh2n2bcb#+ZFIGDSk2wYQ6Q229+WRX3t+BzSqiYO7BShLhS7Ki64
share_updated: 2025-07-04T23:36:12+07:00
---
* [SKKU-NDSU - Team 15](https://github.com/daitranskku/AIC2024-TRACK4-TEAM15)
* [Nota - Team 40](https://github.com/nota-github/AIC2024_Track4_Nota)
* [BUPT - Team 33](https://github.com/lxslxs1/AICITY2024_Track4_Team33_MCPRL)
* [VNPT - Team 9](https://github.com/vnptai/AICITY2024_Track4.git)
* [SKKU-AutoLab - Team 5](https://github.com/SKKUAutoLab/aicity_2024_fisheye8k)
## VNPTAI
### Prepare dataset: 
- ./dataset/fisheye8k
- Dùng VisDrone để tạo dữ liệu giống Fisheye8K (data upsampling) rồi chuyển sang Fisheye image 
	- Dùng ifish_augmentation
### Inference
- Pseudo labeling: dùng 3 fold cross validation approach
- Model: Co-DETR
## Nota team 
[github](https://github.com/nota-github/AIC2024_Track4_Nota/tree/feature_2)
### Prepare dataset
- [Fisheye8K](https://github.com/MoyoG/FishEye8K) dataset into Co-DETR/data/aicity_images
- [AI CITY test set](https://www.aicitychallenge.org/2024-data-and-evaluation/) into Co-DETR/data/aicity_images
- Dùng semi supervision, upscale ảnh dùng SR (giống remini) 
### Inference 
- Có nhắc đến SAHI 
- Dùng SAHI để giữ nguyên chất lượng ảnh nhưng train/chạy nhanh hơn 
	- SAHI giống như CNN, chia một bức ảnh ra thành nhiều phần nhỏ, trượt giống như kernel, dự đoán các object trong kernel đó 
- Có dùng Semi-supervised learnin: 
	- Semi-supervised learning is a type of [machine learning](https://www.geeksforgeeks.org/machine-learning/) that falls in between supervised and unsupervised learning
	- Semi-supervised learning is particularly useful when there is a large amount of unlabeled data available, but it’s too expensive or difficult to label all of it.
- Pseudo labeling: label cho cả các target không cần dự đoán 

## SKKU-Autolab
[github](https://github.com/SKKUAutoLab/aicity_2024_fisheye8k)
### Data: 
- Dùng cycleGAN để tạo thêm dataset ban đêm, cân bằng dữ liệu 
### Inference
- open vocab pseudo labels: 
- YoloR, YoloV7, YoloV8, YoloV9
- Data aug: rotate, scaling, flip, mixup, mosiac....
- Ensembling --> YOLOR-D6

___
Cả ba team đều không tối ưu trên Jetson 
Suitable datasheet: [link](https://universe.roboflow.com/master-final-dataset-curation-v1a/cctv-curation-dataset-1-hhibk)
Related research [link](https://www.albany.edu/faculty/mchang2/files/2023-10_ICIP_Fisheye_MOT.pdf)
