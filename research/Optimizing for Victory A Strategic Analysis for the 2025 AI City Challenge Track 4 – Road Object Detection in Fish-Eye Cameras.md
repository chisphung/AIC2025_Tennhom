---
share_link: https://share.note.sx/wxdae171#+kL74RYRYgdgtpg2GmW4WzxoXVJmacxIQ8Kj+boJIjU
share_updated: 2025-05-14T19:24:41+07:00
---

___

## 1. Decoding the 2025 Challenge: New Frontiers in Fisheye Object Detection

The 2025 AI City Challenge Track 4 continues its focus on Road Object Detection in Fish-Eye Cameras, a domain critical for enhancing traffic monitoring systems. Fisheye lenses offer wide, omnidirectional coverage, reducing the number of cameras needed for comprehensive street and intersection views [User Query]. However, they introduce significant image distortion, necessitating specialized image processing techniques or detector designs.1 This year's challenge builds upon the foundations of previous iterations, particularly the 2024 event, but introduces pivotal new constraints and evaluation metrics that demand fresh strategic thinking.

### 1.1. Deep Dive into Track 4 Specifications and Datasets

A thorough understanding of the challenge's core components—its specific focus, the datasets provided, and data usage constraints—is fundamental for developing a competitive solution.

The primary task remains the detection of five road object classes: Bus (Category ID 0), Bike (Category ID 1), Car (Category ID 2), Pedestrian (Category ID 3), and Truck (Category ID 4) within images captured by fisheye cameras [User Query]. This task was central to the 2024 challenge as well, leveraging the FishEye8K dataset.2

Datasets:

The datasets for the 2025 challenge are clearly defined:

- **Training Data:** The challenge utilizes the train set of the FishEye8K dataset, which comprises 5288 images [User Query]. This dataset was notably published in CVPRW23.1
- **Validation Data:** Participants will use the test set of the FishEye8K dataset, containing 2712 images, for validation purposes [User Query].
- **Test Data:** The official evaluation will be performed on the FishEye1Keval dataset, which consists of 1000 images [User Query].

The FishEye8K dataset features images with resolutions of 1080×1080 and 1280×1280 pixels. Collectively, the train and validation sets (derived from FishEye8K) contain approximately 157,000 annotated bounding boxes across the five specified object classes.1 Annotations are conveniently provided in multiple standard formats: PASCAL VOC (XML), COCO (JSON), and YOLO (TXT).1 This dataset was meticulously curated from traffic monitoring footage in Hsinchu, Taiwan, capturing a variety of traffic patterns, illumination conditions, and viewing angles.1

A critical rule for participation is the **Data Usage Constraint**: "Teams aiming for the public leaderboard and challenge awards must not use non-public datasets in any training, validation, or test sets" [User Query]. This regulation ensures that the competition is driven by algorithmic innovation rather than access to proprietary data.

For practical submission, the organizers provide a Python function to convert image filenames into the required `image_id` format, a necessary step for accurate assembly of the JSON submission file [User Query].

The continuity in using the FishEye8K training and validation sets from the 2024 challenge offers a distinct advantage. It allows current participants to draw upon the experiences, pre-trained models, and data augmentation strategies developed by successful teams from the previous year. However, a significant new element is introduced with the `FishEye1Keval` test set. The images in this set are "extracted from 11 camera videos which were not utilized in the making of FishEye8K dataset" [User Query]. While the fundamental characteristics of fisheye imagery and the object classes remain consistent, these 11 new cameras will inevitably introduce variations. These can include different camera mounting heights and angles, unique lens distortion profiles, and diverse environmental backdrops. The original FishEye8K dataset itself was sourced from 18 different cameras 1, already providing some diversity. Nevertheless, models that might have inadvertently overfit to the specific characteristics of those 18 cameras could face generalization challenges when confronted with data from 11 entirely new sources. Consequently, achieving success in 2025 will hinge not only on mastering the FishEye8K data but also on developing models that exhibit robust generalization capabilities to novel fisheye camera feeds. This underscores the importance of techniques such as diverse data augmentation and domain generalization strategies.

### 1.2. The Paradigm Shift: Understanding the F1-FPS Harmonic Mean Metric

The 2025 challenge introduces a significant change in how solutions are evaluated, moving beyond a singular focus on detection accuracy.

The Primary Evaluation Metric for determining the final ranking is now a harmonic mean that balances F1-score and a normalized Frames Per Second (FPS) score. The formula is:

Score=F1+FPSnormalized​2×F1×FPSnormalized​​

[User Query].

The **F1-Score** component remains the standard harmonic mean of total precision and recall, calculated across all classes using the `eval_f1.py` script provided by the organizers [User Query]. This F1-score was the primary metric for the 2024 leaderboard.6

The FPS Normalization is defined as:

FPSnormalized​=MaxFPSmin(FPS,MaxFPS)​

where MaxFPS is set to 25. The FPS value is the average frames processed per second over all 1000 images in the FishEye1Keval test set [User Query].

This new composite metric represents a major departure from the 2024 challenge, where the F1-score alone determined leaderboard positions.6 The 2025 metric explicitly compels participants to co-optimize for both detection accuracy (F1) and processing speed (FPS).

A key aspect of this new metric is the "25 FPS sweet spot." The `MaxFPS` is capped at 25 for normalization. This means that achieving an actual FPS _above_ 25 yields no further improvement in the `FPS_normalized` component, which saturates at a value of 1.0. For instance, if a solution achieves an FPS of 20, its `FPS_normalized` is 20/25=0.8. If it reaches 25 FPS, `FPS_normalized` becomes 25/25=1.0. However, even if the solution processes images at 30 FPS or higher, `min(30, 25)` remains 25, so `FPS_normalized` is still 25/25=1.0. Once `FPS_normalized` reaches 1.0, the overall score formula simplifies to F1+12×F1×1​, making the F1-score the sole determinant for further improvement. Therefore, the optimal strategy involves first ensuring the solution meets the minimum 10 FPS requirement to avoid disqualification, then aiming to achieve 25 FPS. Beyond 25 FPS, all optimization efforts should pivot to maximizing the F1-score, as pushing FPS higher offers no additional benefit to the final score.

Furthermore, the use of the harmonic mean for the final score has an "unforgiving nature for imbalance." The harmonic mean, 2AB/(A+B), inherently gives more weight to the lower of the two values it combines. This means that solutions excelling significantly in one aspect (e.g., a very high F1-score) but performing poorly in the other (e.g., an FPS only slightly above the 10 FPS minimum) will be heavily penalized. A balanced performance across both F1 and `FPS_normalized` is rewarded more than a lopsided one. Consider two hypothetical teams: Team X achieves an F1 of 0.7 and an FPS of 12 (yielding `FPS_normalized` = 0.48), resulting in a score of approximately 0.569. Team Y, with a lower F1 of 0.6 but an FPS of 25 (`FPS_normalized` = 1.0), achieves a score of 0.75. Despite Team X's superior F1, Team Y's balanced approach leads to a significantly better overall score. This illustrates that contestants cannot treat FPS as a secondary concern. Model selection, training methodologies, and post-processing techniques must be evaluated for their impact on both accuracy and speed from the outset of development. Strategies from 2024 that prioritized F1 at any computational cost, such as those involving complex and slow ensembles, are no longer viable without substantial optimization for speed.

### 1.3. The Jetson Imperative: Real-Time Edge Computing Constraints

A cornerstone of the 2025 challenge is the emphasis on "Eco-Friendly & Real-Time Edge Solution" [User Query]. This introduces stringent hardware and performance requirements.

Participants must submit a **Docker container** housing a real-time running framework. This framework must be optimized for NVIDIA Jetson devices, specifically using ONNX or TensorRT [User Query].

A critical **Efficiency Mandate** is that the submitted solution must achieve an average of at least 10 FPS when evaluated on a Jetson AGX Orin 32GB edge device. Failure to meet this threshold will result in disqualification [User Query]. This is a non-negotiable performance floor.

While owning a Jetson device is not mandatory for participation, the official evaluation of both efficiency (FPS) and effectiveness (F1-score) will be conducted on this specific hardware [User Query]. The **Timing** for FPS calculation is precise: it encompasses the total elapsed real time for processing all 1000 test images, including all framework processes from the moment before the first image is processed until after the last image is fully processed [User Query].

The provision that owning a Jetson device is not mandatory, while inclusive, presents a "develop blind" risk for teams lacking access to such hardware. Performance characteristics, particularly FPS and the efficacy of TensorRT optimizations, can differ substantially between high-end desktop GPUs (commonly used for training) and embedded SoCs like the Jetson AGX Orin. Model architectures, specific operations, memory bandwidth constraints, and cache behaviors vary significantly. For example, an NVIDIA RTX 4090 has vastly different resources compared to a Jetson AGX Orin. TensorRT optimizations are also highly specific to the target NVIDIA GPU architecture; an optimization beneficial on one GPU might not be as effective, or even feasible, on another. Developing without the ability to profile and iterate on the target hardware means teams are making assumptions about performance that might not hold, risking either suboptimal FPS or, worse, disqualification. Therefore, teams should make every effort to gain access to a Jetson AGX Orin or a closely related Jetson family device for development and testing. If direct access is impossible, meticulous research into the performance characteristics of the Jetson AGX Orin and careful selection of models known for Jetson-friendly architectures become paramount.

The requirement for solutions to be optimized using ONNX or TensorRT also heavily influences model architecture choices. TensorRT generally delivers superior performance on NVIDIA hardware like Jetson by performing aggressive layer fusions, precision calibrations, and other hardware-specific optimizations. However, not all model layers or architectures are natively supported by TensorRT. Complex, novel, or experimental layers might necessitate the development of custom plugins, a task that significantly increases development effort and introduces risk. Even the conversion to ONNX can encounter issues with certain operations. Consequently, selecting models with well-established and thoroughly documented ONNX export and TensorRT conversion pathways (e.g., many YOLO variants, EfficientDets) represents a lower-risk, potentially higher-reward strategy for successfully meeting the FPS target. Teams should investigate the TensorRT support for candidate model architectures _before_ committing extensive development resources. Prioritizing models known for good performance _after_ TensorRT optimization on similar edge devices is advisable. This might mean forgoing the absolute newest, largest, or most exotic state-of-the-art model if its edge deployment and optimization story is unclear or unproven.

## 2. Insights from 2024 Track 4 Victors: Architectures and Strategies

Analyzing the methodologies of the top-performing teams from the 2024 AI City Challenge Track 4 provides invaluable lessons. The 2024 challenge shared the same Fisheye8K dataset and the core task of road object detection in fisheye cameras.2 The primary evaluation metric was the F1-score.6 While the 2025 rules introduce new complexities, the successful approaches of 2024 offer a strong starting point.

### 2.1. Performance Overview of 2024 Top Teams (F1 Scores)

The following table summarizes the leading teams from the 2024 AI City Challenge Track 4, their F1 scores, and links to their technical papers and publicly available code repositories. These resources are crucial for understanding the strategies that led to their success.

**Table 1: Summary of 2024 AI City Challenge Track 4 Top Teams, F1 Scores, and Resources**

|   |   |   |   |   |   |
|---|---|---|---|---|---|
|**Rank**|**Team ID**|**Team Name**|**F1 Score**|**Paper Link**|**GitHub Repository Link**|
|1|9|VNPT AI|0.6406|([https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Duong_Robust_Data_Augmentation_and_Ensemble_Method_for_Object_Detection_in_CVPRW_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Duong_Robust_Data_Augmentation_and_Ensemble_Method_for_Object_Detection_in_CVPRW_2024_paper.pdf)) 8|([https://github.com/vnptai/AICITY2024_Track4](https://github.com/vnptai/AICITY2024_Track4)) 9|
|2|40|Nota|0.6196|([https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Shin_Road_Object_Detection_Robust_to_Distorted_Objects_at_the_Edge_CVPRW_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Shin_Road_Object_Detection_Robust_to_Distorted_Objects_at_the_Edge_CVPRW_2024_paper.pdf)) 10|(https://github.com/nota-github/AIC2024_Track4_Nota) 9|
|3|5|SKKU-AutoLab|0.6194|[Pham et al., 2024](https://openaccess.thecvf.com/content/CVPR2024W/AICity/papers/Pham_Improving_Object_Detection_to_Fisheye_Cameras_with_Open-Vocabulary_Pseudo-Label_Approach_CVPRW_2024_paper.pdf) 12|([https://github.com/SKKUAutoLab/aicity_2024_fisheye8k](https://github.com/SKKUAutoLab/aicity_2024_fisheye8k)) 9|

This table provides a concise overview of the top performers, their achieved accuracy benchmarks (F1 scores), and direct access to their detailed methodologies and code. These F1 scores set a high bar for accuracy that 2025 teams will likely still aspire to, even with the added FPS constraint.

### 2.2. Team VNPT AI (1st Place, F1: 0.6406) 6

Team VNPT AI secured the top position with their paper titled "Robust Data Augmentation and Ensemble Method for Object Detection in Fisheye Camera Images".6

Core Strategy:

Their success was built on several pillars:

- **Data Augmentation:** A key innovation was a "novel data augmentation method" applied to the VisDrone dataset.8 VisDrone was selected due to its perceived similarities to the Fisheye8K dataset, and the augmentation aimed to generate synthetic data mimicking Fisheye8K's characteristics. This underscores the value of creatively expanding training data, even by adapting datasets from different (though visually related) domains.
- **Pseudo-Labeling:** The team implemented a sophisticated pseudo-labeling strategy involving a 3-fold cross-validation approach. Pseudo-labels were generated by ensembling the results from these three model folds. These pseudo-labels were then combined with the original training and validation data for subsequent training rounds.8
- **Models:** Their publicly available GitHub repository indicates the use of several powerful object detection architectures, including Co-DETR, YOLOR-W6, YOLOv9-e, and InternImage.14
- **Ensembling:** The paper explicitly highlights an "ensemble method" as a critical component of their high-performing solution.8 Their GitHub repository further confirms the ensembling of different models and checkpoints.

**Fisheye Distortion Handling:** This was primarily addressed through their robust and diverse data augmentation techniques and by training on a rich dataset supplemented with synthetic and pseudo-labeled examples.

**Edge Deployment/FPS:** The team's published paper 8 and related news articles 15 predominantly focus on their F1 score and victory in the 2024 challenge. There is no specific mention of FPS metrics, Jetson deployment, or ONNX/TensorRT optimization for their Track 4 solution. Their GitHub repository for Track 4 14 also lacks these specific details regarding edge deployment.

The approach taken by VNPT AI highlights the significant F1 score improvements achievable through external data augmentation and sophisticated pseudo-labeling. While the FishEye8K dataset is substantial, its inherent variations are finite. Fisheye distortion introduces unique visual appearances for objects. By identifying the VisDrone dataset as having "resemblances" 8—likely in terms of object density, the prevalence of small objects, and varied viewpoints, even if not inherently fisheye—and developing a "novel data augmentation method" to bridge the domain gap, they effectively enriched their training data. Furthermore, their pseudo-labeling process was not a simple, one-time step but involved ensembling from cross-validated models 8, indicating a focus on generating high-quality and reliable pseudo-labels. These combined efforts substantially increased the effective size and diversity of their training data, enabling their models to learn more robust features, particularly for challenging fisheye-specific distortions and variations in object scale. For the 2025 challenge, while adhering to the "no non-public datasets" rule [User Query], teams can still explore the use of _publicly available_ datasets like VisDrone (if licensing permits such use) for pre-training or as inspiration for developing complex augmentation pipelines that simulate similar visual characteristics. The emphasis should be on the _method_ of augmentation and pseudo-labeling inspired by such external resources.

### 2.3. Team Nota (2nd Place, F1: 0.6196) 6

Team Nota achieved the runner-up position with their work, "Road Object Detection Robust to Distorted Objects at the Edge Regions of Images".6

Core Strategy:

Nota's approach was characterized by a targeted effort to address specific challenges in fisheye imagery:

- **Problem Focus:** Their research explicitly concentrated on rectifying common failure modes, such as the inability to detect small or heavily distorted objects located at the periphery of images, and the misclassification of non-target objects due to distortion effects.10
- **Models:** The team employed Co-DETR as their base architecture, utilizing both ViT-L (Vision Transformer - Large) and Swin-L (Swin Transformer - Large) backbones.10 These models were pre-trained on large-scale datasets like Objects365, COCO, and LVIS before being fine-tuned on FishEye8K.10
- **Fisheye Distortion Handling:**
    - **SAHI (Slicing Aided Hyper Inference):** To improve the detection of small objects, particularly those at the image edges, Nota integrated SAHI into their inference pipeline. This technique involves partitioning the original image into overlapping slices (typically 2/3 of the image dimensions with a 0.25 overlap ratio) and performing inference on each slice, often after resizing the slice to the model's input dimensions.10
    - ==**Pseudo-labeling for Non-Target Objects:** They implemented a pseudo-labeling strategy to address the misclassification of distorted non-target objects (e.g., street signs appearing similar to vehicles). A Co-DETR model pre-trained on the LVIS dataset (which includes 1203 object categories) was used to generate pseudo-labels for these non-target objects, enabling the primary detection model to learn to discriminate them from the five target classes.10==
- **Data Augmentation:** Their GitHub repository 11 mentions the use of basic augmentations and image rotation. Their paper also refers to histogram equalization as part of their augmentation strategy.10
- **Super-Resolution:** The team utilized a pre-trained StableSR model to upscale images for both training and testing phases, aiming to enhance detail for better detection.11
- **Ensembling:** Nota employed Weighted Boxes Fusion (WBF) to aggregate predictions from an ensemble of detectors, each configured with different combinations of their proposed methods.10 Their GitHub repository lists nine distinct model outputs that were combined using WBF.11

**Edge Deployment/FPS:** Nota's paper acknowledges that the computational complexity of their ensemble method is "not practical for edge deployment".10 They identified future work to include model compression techniques like knowledge distillation and network pruning. T==heir GitHub repository for Track 4 11 includes a Dockerfile but does not provide specific Jetson optimization scripts (ONNX/TensorRT beyond base Co-DETR capabilities) or FPS results for this particular challenge submission.== It's worth noting that Nota, as a company, specializes in AI optimization and edge AI solutions 17, indicating underlying expertise in this area, even if not fully detailed for their 2024 Track 4 solution.

Nota's targeted attack on fisheye edge distortion and small objects provides significant takeaways. Generic object detectors often exhibit their poorest performance at the peripheries of fisheye images due to extreme geometric distortions and the resulting diminutive scale of objects.1 Nota observed that "state-of-the-art deep learning-based object detectors fail to detect objects in the edge regions" and that "non-target objects exhibit visual distortions at the edges...resulting in visual similarities with the target objects".10 The SAHI technique 10 directly counters the issue of small objects by effectively allowing the detector a "closer look" through processing scaled-up image patches. Simultaneously, their strategy of pseudo-labeling non-target objects 10 helps the model learn to differentiate and ignore distractor items that become visually confusing due to distortion==. For the 2025 challenge, incorporating patch-based inference techniques like SAHI (or similar methods, such as those inspired by SAM from Meta) and developing a robust strategy to mitigate false positives arising from distorted non-target objects will be critical. However, it is important to recognize that SAHI itself introduces computational overhead due to multiple inference passes per image. Therefore, its impact on FPS must be carefully managed to align with the 2025 evaluation metric. An efficient implementation of SAHI or its selective application to specific image regions might be necessary.==

### 2.4. Team SKKU-AutoLab (3rd Place, F1: 0.6194) 6

Team SKKU-AutoLab secured the third position with their paper, "Improving Object Detection to Fisheye Cameras with Open-Vocabulary Pseudo-Label Approach".6

Core Strategy:

Their framework incorporated several innovative elements:

- **Image Generation (Style Transfer):** To address the limited availability of annotated nighttime images in the FishEye8K dataset, SKKU-AutoLab employed CycleGAN. This unsupervised image-to-image translation technique was used to synthetically generate nighttime images from existing daytime images, thereby balancing the dataset across different lighting conditions.12
- **Open-Vocabulary Pseudo-Labels (OVPL):** A distinctive feature of their approach was the use of YOLO-World, a zero-shot open-vocabulary object detector. This model, after being fine-tuned on the FishEye8K dataset, was used to generate pseudo-labels for the _unannotated test set_ of FishEye8K.12 These generated labels were then manually reviewed and slightly corrected before being incorporated into the training data.
- **Models:** The team experimented with a range of single-stage object detectors, including YOLOR, YOLOv7, YOLOv8, and YOLOv9. Their final submission utilized an ensemble of three YOLOR-D6 models, each trained with different input image sizes (1280, 1536, and 1920 pixels).12 Their GitHub repository confirms the usage of YOLOR and YOLO-World.13
- **Data Augmentation:** Standard data augmentation techniques such as rotation, scaling, horizontal flipping, random cropping, mixup, and mosaic were applied during training. Additionally, Test-Time Augmentation (TTA) was used during inference, involving multiple scaled versions and flips of each test image.12
- **Ensembling:** They employed a mean ensemble strategy for their YOLOR-D6 models, averaging bounding box coordinates and confidence scores.12

**Hyperparameter Optimization:** The team actively tuned hyperparameters, particularly confidence thresholds, based on feedback from the challenge's evaluation system. Their focus shifted from maximizing Average Precision (AP) to optimizing the F1-score as the competition progressed.12

**Edge Deployment/FPS:** The published paper 12 and the team's GitHub repository 6 primarily detail methods focused on maximizing the F1-score. While the GitHub repository includes Docker setup instructions, ==it does not provide explicit details on Jetson optimization,== ONNX/TensorRT conversion specifics for Track 4, or reported FPS figures. The presentation media for their work 18 was not accessible for analysis by the research system.

SKKU-AutoLab's creative approach to filling data gaps using generative models and open-vocabulary detectors is noteworthy. Their use of CycleGAN to synthesize nighttime images addressed a potential data imbalance in FishEye8K concerning varied lighting conditions.12 The application of YOLO-World, an open-vocabulary model, for pseudo-labeling the unannotated test set represents a sophisticated method to incorporate broader semantic understanding and leverage unlabeled data effectively.12 Open-vocabulary models can detect objects based on textual prompts, even if not explicitly trained on those exact instances in a fully supervised manner initially. Fine-tuning YOLO-World on FishEye8K and then using it to generate pseudo-labels for more data from the same domain (the test set) constitutes a powerful semi-supervised learning strategy. ==For the 2025 challenge, style transfer techniques like CycleGAN remain relevant for augmenting data across diverse lighting or even weather conditions. Leveraging open-vocabulary models for pseudo-labeling, especially if portions of the provided dataset are unlabelled or if self-training paradigms are pursued, is a cutting-edge technique. However, the computational cost of YOLO-World itself during inference (if it were part of the main detection pipeline rather than an offline tool) would need careful consideration in light of the FPS requirements. Its primary value in SKKU-AutoLab's 2024 solution was in the _offline_ generation of pseudo-labels.==

### 2.5. Comparative Analysis: Common Success Factors and Divergent Paths in 2024

An examination of the top three teams from 2024 reveals both common threads that contributed to their success and distinct approaches that set them apart.

**Common Themes:**

- **Heavy Reliance on Ensembling:** All three leading teams utilized ensembling techniques, combining predictions from multiple models or model checkpoints to boost final performance.8 This was a clear strategy for maximizing F1 scores.
- **Advanced Data Augmentation:** None of the top teams relied solely on basic augmentations. They employed more sophisticated methods, including style transfer (SKKU-AutoLab 12), specialized augmentations leveraging insights from other datasets like VisDrone (VNPT AI 8), or patch-based processing like SAHI (Nota 10).
- **Pseudo-Labeling / Semi-Supervised Learning:** All three teams incorporated pseudo-labeling in some form to augment their training data and improve model robustness.8
- **Powerful Base Detectors:** Architectures like Co-DETR and various YOLO-family models were common choices, indicating a preference for detectors with strong baseline performance.

**Divergent Approaches:**

- **Source and Strategy for Pseudo-Labels:** VNPT AI employed a self-training approach with ensembling from cross-validated models.8 Nota utilized a powerful LVIS pre-trained model to generate pseudo-labels specifically for non-target objects that could be confused with target classes.10 SKKU-AutoLab leveraged an open-vocabulary model (YOLO-World) to generate pseudo-labels from the unannotated test set.12
- **Specific Distortion Handling Mechanisms:** Nota adopted an explicit strategy for edge distortions using SAHI and pseudo-labeling of non-target objects.10 The other teams appeared to rely more on the robustness gained from diverse data augmentation and comprehensive training data.

**Table 2: Comparative Matrix of Top 3 Teams' 2024 Methodologies**

|                                     |                                                                         |                                                                                                                      |                                                                           |
| ----------------------------------- | ----------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Feature**                         | **Team VNPT AI (1st)**                                                  | **Team Nota (2nd)**                                                                                                  | **Team SKKU-AutoLab (3rd)**                                               |
| **Base Models**                     | Co-DETR, YOLOR-W6, YOLOv9-e, InternImage 14                             | Co-DETR (ViT-L, Swin-L backbones) 10                                                                                 | YOLOR-D6 (ensemble), YOLO-World (for OVPL) 12                             |
| **Fisheye Distortion Handling**     | Robust data augmentation, diverse data (implied) 8                      | SAHI for edge objects, pseudo-labeling of distorted non-target objects 10                                            | Distortion-aware training via augmented data, model learning (implied) 12 |
| **Data Augmentation Strategy**      | Novel method using VisDrone to generate Fisheye8K-like synthetic data 8 | Basic augmentation, rotation, histogram equalization, Super-Resolution (StableSR) 10                                 | CycleGAN (day-to-night), standard augs (mosaic, mixup), TTA 12            |
| **Pseudo-Labeling Source/Method**   | Ensemble from 3-fold CV models on training/validation data 8            | LVIS pre-trained Co-DETR for non-target objects 10                                                                   | YOLO-World (OVPL) on unannotated test set, manually corrected 12          |
| **Ensemble Method**                 | Ensemble of different models/checkpoints 8                              | Weighted Boxes Fusion (WBF) of 9 model outputs 10                                                                    | Mean ensemble of 3 YOLOR-D6 models 12                                     |
| **Mention of Edge/FPS for Track 4** | Not detailed in paper/GitHub for Track 4 8                              | Ensemble noted as "not practical for edge deployment"; future work on compression mentioned.10 Dockerfile present.11 | Not detailed in paper/GitHub for Track 4.12 Docker setup present.13       |

A crucial realization from this comparative analysis is that the "F1-score at all costs" mentality prevalent in 2024 is now obsolete for the 2025 challenge. The dominant strategies of 2024—such as heavy ensembling of multiple large models 8 and extensive Test-Time Augmentation (TTA) involving multiple scales and flips 12—were all engineered to maximize the F1 score, often with less stringent constraints on computational expenditure. Team Nota's paper even explicitly stated their 2024 ensemble was not practical for edge deployment.10 These approaches inherently multiply inference time, drastically reducing FPS. The 2025 rules, with their hard 10 FPS minimum on a Jetson device and a new primary evaluation metric that gives significant weight to (normalized) FPS up to 25 FPS [User Query], render such computationally intensive strategies largely incompatible. Directly porting a winning solution from 2024 to the 2025 challenge is highly unlikely to meet the new performance requirements. This necessitates a fundamental shift in design philosophy for 2025: efficiency must be a primary design constraint from the outset, not an optimization applied as an afterthought. This points towards exploring highly efficient single models, very lightweight ensembles, or aggressive model compression techniques if ensembles are deemed necessary.

## 3. Crafting a High-Scoring Solution for 2025: Model and Data Strategies

Translating the lessons from 2024 and the stringent demands of the 2025 challenge into a winning formula requires careful consideration of model selection and data handling. The goal is to strike an optimal balance between detection accuracy (F1-score) and inference speed (FPS) on the Jetson AGX Orin platform.

### 3.1. Balancing Accuracy and Efficiency: Selecting Optimal Model Architectures

The F1-FPS harmonic mean metric necessitates models that are not only accurate but also inherently fast when deployed on the Jetson AGX Orin.

Promising Architectures:

Several model families offer a good starting point for this dual optimization:

- **YOLO Series (e.g., YOLOv7, YOLOv8, YOLOv9, YOLO-NAS, YOLOX):** These models are renowned for their excellent speed/accuracy trade-offs. The availability of various scaled versions (e.g., N - nano, S - small, M - medium, L - large, X - extra-large) allows for fine-tuning the balance. They generally have good support for ONNX export and TensorRT optimization. SKKU-AutoLab utilized YOLOR, YOLOv8, and YOLOv9 in their 2024 experiments 12, and VNPT AI also employed YOLOR and YOLOv9.14
- **EfficientDet/EfficientNet Backbones:** These architectures were explicitly designed with computational efficiency in mind, offering good scalability and performance.
- **Mobile-Friendly Transformers (e.g., MobileViT, EdgeNeXt):** If transformer-based features are desired for their potential accuracy benefits, lightweight variants are essential. While Co-DETR, used by VNPT AI and Nota in 2024, featured larger ViT/Swin backbones 10, smaller vision transformer architectures might be feasible for edge deployment.
- **SSD and variants (e.g., FSSD, MobileNet-SSD):** Single Shot MultiBox Detector (SSD) and its derivatives are classically efficient architectures, though they sometimes lag behind newer models in terms of raw accuracy on complex datasets.

Architectures to approach with caution include very large, slow models, unless aggressive pruning and quantization techniques can demonstrably bring their performance within the required FPS targets on the Jetson. The heavy, multi-model ensembles that characterized 2024 solutions are particularly risky under the new 2025 rules.

**Table 3: Evaluation of Potential Model Architectures for Fisheye Object Detection on Edge**

|   |   |   |   |   |   |   |
|---|---|---|---|---|---|---|
|**Architecture Family**|**Specific Variant Example**|**Typical Performance (e.g., COCO mAP)**|**Strengths**|**Weaknesses**|**Suitability for F1-FPS on Jetson**|**Known Jetson Performance Notes**|
|YOLOv8|YOLOv8s|~37-40 mAP|High FPS, good accuracy, scales well, good TRT support|Smaller variants may struggle with tiny objects|High|Generally performs well with TensorRT on Jetson.|
|YOLOv9|YOLOv9c|~50-53 mAP|State-of-the-art YOLO accuracy, efficient design|Newer, TRT support might be evolving|High|Potentially very good if TRT optimization is smooth.|
|YOLO-NAS|YOLO-NAS-M|~45-48 mAP|Neural Architecture Search optimized for efficiency|Quantization-aware training often needed for best INT8|Medium-High|Designed with quantization in mind.|
|EfficientDet|EfficientDet-D1|~37-40 mAP|Designed for efficiency, scalable|Can be less robust to extreme scale variations than some YOLO variants|Medium-High|Good general efficiency.|
|YOLOX|YOLOX-S|~40-43 mAP|Anchor-free, good speed/accuracy balance||High|Often shows good FPS on edge devices.|
|MobileViT|MobileViT-S|Lower mAP than larger ViTs|Transformer features in a lightweight package|Accuracy might not match CNNs of similar size for detection|Medium|TRT support for transformers needs careful checking.|

This table provides a structured comparison to help narrow down choices. The critical factor is not just theoretical FLOPs or performance on desktop GPUs, but the demonstrated or highly probable efficiency _after_ TensorRT optimization on Arm-based edge SoCs like the Jetson AGX Orin. TensorRT is NVIDIA's key tool for optimizing inference on Jetson devices. Some architectures and their specific operations are better supported or optimize more effectively with TensorRT than others. Therefore, selecting a model architecture known to have a smooth conversion path to TensorRT and good post-optimization speedups on Jetson-like devices significantly de-risks the project and increases the chances of meeting the crucial FPS target. Contestants should actively seek out community benchmarks, NVIDIA's official documentation, or research papers that report FPS figures for various models on Jetson devices when run with TensorRT. Architectures that are notoriously difficult to convert or show limited performance gains from TensorRT should be approached with considerable caution, irrespective of their potential F1 scores on unconstrained hardware.

### 3.2. Addressing Fisheye-Specific Challenges: Distortion, Scale Variation, and Occlusion

Fisheye lenses present unique challenges that standard object detectors, trained primarily on perspective images, may struggle with.

- **Distortion:** Significant radial distortion is inherent to fisheye lenses, particularly at the image periphery. This alters object shapes and appearances, making detection more difficult.1
    - **Strategies:**
        - **Distortion-aware data augmentation:** This is a primary line of defense (detailed further in section 3.3).
        - **Modeling distortion directly:** Some research explores incorporating lens distortion parameters directly into the model architecture or using adaptive mechanisms like deformable convolutions to allow filters to adjust to local geometric distortions.
        - **Patch-based processing:** Techniques such as SAHI, employed by Team Nota in 2024 10, can mitigate distortion effects by processing smaller, potentially less distorted or effectively scaled-up image patches. However, the computational cost of SAHI must be carefully managed for the 2025 F1-FPS metric.
        - **Avoid full image unwarping if too slow:** While transforming the entire fisheye image into a perspective or panoramic view can simplify the task for a standard detector, the unwarping process itself consumes valuable CPU/GPU cycles. This pre-processing step can significantly reduce the overall FPS and must be highly optimized if considered.
- **Scale Variation:** Objects can appear very large when close to the camera center and extremely small when at the periphery or distant.1
    - **Strategies:** Feature Pyramid Networks (FPNs) are a standard architectural component for handling multi-scale objects. Training with images at multiple scales, designing appropriate anchor box sizes and aspect ratios (for anchor-based detectors), or employing anchor-free methods adept at handling scale variations are also important. SAHI can also contribute to better detection of small objects.
- **Occlusion:** Occlusion is common in dynamic traffic scenes.
    - **Strategies:** Robust feature extractors that can infer object presence from partial views, context modeling techniques, and data augmentation methods that simulate various occlusion scenarios are beneficial.

A key consideration for the 2025 challenge is that any pre-processing step, such as full image unwarping, adds to the total inference time per frame and directly impacts the FPS [User Query]. The total time for processing all 1000 test images includes "all framework processes." If a full-image unwarping step is slow, it will degrade the achievable FPS, even if the subsequent object detector is itself very fast. Therefore, methods that handle distortion _within_ the model architecture (e.g., through specialized convolutional layers or attention mechanisms that can adapt to local warping) or via efficient patch-based approaches (like a carefully optimized SAHI) are generally preferable for the F1-FPS metric. The FPS cost of any explicit undistortion pre-processing must be rigorously evaluated. If it proves to be high, priority should be given to models or techniques that can learn robust features directly from distorted images or use computationally cheaper methods to mitigate distortion effects, such as specialized data augmentations or adaptive pooling layers.

### 3.3. Advanced Data Augmentation and Synthetic Data Generation for Fisheye Imagery

Data augmentation is crucial for improving model robustness and addressing the unique challenges posed by fisheye imagery, especially given potentially limited data for specific scenarios. All top-performing teams in 2024 employed advanced augmentation techniques.8

**Techniques to Consider:**

- **Geometric Augmentations Tailored to Fisheye:** Beyond standard operations like flips and rotations, augmentations that simulate varying degrees of fisheye distortion or apply perspective transformations to mimic objects appearing at different locations and orientations within the fisheye field of view can be highly effective.
- **Photometric Augmentations:** Adjusting brightness, contrast, color saturation (color jitter), adding noise, and simulating motion blur can help the model generalize to different lighting conditions, weather effects, and sensor characteristics. SKKU-AutoLab's use of CycleGAN for day-to-night style transfer is an advanced example of this.12
- **Mosaic/Mixup:** These techniques, which combine multiple images or parts of images and their labels, have proven effective in general object detection and were used by SKKU-AutoLab.12 They help the model learn from diverse contexts and reduce overfitting.
- **Cutout/RandErase:** Randomly occluding parts of the image or objects forces the model to learn from partial information, improving robustness to real-world occlusions.
- **Copy-Paste Augmentation:** Carefully selecting objects (especially from rare classes or those exhibiting challenging distortions or poses) and pasting them onto different backgrounds from the dataset can increase the frequency of these critical samples.
- **Leveraging VisDrone-like Datasets (Inspired by VNPT AI 8):** If publicly available datasets with characteristics similar to fisheye traffic scenes (e.g., aerial views with many small objects, varied viewpoints) are permissible under the challenge rules (ensuring no use of non-public ground truth), they can be used to inspire augmentation strategies or for pre-training if licensing allows. The key is the _method_ of augmentation inspired by such datasets, rather than direct use if prohibited.
- **3D Renderers for Synthetic Data (Advanced):** For teams with significant time and resources, creating highly controlled synthetic fisheye scenes using 3D rendering engines can offer maximum control over data generation, allowing for the creation of specific challenging scenarios.

A rich and diverse augmentation strategy serves a dual purpose. Firstly, it improves the F1 score by exposing the model to a wider range of visual variations during training. Secondly, and critically for the 2025 challenge with its new test cameras [User Query], it inherently improves the model's ability to generalize to unseen camera characteristics. The more varied data a model encounters during training, the more robust its learned features become to minor changes in viewpoint, lighting, scale, distortion profiles, and other camera-specific attributes. The 2025 test set, sourced from 11 new cameras, guarantees some degree of domain shift from the FishEye8K training cameras. Aggressive and thoughtfully designed augmentations, particularly those that can mimic variations expected from different camera placements or lens types (e.g., more extreme distortions if an object is at the very edge of a new camera's FoV, different color balances, novel occlusion patterns), will help the model learn features that are less sensitive to the specific source camera. It is advisable not to apply generic augmentations blindly but to think critically about the types of variations fisheye lenses and different camera mountings can introduce and tailor augmentations to cover these anticipated shifts.

### 3.4. The Role of Pseudo-Labeling and Semi-Supervised Learning

Pseudo-labeling and semi-supervised learning (SSL) techniques demonstrated considerable efficacy in the 2024 challenge, with all top teams employing them in some capacity.8 These methods allow models to learn from unlabeled or less confidently labeled data, effectively increasing the size of the training set.

**Sources and Strategies for Pseudo-Labeling:**

- **Self-Training:** This iterative process involves training an initial model, using it to predict labels on unlabeled data, selecting high-confidence predictions as new ground truth (pseudo-labels), and then retraining the model with the augmented dataset. Team VNPT AI used an ensemble from cross-validation to generate robust pseudo-labels in their self-training loop.8
- **Using Stronger Pre-trained Teacher Models:** A powerful model pre-trained on a large, diverse dataset can act as a "teacher" to generate pseudo-labels. Team Nota employed this by using an LVIS-pretrained Co-DETR model to specifically label non-target objects that might be confused with target classes due to fisheye distortion.10
- **Open-Vocabulary Models:** As demonstrated by SKKU-AutoLab, open-vocabulary detectors like YOLO-World can be fine-tuned on the target dataset and then used to generate pseudo-labels for unlabeled portions of the data, such as the unannotated test set.12

Considerations for 2025:

The FishEye1Keval test set is provided without annotations. It could potentially be used for pseudo-labeling or other SSL techniques if the challenge rules permit such use (this needs careful verification; typically, the "no non-public datasets" rule applies to training with external ground truth, while using the challenge's own unlabeled test data for SSL is often allowed). Pseudo-labeling can significantly boost F1 scores but adds complexity and time to the training pipeline. The quality of the generated pseudo-labels is paramount; noisy or incorrect pseudo-labels can degrade model performance.

While highly effective for improving accuracy, complex pseudo-labeling pipelines, especially those involving iterative self-training or requiring multiple model training cycles, can considerably slow down the overall development and experimentation loop. For the 2025 challenge, the added imperative to iterate rapidly on FPS performance and Jetson deployment means that the efficiency of the _entire development cycle_ becomes a critical factor. If the pseudo-labeling pipeline itself is too time-consuming, it will limit the number of full end-to-end experiments—from data preparation through model training to Jetson deployment and benchmarking—that a team can conduct. This suggests a need to strive for pseudo-labeling strategies that are not only effective but also _efficient_. For instance, using a strong teacher model (akin to SKKU-AutoLab's YOLO-World approach 12, or a model pre-trained on a large relevant public dataset) to generate good quality pseudo-labels in a single pass might be preferable to multiple rounds of self-training if development time is constrained. The potential F1 gain from pseudo-labeling must be carefully weighed against the associated development time cost.

## 4. Achieving Real-Time Prowess: Optimization for Jetson AGX Orin

This section is pivotal for success in the 2025 challenge, focusing on the practical steps and considerations for transforming a trained object detection model into a solution that meets the stringent FPS requirements on the NVIDIA Jetson AGX Orin platform.

### 4.1. Strategic Workflow for Model Conversion and Optimization (ONNX, TensorRT)

The journey from a trained model to an optimized edge deployment involves several key stages:

- **Step 1: PyTorch/TensorFlow to ONNX (Open Neural Network Exchange):** Most deep learning models are trained using frameworks like PyTorch or TensorFlow. The initial step in the optimization pipeline is typically to export the trained model to the ONNX format. This requires careful management of parameters such as `opset_version` and handling of dynamic axes (if input sizes can vary). Ensuring that all custom operations within the model are correctly represented or handled (e.g., via `torch.onnx.export` for PyTorch models) is crucial for a successful conversion.
- **Step 2: ONNX to TensorRT Engine:** Once the model is in ONNX format, NVIDIA's TensorRT is used to convert it into a highly optimized inference engine. This can be done using `trtexec`, a command-line tool provided with TensorRT, or via the TensorRT Python API for more programmatic control. It is during this stage that TensorRT performs its key optimizations, including layer fusion (combining multiple layers into a single, more efficient operation), precision calibration (for INT8 quantization), and kernel auto-tuning (selecting the fastest available implementations for each operation on the target GPU).
- **Precision Modes:** TensorRT supports various precision modes, each offering a different trade-off between speed and accuracy:
    - **FP32 (Single Precision):** This is the baseline precision, typically matching the training precision. It offers the highest accuracy but is the slowest.
    - **FP16 (Half Precision):** Using 16-bit floating-point numbers can provide significant speedups with often minimal loss in accuracy. FP16 is well-supported on the Jetson AGX Orin and is likely the first precision mode to target after establishing an FP32 baseline.
    - **INT8 (8-bit Integer):** This mode can offer the largest speedups and memory footprint reduction. However, it requires a calibration step using a representative dataset (e.g., the FishEye8K validation set) to determine the appropriate scaling factors for weights and activations to minimize accuracy degradation. Implementing INT8 quantization correctly is more complex than FP16.
- **Workspace Size:** Sufficient GPU memory must be allocated as workspace for TensorRT during the engine building process. Insufficient workspace can lead to suboptimal engines or build failures.
- **Handling Unsupported Layers:** If TensorRT does not natively support certain layers or operations in the ONNX model, this can pose a significant challenge. Options include implementing custom TensorRT plugins (typically in C++), which requires advanced expertise, or modifying the model architecture to replace unsupported layers with supported alternatives. Another approach, though potentially detrimental to performance, is to split the model graph, running supported parts with TensorRT and unsupported parts using ONNX Runtime on the CPU or GPU. This complexity reinforces the earlier recommendation to choose models with good out-of-the-box TensorRT compatibility.

The conversion and optimization process is rarely a one-shot success. It is an iterative procedure. Teams will likely need to experiment with different ONNX export settings, various TensorRT build configurations, and multiple precision modes (FP32, FP16, INT8). Potentially, minor model modifications might be required to achieve optimal conversion. At each stage of this iteration, it is crucial to profile the FPS and measure the F1-score on the Jetson device (or a closely comparable environment). ONNX export can sometimes introduce subtle issues or inefficiencies depending on the PyTorch model's structure. TensorRT engine building can fail or produce suboptimal engines if settings like workspace size or handling of dynamic shapes are not correctly tuned. While FP16 might cause slight accuracy drops, INT8 quantization almost certainly will if the calibration process is not performed meticulously; the impact on the F1 score must be carefully measured. The actual FPS gain from FP16 or INT8 can only be definitively confirmed by profiling on the Jetson hardware. Therefore, allocating significant time for this optimization phase is essential. Maintaining rigorous version control for model checkpoints, ONNX files, and generated TensorRT engines, along with a detailed log (e.g., a spreadsheet) tracking F1 scores and FPS on the Jetson for each optimization attempt, will enable a systematic approach to finding the best F1-FPS balance.

### 4.2. Techniques for Maximizing FPS: Quantization, Pruning, and Architectural Choices

Beyond TensorRT's inherent optimization capabilities, several other techniques can be employed to maximize FPS:

- **Quantization:** As discussed, TensorRT's INT8 quantization is a key technique for speed.
- **Pruning:** Model pruning involves removing redundant parameters from the network to reduce its size and computational cost.
    - **Structured Pruning:** This method removes entire filters, channels, or even layers. It is generally more hardware-friendly and can lead to direct wall-clock speedups as it alters the model's architecture in a way that can be exploited by standard hardware.
    - **Unstructured Pruning:** This involves zeroing out individual weights, leading to sparse weight matrices. While it can achieve higher levels of sparsity, it may not always translate to significant speedups on general-purpose hardware without specialized sparse computation kernels. Pruning typically requires a fine-tuning step after weights are removed to help the model recover any lost accuracy. This adds to the overall training time. Team Nota's 2024 paper mentioned pruning as a potential future work for compressing their ensemble.10
- **Knowledge Distillation:** This powerful technique involves training a smaller, faster "student" model to mimic the behavior of a larger, more accurate "teacher" model (which could even be an ensemble of models). The student learns not only from the ground truth labels but also from the "soft labels" (output probabilities) or intermediate feature representations of the teacher. Team Nota also identified this as a promising direction.10 Knowledge distillation is a particularly relevant strategy for the 2025 challenge, as it offers a pathway to condense the high accuracy of 2024-style complex ensembles into a single, fast model suitable for edge deployment.
- **Architectural Choices (Reiteration from Section 3.1):**
    - Employ lightweight backbones (e.g., MobileNets, EfficientNets, ShuffleNets).
    - Utilize efficient detection heads (YOLO heads, for instance, are generally designed for speed).
    - Minimize or replace computationally expensive layers where possible without significantly impacting accuracy.
- **Input Resolution:** Reducing the input resolution of images fed to the model can drastically decrease computation and thus increase FPS. However, this usually comes at the cost of a lower F1-score, particularly for detecting small objects. Finding the optimal input resolution is a critical trade-off that needs careful tuning. SKKU-AutoLab experimented with various input sizes during their 2024 training.12
- **Batch Size:** For real-time inference on a single stream of images, a batch size of 1 is typical. While larger batch sizes can improve throughput on server-grade GPUs, they often introduce latency that is undesirable for real-time edge applications. TensorRT optimizes engines for specific batch sizes, so this should be configured appropriately during engine building.

Knowledge distillation stands out as a particularly promising technique to bridge the gap between the high-F1 ensembles of 2024 and the stringent FPS requirements of 2025. The top teams in 2024 relied on powerful but slow ensembles.8 Team Nota explicitly acknowledged their ensemble was not edge-practical and proposed distillation as a solution.10 A well-designed student model, even if architecturally simpler and faster, can learn the complex decision boundaries and nuanced "dark knowledge" captured by a larger teacher ensemble. If successful, the student model can achieve a substantial portion of the teacher's F1 score but with the inference speed of a single, smaller, edge-friendly model. This directly addresses the F1-FPS co-optimization challenge. Teams should seriously consider this approach if they find that their highest F1 scores are achieved with ensembles that are too slow for the Jetson target. The teacher model could be a 2024-style ensemble, and the student model an architecture known for its efficiency on Jetson devices. While potentially a high-effort strategy, the rewards in terms of the final evaluation metric could be significant.

### 4.3. Ensuring Robust Dockerized Submissions Meeting Efficiency Standards

The final submission to the challenge must be a Docker container that encapsulates the entire optimized inference framework [User Query].

- **Docker Environment:**
    - The Docker image should ideally start from an appropriate NVIDIA Jetson base image (e.g., from `nvcr.io/nvidia/l4t-pytorch` or `l4t-tensorflow`). These base images come pre-configured with CUDA, cuDNN, and TensorRT versions compatible with the Jetson Linux for Tegra (L4T) environment.
    - All necessary dependencies, including Python packages and system libraries, must be installed within the Dockerfile.
    - Crucially, the Docker image must contain the pre-trained model weights, the _final, optimized TensorRT engine file(s)_, and all Python scripts required for inference.
    - The GitHub repositories of the 2024 teams, such as Nota 11 and SKKU-AutoLab 13, provide examples of Docker setups, though these will likely need adaptation for Jetson-specific base images and the 2025 requirements.
- **Inference Script:**
    - The script within the Docker container must correctly load the pre-built TensorRT model engine.
    - It should be designed to process images one by one from the designated input directory.
    - The script must generate the output in the specified JSON format [User Query], ensuring correct `image_id` (using the provided conversion function), `category_id`, `bbox` coordinates (x1, y1, width, height), and `score`.
    - The timing for FPS calculation starts "immediately before the first image is processed and ends after the last image is fully processed" [User Query]. This implies that efficient image loading, pre-processing (if any, beyond what's baked into the model), and post-processing (e.g., Non-Maximum Suppression, NMS) are also important contributors to the overall FPS.
- **Testing the Docker Container:**
    - Thorough testing of the Docker container is essential, ideally on a Jetson AGX Orin device. If direct access is unavailable, testing on a Linux machine with Docker installed can catch some issues, but FPS measurements will not be representative.
    - The correctness of the output JSON format should be verified using the provided `eval_f1.py` script.
    - Ensure that the Docker container has no dependencies on internet connectivity during runtime and does not rely on non-standard file paths that might not exist on the evaluation server.

The Docker submission is more than just a package for code; it must be a self-contained, optimized inference engine. The container is not merely for packaging the source code; it must include the _final, pre-built, and optimized TensorRT engine_ (or ONNX model if that is the chosen deployment format). The model should not be built or converted to a TensorRT engine _inside_ the Docker container at runtime on the evaluation server. Such an on-the-fly conversion process would be far too slow (TensorRT engine generation can take minutes to hours) and would lead to unpredictable performance, likely failing the FPS requirement. The inference script within the Docker environment should simply load this pre-built engine and execute inference. Therefore, the Docker build process must include steps to copy the pre-built TensorRT engine(s) into the image. Teams need to manage these engine files carefully, especially if iterating with different precision modes (FP16, INT8) or model versions. The Dockerfile should specify a base image compatible with the Jetson's L4T version and its pre-installed NVIDIA libraries to ensure smooth execution on the evaluation hardware.

## 5. Strategic Blueprint for Success in the 2025 AI City Challenge Track 4

Synthesizing the analyses of the challenge requirements, insights from past winners, and considerations for edge deployment, this section outlines a strategic blueprint for achieving a high score in the 2025 AI City Challenge Track 4.

### 5.1. Key Recommendations for a Competitive Edge

To gain a competitive edge, participants should consider the following core recommendations:

- **Jetson First, F1 Second (Initially):** The immediate priority should be to establish a working pipeline that can run _any_ reasonable object detection model above the 10 FPS threshold on a Jetson AGX Orin (or a closely emulated environment) using TensorRT. This validates the deployment process and provides a performance baseline. Once this is achieved, efforts can shift towards iteratively improving the F1-score while constantly re-benchmarking FPS to ensure the 10 FPS minimum is maintained and the solution moves towards the 25 FPS target for optimal `FPS_normalized`.
- **Embrace Lightweight Architectures:** Favor models inherently designed or known for efficiency on edge devices. Examples include smaller variants of the YOLO series, MobileNets, or other specialized edge AI architectures. Critically evaluate all candidate models for their compatibility and performance with TensorRT.
- **Master TensorRT Optimization:** A deep understanding and proficient application of TensorRT are crucial. This includes effectively utilizing FP16 precision and, for maximum speed, INT8 quantization with meticulous calibration. Significant FPS gains are often realized through successful TensorRT optimization.
- **Strategic Data Augmentation:** Focus on data augmentation techniques that specifically address the challenges of fisheye imagery: geometric distortions, wide scale variations in objects, and potential domain shifts introduced by the new cameras in the test set. Consider advanced techniques like style transfer for varying lighting conditions if they prove beneficial and computationally feasible within the overall workflow.
- **Efficient Pseudo-Labeling:** If pseudo-labeling is incorporated to boost F1-scores, opt for methods that are efficient in terms of computational and development time. For example, using a strong teacher model for a one-shot pseudo-label generation might be more practical than lengthy iterative self-training, given the need to also iterate on edge deployment.
- **Rethink Ensembling:** The heavy, multi-model ensembles common in 2024 are unlikely to meet the 2025 FPS requirements. If ensembling is considered, it must be through extremely lightweight models or, more strategically, via knowledge distillation where the collective intelligence of an ensemble is transferred to a single, fast student model.
- **Manage Fisheye Edges Efficiently:** Implement techniques to improve detection at the distorted image peripheries, such as SAHI or distortion-aware model components. However, rigorously profile the FPS impact of any such technique, as methods like SAHI can be computationally intensive if not carefully optimized or selectively applied.
- **Thorough Dockerization and Testing:** Create a clean, minimal Docker image containing pre-built TensorRT engines. Test the submission extensively, ideally on a Jetson AGX Orin, to verify functionality, output format, and performance.

### 5.2. Development Roadmap and Prioritization

A structured development roadmap can help manage the complexity of the challenge:

- **Phase 1: Setup and Baseline (e.g., Weeks 1-2 of development period)**
    - **Objective:** Establish a functional end-to-end pipeline and secure access to testing hardware.
    - **Tasks:**
        - Prioritize acquiring or gaining access to a Jetson AGX Orin 32GB or a suitable alternative for testing and profiling.
        - Set up the complete development environment, including Docker, PyTorch/TensorFlow, CUDA, cuDNN, and TensorRT.
        - Train a simple, fast baseline object detection model (e.g., YOLOv8n or YOLOv8s) on the FishEye8K dataset.
        - Successfully convert this baseline model to a TensorRT engine (initially targeting FP16 precision).
        - Benchmark the FPS of this baseline on the Jetson device. The primary goal here is to exceed the 10 FPS disqualification threshold and understand the basic performance characteristics.
- **Phase 2: F1 Improvement and Advanced Techniques (e.g., Weeks 3-6)**
    - **Objective:** Enhance detection accuracy while monitoring FPS.
    - **Tasks:**
        - Experiment with more advanced or larger model architectures if the baseline FPS from Phase 1 provides sufficient headroom.
        - Implement and refine advanced data augmentation strategies tailored for fisheye imagery.
        - Develop and integrate a pseudo-labeling pipeline, focusing on efficiency.
        - Iteratively train models, evaluate their F1-scores on the validation set, convert them to TensorRT, and re-check FPS on the Jetson. Maintain a careful balance.
- **Phase 3: Optimization and Ensemble/Distillation Strategies (e.g., Weeks 7-9)**
    - **Objective:** Maximize the F1-FPS harmonic mean score.
    - **Tasks:**
        - Focus on INT8 quantization using TensorRT, including careful selection of a calibration dataset and methodology.
        - If pursuing ensemble-like performance, develop and train a student model via knowledge distillation from a more powerful teacher (which could be an ensemble of models developed in Phase 2).
        - Fine-tune crucial hyperparameters such as confidence thresholds for detection and Non-Maximum Suppression (NMS) settings to achieve the optimal balance for the F1-FPS metric.
- **Phase 4: Finalization and Submission (e.g., Weeks 10-12)**
    - **Objective:** Prepare a robust and compliant final submission.
    - **Tasks:**
        - Finalize the Docker container, ensuring it is clean, minimal, and includes all necessary pre-built engines and dependencies.
        - Conduct rigorous testing of the complete Dockerized submission, simulating the evaluation conditions as closely as possible.
        - Prepare the JSON submission file according to the specified format and verify its correctness.
        - Compile any required documentation or code for open-sourcing.

### 5.3. Anticipating Challenges and Mitigation Strategies

Several challenges are likely to arise during development. Proactive planning can help mitigate their impact:

- **Challenge: Consistently Low FPS on Jetson.**
    - **Mitigation:** Systematically explore options: switch to a more lightweight model architecture; apply more aggressive quantization (e.g., ensure INT8 is working optimally); implement model pruning; reduce the input image resolution (while monitoring F1 impact); optimize any custom pre-processing or post-processing code for speed.
- **Challenge: TensorRT Conversion Failures or Unsupported Operations.**
    - **Mitigation:** Attempt to simplify the model architecture by replacing problematic layers with TensorRT-supported alternatives. If certain operations are truly unsupported and unavoidable, consider implementing custom TensorRT plugins (a high-effort task requiring C++ expertise) or, as a less ideal fallback, investigate splitting the model graph to run problematic subgraphs using ONNX Runtime (though this can introduce overhead and negatively impact FPS).
- **Challenge: Significant Accuracy Drop after Quantization (especially INT8).**
    - **Mitigation:** Revisit the INT8 calibration process. Ensure the calibration dataset is sufficiently large and representative of the test data. Experiment with different calibration algorithms if available. As a more advanced and time-consuming solution, consider Quantization-Aware Training (QAT), where the model is fine-tuned with simulated quantization effects, often leading to better post-quantization accuracy.
- **Challenge: Poor Generalization to the New Cameras in the Test Set.**
    - **Mitigation:** Intensify efforts in robust data augmentation, specifically designing augmentations that can simulate a wider range of camera perspectives, lens distortions, and environmental conditions. If permissible by the rules and feasible, explore domain randomization techniques. Choose model architectures known for good generalization capabilities.
- **Challenge: Effective Time Management for a Complex, Multi-faceted Problem.**
    - **Mitigation:** Adhere to a structured development roadmap with clear phases and milestones. Prioritize tasks based on their potential impact on the final F1-FPS score (e.g., meeting the 10 FPS minimum is a non-negotiable early gate). Adopt a "fail fast" mentality for approaches that show little promise after initial investigation to avoid sinking excessive time into dead ends.

Perhaps the most significant risk, particularly for teams without dedicated hardware, is the "no Jetson access" blind spot. The challenge rules permit participation without owning a Jetson AGX Orin, yet the final evaluation is performed on this specific device [User Query]. This creates a substantial handicap. FPS is a critical component for both qualification (>=10 FPS) and the final score. The Jetson AGX Orin has unique performance characteristics, and TensorRT is finely tuned for NVIDIA's GPU architectures. Without the ability to test and profile directly on the target hardware (or a very close substitute), a team cannot be certain that their solution meets the FPS requirements or that their TensorRT engine is performing optimally. A solution that appears fast on a desktop GPU might be unexpectedly slow on the Jetson, or TensorRT conversion might exhibit subtle issues or suboptimal performance only apparent on the actual Jetson platform. Teams in this situation are strongly advised to make every conceivable effort to gain access to a Jetson AGX Orin 32GB for development and testing. If this proves absolutely impossible, the strategy must rely on:

- Thoroughly studying NVIDIA's published benchmarks for various object detection models on the Jetson AGX Orin.
- Selecting extremely conservative, lightweight model architectures that have a proven track record of running efficiently on Jetson devices.
- Being exceptionally meticulous with the TensorRT conversion process, strictly using L4T base Docker images, and hoping that performance translates as expected. This path carries a significant inherent risk.

By carefully considering these strategic elements, leveraging insights from previous challenges, and focusing intently on the new F1-FPS metric and Jetson deployment requirements, contestants can significantly enhance their prospects of achieving a high score in the 2025 AI City Challenge Track 4.