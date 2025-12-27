### Phase 1: Enrich data and train simple models 
- **Objective**: Understand task by working in small architect
- Similar dataset: [link](https://universe.roboflow.com/master-final-dataset-curation-v1a/cctv-curation-dataset-1-hhibk/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true)
	- Does it contain similar labels 
	- EDA
	- Convert to fisheye images [link](https://github.com/vnptai/AICITY2024_Track4/blob/main/dataprocessing/ifish_augmentation/ifisheye.py)
- Pseudo-Labeling
- Model training 
	- Yolov11n
- Convert the baseline model to a TensorRT engine (initially targeting FP16 precision).
- Ensembling
- Benchmark FPS
### Phase 2: F1 Improvement and advanced techniques
- **Objective:** Enhance detection accuracy while monitoring FPS.
-  **Tasks:**
	- Experiment with more advanced architectures 
	- Fine tune model for fish eye task
	- Develop and integrate a pseudo-labeling pipeline, focusing on efficiency.
	- Convert them to TensorRT, and re-check FPS on the Jetson.
### Phase 3: Optimization and Ensemble/Distillation Strategies 
- **Objective:** Maximize the F1-FPS harmonic mean score.
- **Tasks:**
	- Focus on INT8 quantization using TensorRT
	- Fine-tune crucial hyperparameters (confidence thresholds for detection and Non-Maximum Suppression (NMS) settings) to achieve the optimal balance for the F1-FPS metric.
### Phase 4: Finalization and Submission 
- **Objective:** Prepare a robust and compliant final submission.
- **Tasks:**
	- Docker implementation, clean code
	- Prepare the JSON submission file
	- Write document and other required docs
![[Pasted image 20250517083312.png]]
[FisheyeYOLO](https://www.researchgate.net/publication/346931586_FisheyeYOLO_Object_Detection_on_Fisheye_Cameras_for_Autonomous_Driving)
[YOLOv12: Redefining Real-Time Object Detection](https://henrynavarro.org/yolov12-redefining-real-time-object-detection-4dd49c293d19)


