# Chapter 11

[Source - Click here for lab steps](https://github.com/dangolightly/tensorflow_object_counting_api)  
# TensorFlow Object Counting API
The TensorFlow Object Counting API is an open source framework built on top of TensorFlow that makes it easy to develop object counting systems.

## QUICK DEMO

---
### Cumulative Counting Mode:
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166455-45964aac-8f9f-11e8-9ddf-f71d05f0c7f5.gif" | width=430> <img src="https://user-images.githubusercontent.com/22610163/43166945-c0744de0-8fa0-11e8-8985-9f863c59e859.gif" | width=411>
</p>

---
### Real-Time Counting Mode:
<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42237325-1f964e82-7f06-11e8-966b-dfde98701c66.gif" | width=430> <img src="https://user-images.githubusercontent.com/22610163/42238435-77ac0d34-7f09-11e8-9609-e7c3c2c5af74.gif" | width=430>
</p>

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42241094-14163cc8-7f12-11e8-83ed-68021b5e3b33.gif" | width=430><img src="https://user-images.githubusercontent.com/22610163/42237904-d6a3ac22-7f07-11e8-88f8-5f21430d9503.gif" | width=430>
</p>

---
### Object Counting On Single Image:
<p align="center">
<img src="https://user-images.githubusercontent.com/22610163/47524870-7c830e80-d8a4-11e8-8fd1-741193615a04.png" | width=750></p>

---

***The development is on progress! The API will be updated soon, the more talented and light-weight API will be available in this repo!***

- ***Detailed API documentation and sample jupyter notebooks that explain basic usages of API will be added!***

**You can find a sample project - case study that uses TensorFlow Object Counting API in [*this repo*](https://github.com/ahmetozlu/vehicle_counting_tensorflow).**

---

## USAGE

### 1.) Usage of "Cumulative Counting Mode"

#### 1.1) For detecting, tracking and counting *the pedestrians* with disabled color prediction

*Usage of "Cumulative Counting Mode" for the "pedestrian counting" case:*

    fps = 30 # change it with your input video fps
    width = 626 # change it with your input video width
    height = 360 # change it with your input vide height
    is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    roi = 385 # roi line position
    deviation = 1 # the constant that represents the object counting area

    object_counting_api.cumulative_object_counting_x_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation) # counting all the objects
    
*Result of the "pedestrian counting" case:*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166945-c0744de0-8fa0-11e8-8985-9f863c59e859.gif" | width=700>
</p>

---

**Source code of "pedestrian counting case-study": [pedestrian_counting.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/pedestrian_counting.py)**

---

**1.2)** For detecting, tracking and counting *the vehicles* with enabled color prediction

*Usage of "Cumulative Counting Mode" for the "vehicle counting" case:*

    fps = 24 # change it with your input video fps
    width = 640 # change it with your input video width
    height = 352 # change it with your input vide height
    is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    roi = 200 # roi line position
    deviation = 3 # the constant that represents the object counting area

    object_counting_api.cumulative_object_counting_y_axis(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height, roi, deviation) # counting all the objects
    
*Result of the "vehicle counting" case:*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/43166455-45964aac-8f9f-11e8-9ddf-f71d05f0c7f5.gif" | width=700>
</p>

---

**Source code of "vehicle counting case-study": [vehicle_counting.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/vehicle_counting.py)**

---

### 2.) Usage of "Real-Time Counting Mode"

#### 2.1) For detecting, tracking and counting the *targeted object/s* with disabled color prediction
 
 *Usage of "the targeted object is bicycle":*
 
    is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    targeted_objects = "bicycle"
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
    
 *Result of "the targeted object is bicycle":*
 
 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411751-1ae1d3f0-820a-11e8-8465-9ec9b44d4fe7.gif" | width=700>
</p>

*Usage of "the targeted object is person":*

    is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    targeted_objects = "person"
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.targeted_object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, targeted_objects, fps, width, height) # targeted objects counting
 
 *Result of "the targeted object is person":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411749-1a80362c-820a-11e8-864e-acdeed85b1f2.gif" | width=700>
</p>

*Usage of "detecting, counting and tracking all the objects":*

    is_color_prediction_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
 
 *Result of "detecting, counting and tracking all the objects":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411750-1aae0d72-820a-11e8-8726-4b57480f4cb8.gif" | width=700>
</p>
 
 
#### 1.2) For detecting, tracking and counting "all the objects with disabled color prediction"

*Usage of detecting, counting and tracking "all the objects with disabled color prediction":*
    
    is_color_prediction_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
    
 *Result of detecting, counting and tracking "all the objects with disabled color prediction":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411748-1a5ab49c-820a-11e8-8648-d78ffa08c28c.gif" | width=700>
</p>


*Usage of detecting, counting and tracking "all the objects with enabled color prediction":*

    is_color_prediction_enabled = 1 # set it to 1 for enabling the color prediction for the detected objects
    fps = 24 # change it with your input video fps
    width = 854 # change it with your input video width
    height = 480 # change it with your input vide height    

    object_counting_api.object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, fps, width, height) # counting all the objects
    
 *Result of detecting, counting and tracking "all the objects with enabled color prediction":*

 <p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/42411747-1a215e4a-820a-11e8-8aef-faa500df6836.gif" | width=700>
</p>

---

**For sample usages of "Real-Time Counting Mode": [real_time_counting.py](https://github.com/ahmetozlu/tensorflow_object_counting_api/blob/master/real_time_counting.py)**

---

## General Capabilities of The TensorFlow Object Counting API

<p align="center">
  <img src="https://user-images.githubusercontent.com/22610163/48421361-6662c280-e76d-11e8-9680-ec86e245fdac.jpg" | width = 720>
</p>

Here are some cool capabilities of TensorFlow Object Counting API:

- Detect just the targeted objects
- Detect all the objects
- Count just the targeted objects
- Count all the objects
- Predict color of the targeted objects
- Predict color of all the objects
- Predict speed of the targeted objects
- Predict speed of all the objects
- Print out the detection-counting result in a .csv file as an analysis report
- Save and store detected objects as new images under [detected_object folder](www)
- Select, download and use state of the art [models that are trained by Google Brain Team](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
- Use [your own trained models](https://www.tensorflow.org/guide/keras) or [a fine-tuned model](https://github.com/Hvass-Labs/TensorFlow-Tutorials/blob/master/10_Fine-Tuning.ipynb) to detect spesific object/s
- Save detection and counting results as a new video or show detection and counting results in real time
- Process images or videos depending on your requirements

Here are some cool architectural design features of TensorFlow Object Counting API:
