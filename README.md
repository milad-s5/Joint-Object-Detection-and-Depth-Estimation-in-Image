# Joint Object Detection and Depth Estimation in Image
Repository for the Deep Learning course project:

- Milad Samimifar: 400205577
- Pooria Ashrafian: 96101227

Object detection method that can simultaneously estimate the positions and depth of the objects from images

---

### Joint object detection and depth estimation web app using streamlit (recomended): Object_detection_web_app_streamlit repository

1.  Download yolov3 weights and put it in Object_detection_web_app_streamlit folder:

```
wget -nc https://pjreddie.com/media/files/yolov3.weights -P Object_detection_web_app_streamlit
```

2.  Download our depth model weights and put it in ./models_depth/Unet folder:

```
gdown --id 1fvzPVqKj46WjaYw6OUr1a38KblWkGC4W -O ./models_depth/Unet/2.h5 # weights

gdown --id 1g685-v1qGv6NBE7nhY0LSQDgUemJXarc -O ./models_depth/Unet/2.json # model(json file)
```

3. install requirements:

```
pip install streamlit opencv-python black
```

4.	Execute: 

```
cd Object_detection_web_app_streamlit
streamlit run ./src/app.py
```

![image](https://user-images.githubusercontent.com/82322980/152588006-5e305f46-3a49-474f-8714-e80be1a5aeb4.png)

---

### Joint object detection and depth estimation web app using django: Object_detection_web_app repository

1.  Download yolov3 weights and put it in Object_detection_web_app folder:

```
wget -nc https://pjreddie.com/media/files/yolov3.weights -P Object_detection_web_app
```

2.	Execute the code below: (Only once) 

```
cd Object_detection_web_app
python manage.py collectstatic
```

3.	Execute: 

```
python manage.py runserver
```

![image](https://user-images.githubusercontent.com/82322980/152589707-0c5f9c29-2b75-4499-87c3-d28a96b3e070.png)

---

### Joint object detection and depth estimation: demo

Joint_notebook.ipynb:

* Download dataset and preprocessing
* Train Unet model for depth estimation
* Yolov3 Object detection
* Joint model and test

---
