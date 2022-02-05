# Joint Object Detection and Depth Estimation in Image
Repository for the Deep Learning course project:

- Milad Samimifar: 400205577
- Pooria Ashrafian: 96101227

Object detection method that can simultaneously estimate the positions and depth of the objects from images. (Based on NYU-Depth V2 dataset)

---

### Joint object detection and depth estimation web app using streamlit (recomended): Object_detection_web_app_streamlit repository

1.  Download yolov3 weights and put it in Object_detection_web_app_streamlit folder:

```
wget -nc https://pjreddie.com/media/files/yolov3.weights -P Object_detection_web_app_streamlit
```

2.  Download our Unet depth model weights and put it in ./models_depth/Unet folder:

Weighs: https://drive.google.com/file/d/1fvzPVqKj46WjaYw6OUr1a38KblWkGC4W/view?usp=sharing

Model (Json file): https://drive.google.com/file/d/1g685-v1qGv6NBE7nhY0LSQDgUemJXarc/view?usp=sharing

Shell commmand:

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

![image](https://user-images.githubusercontent.com/82322980/152640791-96167abe-038a-49e0-a4d7-189d90908686.png)

---

### Joint object detection and depth estimation web app using django: Object_detection_web_app repository

1.  Download yolov3 weights and put it in Object_detection_web_app folder:

```
wget -nc https://pjreddie.com/media/files/yolov3.weights -P Object_detection_web_app
```

2.  Install requirements:

```
pip install -r ./Object_detection_web_app/requirements.txt
```

3.	Execute the code below: (Only once) 

```
cd Object_detection_web_app
python manage.py collectstatic
```

4.	Execute: 

```
python manage.py runserver
```

![image](https://user-images.githubusercontent.com/82322980/152633618-3ca2c6a7-f931-41a9-9089-c5d18d32d937.png)

---

### Joint object detection and depth estimation: demo

Joint_notebook.ipynb:

* Download dataset and preprocessing
* Train Unet model for depth estimation

![image](https://user-images.githubusercontent.com/82322980/152643472-a6e9a285-3fd0-4d47-9efe-e2299a5858ae.png)

* Yolov3 Object detection

![image](https://user-images.githubusercontent.com/82322980/152643493-9863a272-01b7-4cc1-a165-400b547c3a0f.png)

* Joint model and test

![image](https://user-images.githubusercontent.com/82322980/152643505-78c7084d-df29-4f39-83d6-e2511b14a96d.png)

---
