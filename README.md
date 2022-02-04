# Joint Object Detection and Depth Estimation in Image
Repository for the Deep Learning course project:

- Milad Samimifar
- Pooria Ashrafian

Object detection method that can simultaneously estimate the positions and depth of the objects from images

---

### Joint object detection and depth estimation web app:

1.  Download yolov3 weights and put it in Object_detection_web_app folder:

```
wget -nc https://pjreddie.com/media/files/yolov3.weights -P Object_detection_web_app
```

2.  Download our depth model weights and put it in ./models_depth/Unet folder:

```
gdown --id 1fvzPVqKj46WjaYw6OUr1a38KblWkGC4W -O ./models_depth/Unet/2.h5 # weights

gdown --id 1g685-v1qGv6NBE7nhY0LSQDgUemJXarc -O ./models_depth/Unet/2.json # model(json file)
```

3.	Execute the code below: (Only once) 

```
python manage.py collectstatic
```

4.	Execute: 

```
python manage.py runserver
```

---

### Joint object detection and depth estimation: demo

Joint_notebook.ipynb:

* Download dataset and preprocessing
* Train Unet model for depth estimation
* Yolov3 Object detection
* Joint model and test
