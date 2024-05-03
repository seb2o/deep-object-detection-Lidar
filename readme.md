Object detection on lidar images. dataset undisclosed. models are yolo and retina. best performing can be found at yolo/s_1024_square_nomos


## Running the models

### Preprocess
- Download the dataset NAPLab-LiDAR/
- put the dataset absolute path in the corresponding variable in ````main()```` of ````preprocessing.py````
- run ````preproccessing.py````  ( it creates a directory "splitted" at the dataset location and put preproccessed 
labels into train, validation and test sets )

### Yolo
- edit the dataset path in ```yolo/data.yaml``` to match where the ```splitted``` directory is
- run the ````run.py```` file
  - when prompted, enter a name for the model 
- to generate predictions, use ````yolo/load.ipynb````, 
specifying the previously choosen model name into the ```chosen_model``` variable 
( they will be created at the model folder, at ```<model name>/test/predict```)
- to compute metrics on a test set for which labels are availables, run the `````val`````
cell of `````load.ipynb`````, specifying the path of the test dataset in the ```val``` clause of the ```yolo/test.yaml```
- the model settings are in ````yolo/run.py````
### Retina
 - ```cd retina/ && python retinanet_train.py```
   - saves the model weights at ````/retinanet_<epochs>pt````
 - ```python retinanet_test.py``` will print ````mAP@.5:.95```` to the console
 - The model settings are located at the top of each python file 