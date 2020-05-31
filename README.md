# mrcnnModuleService
A module build on the  matterport/Mask_RCNN Repository providing the use of the neural network module in a separate Process for easy Multiprocessing Implementation.

## Imports required:
Mostly I used the same build as like the MRCNN Repository and the Tutorial on Training your own neural network(see below).
- **MRCNN Repository**: https://github.com/matterport/Mask_RCNN (Download the **mrcnn** directory and copy it in your Project directory)

### other Dependencies:
- **tensorflow 1.14**
- **keras 2.2.4**
- **numpy**
- **scipy**
- **Pillow**
- **cython**
- **matplotlib**
- **scikit-image**
- **opencv-python**
- **h5py**
- **imgaug**
- **IPython[all]**

## how to use:
- Make shure you have the Imports/Dependencies above
- Copy the python file **mrcnnModuleService** in the same directory as like the **mrcnn** directory
- Import in your Code: from mrcnnModuleService import NeuralNetService
- Create a new network Configuration(example found in **mrcnnModuleService** I used in a different Project), should be similar to the Configuration your network was trained with
- Create a new NeuralNetService-Object, as params you give the Configuration and the absolute Path to the loading weights
- Start the Process using the **start_service()** Method
- Add a prediction task with **add_prediction_task(imgList)**
- Get the corresponding Predictions with **get_preditcion_task()**
- Shut down the Service if you do not need it anymore with **shut_down_service()**
- To restart the Service you need to initiate it first: **init_service(config, path)** and **start_service()**
- look at the Description in the Methods for more info

## other interesting Stuff:
- Tutorial on Training your own Mask_RCNN Network: https://github.com/akTwelve/cocosynth/blob/master/notebooks/train_mask_rcnn.ipynb
