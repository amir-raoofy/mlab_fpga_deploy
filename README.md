# FPGA Deep Learning Benchmark

This repo consists of baselines for different ML applications to be deployed on FPGAs using Vitis AI:

1 - Classification baseline (earthvision)

- Classification baseline models in [00_model_training](00_model_training), which depends on BigEarthNet available from BGDM cluster nodes (rsync, more automatic access is done soon)
- Classification deployed on Xilinx FPGAs models in [01_end_to_end_training_to_deployment_classification](00_model01_end_to_end_training_to_deployment_classification_training). We rely on this end-to-end deployment workflow taken from [Xilinx Vitis Tutorials] (https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning/Design_Tutorials/02-MNIST_classification_tf). It is still not fully functional because of the issues of the compatability of model format (see this [link](https://forums.xilinx.com/t5/AI-and-Vitis-AI/Quantizing-TensorFlow-Hub-Models-Transfer-Learning-Workflow/m-p/1210723)).

2 - Segmentation baseline (ship detection)
- Kaggle ship detection baseline [02_ship_detection_kaggle_reference_notebooks](02_ship_detection_kaggle_reference_notebooks), taken from this [link](https://www.kaggle.com/awater1223/unet-resnet34-for-ships ), and depends on the Airbus Ship Detection dataset in [here](https://www.kaggle.com/c/airbus-ship-detection/data)
- Deployment of the ship detection model on Xilinx FPGAs [03_end_to_end_ship_detection](03_end_to_end_ship_detection). We rely on this end-to-end deployment workflow taken from [Xilinx Vitis Tutorials] (https://github.com/Xilinx/Vitis-Tutorials/tree/master/Machine_Learning/Design_Tutorials/05-Keras_FCN8_UNET_segmentation)