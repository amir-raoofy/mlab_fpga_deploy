#!/bin/bash

##############################################
export IMG_HW="768"
export WIDTH="224"
export HEIGHT="224"
export BATCH_SIZE="32"
export EPOCHS="50"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="50"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="resnet34"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="20"

. ./run_unet.sh
##############################################
export IMG_HW="768"
export WIDTH="224"
export HEIGHT="224"
export BATCH_SIZE="32"
export EPOCHS="50"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="50"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="resnet34"
export WEIGHT_BIT="4"
export ACTIVATION_BIT="4"
export CALIB_ITERS="20"

. ./run_unet.sh
##############################################
export IMG_HW="768"
export WIDTH="384"
export HEIGHT="384"
export BATCH_SIZE="8"
export EPOCHS="50"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="50"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="resnet34"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="20"

. ./run_unet.sh
##############################################
export IMG_HW="768"
export WIDTH="384"
export HEIGHT="384"
export BATCH_SIZE="8"
export EPOCHS="50"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="50"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="resnet34"
export WEIGHT_BIT="4"
export ACTIVATION_BIT="4"
export CALIB_ITERS="20"

. ./run_unet.sh
##############################################