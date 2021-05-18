#!/bin/bash

# Vitis AI should theoretically also support quantization on a 1-bit basis, but in practice it doesn't work yet:
# limitations for reduced precisoin:
# https://forums.xilinx.com/t5/AI-and-Vitis-AI/Vitis-AI-Quantizer-weight-bit-and-activation-bit-values/td-p/1099831
# We also plan to look at Brevitas and FINN for use with 1-bit based models. I believe hls4ml also supports 1-bit based models this through a framework called QKeras.

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="mobilenetv2"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="mobilenetv2"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="inceptionresnetv2"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="densenet169"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="vgg19"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################

##############################################
export IMG_HW="768"
export WIDTH="64"
export HEIGHT="64"
export BATCH_SIZE="32"
export EPOCHS="10"
export STEPS_PER_EPOCH="200"
export VAL_STEPS="20"
export ENABLE_FINE_TUNNING="False"
export BACKBONE="mobilenet"
export WEIGHT_BIT="8"
export ACTIVATION_BIT="8"
export CALIB_ITERS="10"

. ./run_unet.sh
##############################################