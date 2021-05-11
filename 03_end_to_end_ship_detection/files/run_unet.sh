#!/bin/bash

CNN=unet

#export IMG_HW="768"
#export WIDTH="224"
#export HEIGHT="224"
#export BATCH_SIZE="32"
#export EPOCHS="1"
#export STEPS_PER_EPOCH="5"
#export VAL_STEPS="2"
#export ENABLE_FINE_TUNNING="False"
#export BACKBONE="resnet34"
#export WEIGHT_BIT="8"
#export ACTIVATION_BIT="8"
#export CALIB_ITERS="2"

export GIF_1="graph_input_fn.calib_input1"
export GIF_2="graph_input_fn.calib_input2"
export GIF_3="graph_input_fn.calib_input3"
export GIF_4="graph_input_fn.calib_input4"

#export GIF_1="graph_input_fn.calib_input_dbg"
#export GIF_2="graph_input_fn.calib_input_dbg"
#export GIF_3="graph_input_fn.calib_input_dbg"
#export GIF_4="graph_input_fn.calib_input_dbg4"

export TF_FORCE_GPU_ALLOW_GROWTH="true"
export CUDA_DEVICE_ORDER="PCI_BUS_ID"
#export CUDA_VISIBLE_DEVICES="1"

# folders
ARCHIVE_DIR=./archive
BUILD_DIR=./build
HOME_DIR=${BUILD_DIR}/..
LOG_DIR=${BUILD_DIR}/../log
RPT_DIR=${BUILD_DIR}/../rpt
TARGET_190=${BUILD_DIR}/../target_vck190
TARGET_102=${BUILD_DIR}/../target_zcu102
TARGET_104=${BUILD_DIR}/../target_zcu104
KERAS_MODEL_DIR=${BUILD_DIR}/../keras_model
DATASET_DIR=${BUILD_DIR}/dataset
RAW_DATASET_DIR=./dataset

TB_LOG_DIR=${BUILD_DIR}/tb_log
CHKPT_DIR=${BUILD_DIR}/tf_chkpts
FREEZE_DIR=${BUILD_DIR}/freeze
COMPILE_DIR=${BUILD_DIR}/compile
QUANT_DIR=${BUILD_DIR}/quantize_results

# checkpoints & graphs filenames
CHKPT_FILENAME=float_model.ckpt
META_GRAPH_FILENAME=${CHKPT_FILENAME}.meta
FROZEN_GRAPH_FILENAME=frozen_graph.pb
QUANTIZED_FILENAME=quantize_eval_model.pb

# logs & results files
PREPARE_DATA_LOG=${CNN}_prepare_data.log
TRAIN_LOG=${CNN}_training.log
FREEZE_LOG=${CNN}_freeze_graph.log
EVAL_FR_LOG=${CNN}_evaluate_frozen_graph.log
QUANT_LOG=${CNN}_quantize.log
EVAL_Q_LOG=${CNN}_evaluate_quantized_graph.log
COMP_LOG=${CNN}_compile.log

# CNN parameters
INPUT_NODE="input_1"
#OUTPUT_NODE="conv2d_18/Relu" # output node of floating point CNN UNET v1 and v3
OUTPUT_NODE="conv2d_19/Sigmoid" # output node of floating point CNN UNET v1 and v3


##################################################################################
#setup the environment and check DNNDK relese
#source ${HOME}/scripts/activate_py36_dnndk3v1.sh

##################################################################################
0_prep_data() {
    cp -r /workspace/dataset1/* ${DATASET_DIR}
}

##################################################################################
1_generate_images() {
    echo " "
    echo "##################################################################################"
    echo "Step1: CREATE DATA AND FOLDERS"
    echo "##################################################################################"
    echo " "
    # clean files in pre-built sub-directories
    rm -f ${DATASET_DIR}/img_*/* ${DATASET_DIR}/seg_*/*
    # unzip the original dataset
    #unzip ${BUILD_DIR}/../dataset.zip -d ${BUILD_DIR}
    cd code
    # put the data into proper folders
    python prepare_data.py
    cd ..
    # clean previous directories
    #rm -rf ${DATASET_DIR}/annotations_* ${DATASET_DIR}/images_*
}

##################################################################################
# effective training
2_unet_train() {
    cd code
    # effective training and predictions
    echo " "
    echo "##################################################################################"
    echo "Step2a: TRAINING"
    echo "##################################################################################"
    echo " "
    python unet_training.py -m 1
    python unet_training.py -m 2
    python unet_training.py -m 3
    python unet_training.py -m 4
    echo " "

    #cd ../code
    #echo "##################################################################################"
    #echo "Step2b: MAKING PREDICTIONS"
    #echo "##################################################################################"
    #echo " "
    #python unet_make_predictions.py -m 1
    #python unet_make_predictions.py -m 2
    #python unet_make_predictions.py -m 3
    #cd ..

}

##################################################################################
# Keras to TF chkpt files
3_unet_Keras2TF() {
    echo " "
    echo "#######################################################################################"
    echo "Step3: KERAS to TENSORFLOW GRAPH CONVERSION"
    echo "#######################################################################################"
    echo " "
    # clean TF Check Point files
    #rm ${CHKPT_DIR}/${CNN}/*
    # from Keras to TF
    cd code
    python Keras2TF.py --model  "unet1"
    python Keras2TF.py --model  "unet2"
    python Keras2TF.py --model  "unet3"
    python Keras2TF.py --model  "unet4"
    cd ..
}



##################################################################################
# freeze the inference graph
4a_unet_freeze() {
    echo " "
    echo "##############################################################################"
    echo "Step4a: FREEZE TF GRAPHS"
    echo "##############################################################################"
    echo " "
    
    # freeze the TF graph
    freeze_graph \
  --input_meta_graph  ${CHKPT_DIR}/${CNN}1/${META_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}1/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}

    # freeze the TF graph
    freeze_graph \
  --input_meta_graph  ${CHKPT_DIR}/${CNN}2/${META_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}2/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	--output_node_names "conv2d_23/Sigmoid"

    # freeze the TF graph
    freeze_graph \
  --input_meta_graph  ${CHKPT_DIR}/${CNN}3/${META_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}3/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME} \
	--output_node_names ${OUTPUT_NODE}

    freeze_graph \
  --input_meta_graph  ${CHKPT_DIR}/${CNN}4/${META_GRAPH_FILENAME} \
	--input_checkpoint  ${CHKPT_DIR}/${CNN}4/${CHKPT_FILENAME} \
	--input_binary      true \
	--output_graph      ${FREEZE_DIR}/${CNN}4/${FROZEN_GRAPH_FILENAME} \
	--output_node_names "sigmoid/Sigmoid"


    echo " "
    echo "##############################################################################"
    echo "Step4a: INSPECT FROZEN GRAPH"
    echo "##############################################################################"
    echo " "
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME}
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME}
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME}
    vai_q_tensorflow inspect --input_frozen_graph ${FREEZE_DIR}/${CNN}4/${FROZEN_GRAPH_FILENAME}
}


##################################################################################
# evaluate the original graph
4b_eval_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step4b: EVALUATING THE ORIGINAL GRAPH"
    echo "##############################################################################"
    echo " "
    cd code

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=1 \
     --model=1

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node="conv2d_23/Sigmoid" \
	   --gpu=1 \
     --model=2

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=1 \
     --model=3

    python eval_graph.py \
	   --graph=../${FREEZE_DIR}/${CNN}4/${FROZEN_GRAPH_FILENAME} \
	   --input_node="data" \
	   --output_node="sigmoid/Sigmoid" \
	   --gpu=1 \
     --model=4
    
    cd ..
}


##################################################################################
5a_unet_quantize() {
    echo " "
    echo "##########################################################################"
    echo "Step5a: QUANTIZATION"
    echo "##########################################################################"
    echo " "
    #quantize
    cd code

    vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}1/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        ${OUTPUT_NODE} \
	 --output_dir          ../${QUANT_DIR}/${CNN}1/ \
	 --method              1 \
	 --input_fn            ${GIF_1} \
	 --calib_iter          ${CALIB_ITERS} \
	 --gpu                 1 \
   --activation_bit      ${ACTIVATION_BIT} \
   --weight_bit         ${WEIGHT_BIT}

   vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}2/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        "conv2d_23/Sigmoid" \
	 --output_dir          ../${QUANT_DIR}/${CNN}2/ \
	 --method              1 \
	 --input_fn            ${GIF_2} \
	 --calib_iter          ${CALIB_ITERS} \
	 --gpu                 1 \
   --activation_bit      ${ACTIVATION_BIT} \
   --weight_bit         ${WEIGHT_BIT}

   vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}3/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         ${INPUT_NODE} \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        ${OUTPUT_NODE} \
	 --output_dir          ../${QUANT_DIR}/${CNN}3/ \
	 --method              1 \
	 --input_fn            ${GIF_3} \
	 --calib_iter          ${CALIB_ITERS} \
	 --gpu                 1 \
   --activation_bit      ${ACTIVATION_BIT} \
   --weight_bit         ${WEIGHT_BIT}

    vai_q_tensorflow quantize \
	 --input_frozen_graph  ../${FREEZE_DIR}/${CNN}4/${FROZEN_GRAPH_FILENAME} \
	 --input_nodes         "data" \
	 --input_shapes        ?,224,224,3 \
	 --output_nodes        "sigmoid/Sigmoid" \
	 --output_dir          ../${QUANT_DIR}/${CNN}4/ \
	 --method              1 \
	 --input_fn            ${GIF_4} \
	 --calib_iter          ${CALIB_ITERS} \
	 --gpu                 1 \
   --activation_bit      ${ACTIVATION_BIT} \
   --weight_bit         ${WEIGHT_BIT}

    cd ..
}

##################################################################################
# make predictions with quantized graph

5b_eval_quantized_graph() {
    echo " "
    echo "##############################################################################"
    echo "Step5b: EVALUATE QUANTIZED GRAPH"
    echo "##############################################################################"
    echo " "
    cd code

    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}1/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0 \
     --model=1

    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}2/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node="conv2d_23/Sigmoid" \
	   --gpu=0 \
     --model=2

    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}3/${QUANTIZED_FILENAME} \
	   --input_node=${INPUT_NODE} \
	   --output_node=${OUTPUT_NODE} \
	   --gpu=0 \
     --model=3


    python eval_quantized_graph.py \
	   --graph=../${QUANT_DIR}/${CNN}4/${QUANTIZED_FILENAME} \
	   --input_node="data" \
	   --output_node="sigmoid/Sigmoid" \
	   --gpu=1 \
     --model=4


    cd ..
}


##################################################################################
# Compile ELF file for VCK190 with Vitis AI
6_compile_vai_vck190() {
  echo " "
  echo "##########################################################################"
  echo "COMPILE UNET XMODEL FILE WITH Vitis AI for VCK190 TARGET"
  echo "##########################################################################"
  echo " "

  vai_c_tensorflow \
      --frozen_pb ${QUANT_DIR}/${CNN}1/quantize_eval_model.pb \
      --arch /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN}1 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}1

  vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}2/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN}2 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}2

  vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}3/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN}3 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}3

  vai_c_tensorflow \
	 --frozen_pb ${QUANT_DIR}/${CNN}4/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCVDX8G/VCK190/arch.json \
	 --output_dir ${COMPILE_DIR}/${CNN}4 \
	 --options    "{'mode':'normal'}" \
	 --net_name ${CNN}4
 }

 6_compile_vai_zcu102() {
   echo " "
   echo "##########################################################################"
   echo "COMPILE UNET XMODEL FILE WITH Vitis AI for ZCU102 TARGET"
   echo "##########################################################################"
   echo " "

   vai_c_tensorflow \
       --frozen_pb ${QUANT_DIR}/${CNN}1/quantize_eval_model.pb \
       --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
 	 --output_dir ${COMPILE_DIR}/${CNN}1 \
 	 --options    "{'mode':'normal'}" \
 	 --net_name ${CNN}1

   vai_c_tensorflow \
 	 --frozen_pb ${QUANT_DIR}/${CNN}2/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
 	 --output_dir ${COMPILE_DIR}/${CNN}2 \
 	 --options    "{'mode':'normal'}" \
 	 --net_name ${CNN}2

   vai_c_tensorflow \
 	 --frozen_pb ${QUANT_DIR}/${CNN}3/quantize_eval_model.pb \
   --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
 	 --output_dir ${COMPILE_DIR}/${CNN}3 \
 	 --options    "{'mode':'normal'}" \
 	 --net_name ${CNN}3

   vai_c_tensorflow \
 	 --frozen_pb ${QUANT_DIR}/${CNN}4/quantize_eval_model.pb \
     --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU102/arch.json \
 	 --output_dir ${COMPILE_DIR}/${CNN}4 \
 	 --options    "{'mode':'normal'}" \
 	 --net_name ${CNN}4
  }

  6_compile_vai_zcu104() {
    echo " "
    echo "##########################################################################"
    echo "COMPILE UNET XMODEL FILE WITH Vitis AI for ZCU104 TARGET"
    echo "##########################################################################"
    echo " "

    vai_c_tensorflow \
        --frozen_pb ${QUANT_DIR}/${CNN}1/quantize_eval_model.pb \
        --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  	 --output_dir ${COMPILE_DIR}/${CNN}1 \
  	 --options    "{'mode':'normal'}" \
  	 --net_name ${CNN}1

    vai_c_tensorflow \
  	 --frozen_pb ${QUANT_DIR}/${CNN}2/quantize_eval_model.pb \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  	 --output_dir ${COMPILE_DIR}/${CNN}2 \
  	 --options    "{'mode':'normal'}" \
  	 --net_name ${CNN}2

    vai_c_tensorflow \
  	 --frozen_pb ${QUANT_DIR}/${CNN}3/quantize_eval_model.pb \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  	 --output_dir ${COMPILE_DIR}/${CNN}3 \
  	 --options    "{'mode':'normal'}" \
  	 --net_name ${CNN}3

    vai_c_tensorflow \
  	 --frozen_pb ${QUANT_DIR}/${CNN}4/quantize_eval_model.pb \
    --arch /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json \
  	 --output_dir ${COMPILE_DIR}/${CNN}4 \
  	 --options    "{'mode':'normal'}" \
  	 --net_name ${CNN}4

   }

##################################################################################
##################################################################################

main() {

  conda activate vitis-ai-tensorflow

  # assuming you have run first the run_fcn8.sh script, you do not need to clean up anything

    # clean up previous results
    rm -rf ${BUILD_DIR}; mkdir ${BUILD_DIR}
    rm -rf ${LOG_DIR}; mkdir ${LOG_DIR}
    rm -rf ${RPT_DIR}; mkdir ${RPT_DIR}
    rm -rf ${CHKPT_DIR}; mkdir ${CHKPT_DIR}
    rm -rf ${KERAS_MODEL_DIR}; mkdir ${KERAS_MODEL_DIR}
    rm -rf ${KERAS_MODEL_DIR}/${CNN}; mkdir ${KERAS_MODEL_DIR}/${CNN}
    rm -rf ${DATASET_DIR}; mkdir ${DATASET_DIR}

    rm -rf ${FREEZE_DIR}; mkdir ${FREEZE_DIR}
    rm -rf ${QUANT_DIR}; mkdir ${QUANT_DIR}
    rm -rf ${COMPILE_DIR}; mkdir ${COMPILE_DIR}
    
    rm -rf ${TARGET_190}/*.tar.gz ${TARGET_102}/*.tar.gz ${TARGET_104}/*.tar.gz
    rm -rf ${TARGET_190}/rpt_deploy/ ${TARGET_102}/rpt_deploy/ ${TARGET_104}/rpt_deploy/
    rm -rf ${TARGET_190}/log_deploy ${TARGET_102}/log_deploy ${TARGET_104}/log_deploy
    rm -rf *.tar.gz *.tar

    mkdir ${LOG_DIR}/${CNN}

    rm -rf ${CHKPT_DIR}/${CNN}1   ${CHKPT_DIR}/${CNN}2   ${CHKPT_DIR}/${CNN}3
    rm -rf ${FREEZE_DIR}/${CNN}1  ${FREEZE_DIR}/${CNN}2  ${FREEZE_DIR}/${CNN}3
    rm -rf ${QUANT_DIR}/${CNN}1   ${QUANT_DIR}/${CNN}2   ${QUANT_DIR}/${CNN}3
    rm -rf ${COMPILE_DIR}/${CNN}1 ${COMPILE_DIR}/${CNN}2 ${COMPILE_DIR}/${CNN}3
    mkdir ${CHKPT_DIR}/${CNN}1   ${CHKPT_DIR}/${CNN}2   ${CHKPT_DIR}/${CNN}3 ${CHKPT_DIR}/${CNN}4  
    mkdir ${FREEZE_DIR}/${CNN}1  ${FREEZE_DIR}/${CNN}2  ${FREEZE_DIR}/${CNN}3 ${FREEZE_DIR}/${CNN}4 
    mkdir ${QUANT_DIR}/${CNN}1   ${QUANT_DIR}/${CNN}2   ${QUANT_DIR}/${CNN}3 ${QUANT_DIR}/${CNN}4  
    mkdir ${COMPILE_DIR}/${CNN}1 ${COMPILE_DIR}/${CNN}2 ${COMPILE_DIR}/${CNN}3 ${COMPILE_DIR}/${CNN}4
    
    #0_prep_data
    ## create the proper folders and images from the original dataset
    1_generate_images 2>&1 | tee ${LOG_DIR}/${CNN}/${PREPARE_DATA_LOG}

    # do the training and make predictions
    2_unet_train     2>&1 | tee ${LOG_DIR}/${CNN}/${TRAIN_LOG}

    # from Keras to TF
    3_unet_Keras2TF  2>&1 | tee ${LOG_DIR}/${CNN}/unet_keras2tf.log

    # freeze the graph and inspect it
    4a_unet_freeze   2>&1 | tee ${LOG_DIR}/${CNN}/${FREEZE_LOG}

    # evaluate the frozen graph performance
    4b_eval_graph 2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_FR_LOG}

    # quantize
    5a_unet_quantize 2>&1 | tee ${LOG_DIR}/${CNN}/${QUANT_LOG}

    # evaluate post-quantization model
    5b_eval_quantized_graph 2>&1 | tee ${LOG_DIR}/${CNN}/${EVAL_Q_LOG}

    # compile with Vitis AI to generate elf file for ZCU102
    #6_compile_vai_vck190 2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel to  target board directory
    #mv  ${COMPILE_DIR}/${CNN}4/*.xmodel    ${TARGET_190}/${CNN}/v4/model/
    #rm ${TARGET_190}/${CNN}/v4/model/*_org.xmodel

    # compile with Vitis AI to generate elf file for ZCU102
    6_compile_vai_zcu102 2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel to  target board directory
    mv ${COMPILE_DIR}/${CNN}4/*.xmodel    ${TARGET_102}/${CNN}/v4/model/
    rm ${TARGET_102}/${CNN}/v4/model/*_org.xmodel

    # compile with Vitis AI to generate elf file for ZCU104
    #6_compile_vai_zcu104 2>&1 | tee ${LOG_DIR}/${CNN}/${COMP_LOG}
    # move xmodel to  target board directory
    #mv  ${COMPILE_DIR}/${CNN}4/*.xmodel    ${TARGET_104}/${CNN}/v4/model/
    #rm ${TARGET_104}/${CNN}/v4/model/*_org.xmodel

    # copy test images into target board
    cd ${DATASET_DIR}
    mkdir -p test_data
    cd ../../

    cd ${RAW_DATASET_DIR}/train_v2
    cat ../../${DATASET_DIR}/calib.csv | cut -d \, -f 2 | xargs -I % cp % ../../${DATASET_DIR}/test_data
    cd ../../
    cp ${DATASET_DIR}/calib.csv ${DATASET_DIR}/test_data

    tar -cvf "test.tar" ${DATASET_DIR}/test_data
    gzip test.tar
    cp test.tar.gz ${ARCHIVE_DIR}/
    
    #cp test.tar.gz ${TARGET_190}/ 
    #cp -r rpt ${TARGET_190}/rpt_deploy
    #cp -r log ${TARGET_190}/log_deploy
    #tar -cvf target_vck190.tar ${TARGET_190}/
    #gzip target_vck190.tar
    #mv target_vck190.tar.gz ${ARCHIVE_DIR}/target_vck190_unet_${BACKBONE}_${EPOCHS}ep_${BATCH_SIZE}x${WIDTH}x${HEIGHT}.tar.gz

    #cp test.tar.gz ${TARGET_102}/
    cp -r rpt ${TARGET_102}/rpt_deploy
    cp -r log ${TARGET_102}/log_deploy
    tar -cvf target_zcu102.tar ${TARGET_102}/
    gzip target_zcu102.tar
    mv target_zcu102.tar.gz ${ARCHIVE_DIR}/target_zcu102_unet_${BACKBONE}_${EPOCHS}ep_${BATCH_SIZE}x${WIDTH}x${HEIGHT}.tar.gz

    ##cp test.tar.gz ${TARGET_104}/
    #cp -r rpt ${TARGET_104}/rpt_deploy
    #cp -r log ${TARGET_104}/log_deploy
    #tar -cvf target_zcu104.tar ${TARGET_104}/
    #gzip target_zcu104.tar
    #smv target_zcu104.tar.gz ${ARCHIVE_DIR}/target_zcu104_unet_${BACKBONE}_${EPOCHS}ep_${BATCH_SIZE}x${WIDTH}x${HEIGHT}.tar.gz

    echo "#####################################"
    echo "MAIN UNET FLOW COMPLETED"
    echo "#####################################"
}

main
