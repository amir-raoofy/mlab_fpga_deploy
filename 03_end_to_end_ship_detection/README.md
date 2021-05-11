# Notes

- Scripts of workflow for the deployment is located in `./files`

- Dataset needs to be downloaded and extracted manually it will be lolcated in `./files` as `dataset`. the script takes care of the preparation of the data.

# Instruction

- put extract the `airbus-ship-detection.zip` in `./files` as `dataset`.
- run the container `./docker_run.sh xilinx/vitis-ai-gpu`.
- run the driver `./run_all.sh`.

The above instruction creates the xmodel required for the deployment on FPGA, however we still need to build the host application for the target baord. We can either use the python codes which already use vart and xir or use the following procedure to cross-compile C codes code for the host of the target board: 

  ```bash
  unset LD_LIBRARY_PATH   
  sh ~/petalinux_sdk/2020.2/environment-setup-aarch64-xilinx-linux # set petalinux environment of Vitis AI 1.3
  cd <WRK_DIR>/tutorials/VAI-KERAS-FCN8-SEMSEG/files
  cd target_zcu102/code
  bash -x ./build_app.sh
  cd ..
  tar -cvf target_zcu102.tar ./target_zcu102 # to be copied on the SD card
  ```
