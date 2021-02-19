# FPGA Deep Learning Models


This directory contains self-explaining, minimal  model files depending only on publicly available pip modules (reqiurements.txt)

- *00_initial_transfer_from_iclr_paper.py*: Use adapted resnet weights from ICLR workshop paper and train a decision layer directly on top of it. Current Baseline model, uses class weights. 


## Install and Run
Usage:
```
pip install -r requirements.txt
python3 <name of file> <name of config json>
```
where not each model consumes a config JSON.

In fact, I usually do something along
```
docker run --gpus=all -it --name=earthvision  -v $PWD:/tf tensorflow/tensorflow:latest-gpu-jupyter 
```
and then
```
docker exec -it earthvision bash
```

If you are a fan of Jupyter, you might also forward ports and configure your firewall or proxy accordingly.


## Shell Magic for Configuration
Note that with some shell magic you can generate the config JSON inline similar to 
```
python model.py <(cat << EOF
cfg={"key":42
}
EOF
)

```
This avoids having a file for the config.

## Dependencies / Preparations
- You need to run this sufficiently near to a dataset.
- You should run it via docker
