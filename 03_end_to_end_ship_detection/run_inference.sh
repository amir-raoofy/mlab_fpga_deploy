# ================================================
# dataset
# tar -xzf test.tar.gz

# ================================================
# ================================================
# functions 
# ================================================
# ================================================
# convnet I
unet1_s () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size64x160x160.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel $FRAMES
}

unet1_m () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size32x224x224.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel $FRAMES
}

unet1_l () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size8x384x384.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel $FRAMES
}
# ================================================
# convnet III
unet3_s () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size64x160x160.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel $FRAMES
}

unet3_m () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size32x224x224.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel $FRAMES
}

unet3_l () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size8x384x384.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel $FRAMES
}

# ================================================

# resnet
unet_resnet_s () {
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_resnet_160.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_resnet_m(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_resnet_224.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_resnet_l(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102 
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_resnet_384.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

# ================================================
# mobilenet
unet_mobilenetv2_s(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size64x160x160.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_mobilenetv2_m(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size32x224x224.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_mobilenetv2_l(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size8x384x384.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

# ================================================
# inceptionresnetv2
unet_inceptionresnetv2_s(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size64x160x160.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_inceptionresnetv2_160.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_inceptionresnetv2_m(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size32x224x224.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_inceptionresnetv2_224.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}

unet_inceptionresnetv2_l(){
    THREADS=$1
    FRAMES=$2
    echo ""
    echo "${FUNCNAME[ 0 ]} $1 $2"
    cd ~
    rm -rf target_zcu102
    FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size8x384x384.tar.gz
    tar -xzf $FILE &> /dev/null
    cp -r code target_zcu102
    cp Makefile target_zcu102
    cd target_zcu102/
    ln -s ../build/dataset
    python3 ./code/src/app_mt_inceptionresnetv2_384.py $THREADS ./unet/v4/model/unet4.xmodel $FRAMES
}
# ================================================
# cleanup dataset
# cd ~ rm -rf dataset

# ================================================
# ================================================
# runs
# ================================================
# ================================================

# unet1
# ================================================
unet1_s 8 256
unet1_s 4 256
unet1_s 2 256
unet1_s 1 256

unet1_m 8 256
unet1_m 4 256
unet1_m 2 256
unet1_m 1 256

unet1_l 8 64
unet1_l 4 64
unet1_l 2 64
unet1_l 1 64

# unet3
# ================================================
unet3_s 8 256
unet3_s 4 256
unet3_s 2 256
unet3_s 1 256

unet3_m 8 256
unet3_m 4 256
unet3_m 2 256
unet3_m 1 256

unet3_l 8 64
unet3_l 4 64
unet3_l 2 64
unet3_l 1 64

# unet_mobilenetv2
# ================================================
unet_mobilenetv2_s 8 256
unet_mobilenetv2_s 4 256
unet_mobilenetv2_s 2 256
unet_mobilenetv2_s 1 256

unet_mobilenetv2_m 8 256
unet_mobilenetv2_m 4 256
unet_mobilenetv2_m 2 256
unet_mobilenetv2_m 1 256

unet_mobilenetv2_l 8 64
unet_mobilenetv2_l 4 64
unet_mobilenetv2_l 2 64
unet_mobilenetv2_l 1 64

# unet_resnet
# ================================================
unet_resnet_s 8 256
unet_resnet_s 4 256
unet_resnet_s 2 256
unet_resnet_s 1 256

unet_resnet_m 8 256
unet_resnet_m 4 256
unet_resnet_m 2 256
unet_resnet_m 1 256

unet_resnet_l 8 64
unet_resnet_l 4 64
unet_resnet_l 2 64
unet_resnet_l 1 64

# unet_inceptionresnetv2
# ================================================
unet_inceptionresnetv2_s 8 256
unet_inceptionresnetv2_s 4 256
unet_inceptionresnetv2_s 2 256
unet_inceptionresnetv2_s 1 256

unet_inceptionresnetv2_m 8 256
unet_inceptionresnetv2_m 4 256
unet_inceptionresnetv2_m 2 256
unet_inceptionresnetv2_m 1 256

unet_inceptionresnetv2_l 8 64
unet_inceptionresnetv2_l 4 64
unet_inceptionresnetv2_l 2 64
unet_inceptionresnetv2_l 1 64