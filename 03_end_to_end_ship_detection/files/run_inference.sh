THREADS=4

# ================================================
# dataset
# tar -xzf test.tar.gz

# ================================================
# convnet I
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v1/model/unet1.xmodel 32

# ================================================
# convnet III
cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py $THREADS ./unet/v3/model/unet3.xmodel 32
# ================================================
# resnet
cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_160.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_224.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_384.py $THREADS ./unet/v4/model/unet4.xmodel 32
# ================================================
# mobilenet
cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size64x160x160.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size32x224x224.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size8x384x384.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py $THREADS ./unet/v4/model/unet4.xmodel 32
# ================================================
# mobilenet
cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size64x160x160.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_160.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size32x224x224.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_224.py $THREADS ./unet/v4/model/unet4.xmodel 256

cd ~
rm -rf target_zcu102
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size8x384x384.tar.gz
tar -xzf $FILE &> /dev/null
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_384.py $THREADS ./unet/v4/model/unet4.xmodel 32
# ================================================
# cleanup dataset
# cd ~ rm -rf dataset
