# ================================================
# dataset
# tar -xvzf test.tar.gz

# ================================================
# convnet I
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v1/model/unet1.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v1/model/unet1.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v1/model/unet1.xmodel

# ================================================
# convnet III
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v3/model/unet3.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v3/model/unet3.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_single_layer.py 1 ./unet/v3/model/unet3.xmodel
# ================================================
# resnet
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size64x160x160.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_160.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size32x224x224.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_224.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_resnet50_ep100_bit8_8_size8x384x384.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102 
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_resnet_384.py 1 ./unet/v4/model/unet4.xmodel
# ================================================
# mobilenet
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size64x160x160.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size32x224x224.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_mobilenetv2_ep100_bit8_8_size8x384x384.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_mobilenetv2.py 1 ./unet/v4/model/unet4.xmodel
# ================================================
# mobilenet
rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size64x160x160.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_160.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size32x224x224.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_224.py 1 ./unet/v4/model/unet4.xmodel

rm -rf target_zcu102
cd ~
FILE=archive/target_zcu102_unet_inceptionresnetv2_ep100_bit8_8_size8x384x384.tar.gz
tar -xvzf $FILE
cp -r code target_zcu102
cp Makefile target_zcu102
cd target_zcu102/
ln -s ../build/dataset
python3 ./code/src/app_mt_inceptionresnetv2_384.py 1 ./unet/v4/model/unet4.xmodel
# ================================================
# cleanup dataset
# cd ~ rm -rf dataset
