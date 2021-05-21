all: build

dataset:
	tar -xvf ../test.tar.gz
	mv ./build/dataset/ .
	rm -r build

build:
	#cd code
	bash -x code/build_app.sh
	mv code/code ./run_cnn # change name of the application

run:
	#./run_cnn ./unet/v3/model/unet3.xmodel ./dataset/test_data/ 1 | tee ./log/logfile_cpp_unet3.txt
	python3 ./code/src/app_mt.py 1 ./unet/v4/model/unet4.xmodel

clean_data:
	rm -rf dataset

clean_output:
	rm -rf ./rpt/*jpg

clean:
	rm -rf run_cnn

