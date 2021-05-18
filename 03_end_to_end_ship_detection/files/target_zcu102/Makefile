all: build 

dataset:
	tar -xvf ../test.tar.gz
	mv build/dataset .

.PHONY: build
build:
	bash -x code/build_app.sh
	mv code/code ./run_cnn 

run:
	./run_cnn ./unet/v3/model/unet3.xmodel ./dataset/test_data/ 1 | tee ./rpt/logfile_cpp_unet2.txt

clean:
	./run_cnn 
