/*
 * Copyright 2019 Xilinx Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <assert.h>
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

#include "common.h"
/* header file OpenCV for image processing */
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// flags for each thread
bool is_running_1 = true;
bool is_running_2 = true;
bool is_running_3 = true;
bool is_running_4 = true;
bool is_running_5 = true;

GraphInfo shapes;

//const string baseImagePath = "./src/img_test/";
string baseImagePath; // they will get their values via argv[]

uint8_t colorB[] = {128, 232, 70, 156, 153, 153, 30, 0, 35, 152,
                    180, 60, 0, 142, 70, 100, 100, 230, 32};
uint8_t colorG[] = {64, 35, 70, 102, 153, 153, 170, 220, 142, 251,
                    130, 20, 0, 0, 0, 60, 80, 0, 11};
uint8_t colorR[] = {128, 244, 70, 102, 190, 153, 250, 220, 107, 152,
                    70, 220, 255, 0, 0, 0, 0, 0, 119};

/**
 * @brief put image names to a vector
 *
 * @param path - path of the image direcotry
 * @param images_list - the vector of image name
 *
 * @return none
 */
void ListImages(string const &path, vector<string> &images_list)
{
  images_list.clear();
  struct dirent *entry;

  /*Check if path is a valid directory path. */
  struct stat s;
  lstat(path.c_str(), &s);
  if (!S_ISDIR(s.st_mode))
  {
    fprintf(stderr, "Error: %s is not a valid directory!\n", path.c_str());
    exit(1);
  }

  DIR *dir = opendir(path.c_str());
  if (dir == nullptr)
  {
    fprintf(stderr, "Error: Open %s path failed.\n", path.c_str());
    exit(1);
  }

  while ((entry = readdir(dir)) != nullptr)
  {
    if (entry->d_type == DT_REG || entry->d_type == DT_UNKNOWN)
    {
      string name = entry->d_name;
      string ext = name.substr(name.find_last_of(".") + 1);
      if ((ext == "JPEG") || (ext == "jpeg") || (ext == "JPG") ||
          (ext == "jpg") || (ext == "PNG") || (ext == "png"))
      {
        images_list.push_back(name);
      }
    }
  }

  closedir(dir);
}

/**
 * @brief Run DPU Task for CNN
 *
 * @param taskFCN8 - pointer to FCN8 Task
 *
 * @return none
 */
void run_CNN(vart::Runner *runner, bool &is_running)
{

  vector<string> kinds, images_list;
  //vector<Mat> imageList;
  Mat image;

  /* Load all image names.*/
  ListImages(baseImagePath, images_list);
  if (images_list.size() == 0)
  {
    cerr << "\nError: No images existing under " << baseImagePath << endl;
    return;
  }

  //cout << "images_list.size(): " << images_list.size() << endl;

  // get in/out tensors and dims
  //auto outputTensors = runner->get_output_tensors();
  //auto inputTensors = runner->get_input_tensors();
  auto inputTensors = cloneTensorBuffer(runner->get_input_tensors());
  auto outputTensors = cloneTensorBuffer(runner->get_output_tensors());
  auto in_dims = inputTensors[0]->get_shape();
  auto out_dims = outputTensors[0]->get_shape();

  // get shape info
  int outSize = shapes.outTensorList[0].size;
  int outHeight = shapes.outTensorList[0].height;
  int outWidth = shapes.outTensorList[0].width;
  int outChannels = shapes.outTensorList[0].channel; // outSize / (inSize / 3);

  int inSize = shapes.inTensorList[0].size;
  int inHeight = shapes.inTensorList[0].height;
  int inWidth = shapes.inTensorList[0].width;
  int inpChannels = shapes.inTensorList[0].channel;   // outSize / (inSize / 3);
  int batchSize = in_dims[0];
  unsigned int runSize = 1;
  unsigned int i = 0;

  std::vector<std::unique_ptr<vart::TensorBuffer>> inputs, outputs;
  std::vector<vart::TensorBuffer *> inputsPtr, outputsPtr;
  std::vector<std::shared_ptr<xir::Tensor>> batchTensors;

  float *InpData = new float[batchSize * inSize];
  float *OutData = new float[batchSize * outSize];

  // debug messages
  cout << "outSize  " << outSize << endl;
  cout << "outW     " << outWidth << endl;
  cout << "outH     " << outHeight << endl;
  cout << "outChannels  " << outChannels << endl;
  cout << "inSize  " << inSize << endl;
  cout << "inpW     " << inWidth << endl;
  cout << "inpH     " << inHeight << endl;
  cout << "inpChannels     " << inpChannels << endl;
  cout << "batchSize " << batchSize << endl; // alway 1 for Edge

  for (int i = 0; i < inHeight * inWidth * inpChannels; i++)
  {
    InpData[i] = 0;
  }

  for (int i = 0; i < outHeight * outWidth * outChannels; i++)
  {
    OutData[i] = i;
  }

  // in/out tensor refactory for batch inout/output
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(inputTensors[0]->get_name(), in_dims, xir::DataType::FLOAT, sizeof(float) * 8u)));
  inputs.push_back(std::make_unique<CpuFlatTensorBuffer>(InpData, batchTensors.back().get()));
  batchTensors.push_back(std::shared_ptr<xir::Tensor>(xir::Tensor::create(outputTensors[0]->get_name(), out_dims, xir::DataType::FLOAT, sizeof(float) * 8u)));
  outputs.push_back(std::make_unique<CpuFlatTensorBuffer>(OutData, batchTensors.back().get()));

  // tensor buffer input/output
  inputsPtr.push_back(inputs[0].get());
  outputsPtr.push_back(outputs[0].get());

  // run
  auto job_id = runner->execute_async(inputsPtr, outputsPtr);
  int st = runner->wait(job_id.first, -1);
  cout << "status: " << st << endl;

  for (int row = 0; row < inHeight; row++)
  {
    for (int col = 0; col < inWidth; col++)
    {
      for (int c = 0; c < inpChannels; c++)
      {
        int ii = row * inWidth * inpChannels + col * inpChannels + c;
        cout << InpData[ii] << ", ";
      }
      cout << endl;
    }
  }

  for (int row = 0; row < outHeight; row++)
  {
    for (int col = 0; col < outWidth; col++)
    {
      for (int c = 0; c < outChannels; c++)
      {
        int ii = row * outWidth * outChannels + col * outChannels + c;
        cout << OutData[ii] << ", ";
      }
      cout << endl;
    }
  }

  inputsPtr.clear();
  outputsPtr.clear();
  inputs.clear();
  outputs.clear();
  delete[] OutData;
  delete[] InpData;
}

/**
 * @brief Entry for running CNN neural network
 *
 * @note Runner APIs prefixed with "dpu" are used to easily program &
 *       deploy CNN on DPU platform.
 *
 */
int main(int argc, char *argv[])
{
  // Check args
  if (argc != 4)
  {
    cout << "Usage: run_cnn xmodel_pathName test_images_pathname threads" << endl;
    return -1;
  }
  baseImagePath = std::string(argv[2]); //path name of the folder with test images
  int num_threads = stoi(std::string(argv[3]));

  auto graph = xir::Graph::deserialize(argv[1]);
  auto subgraph = get_dpu_subgraph(graph.get());
  //CHECK_EQ(subgraph.size(), 1u)
  //    << "CNN should have one and only one dpu subgraph.";
  LOG(INFO) << "create running for subgraph: " << subgraph[0]->get_name();

  // create runner
  auto runner = vart::Runner::create_runner(subgraph[0], "run");
  //auto runner2 = vart::Runner::create_runner(subgraph[0], "run");
  // ai::XdpuRunner* runner = new ai::XdpuRunner("./");
  // get in/out tensor
  auto inputTensors = runner->get_input_tensors();
  auto outputTensors = runner->get_output_tensors();

  // get in/out tensor shape
  int inputCnt = inputTensors.size();
  int outputCnt = outputTensors.size();
  TensorShape inshapes[inputCnt];
  TensorShape outshapes[outputCnt];
  shapes.inTensorList = inshapes;
  shapes.outTensorList = outshapes;
  getTensorShape(runner.get(), &shapes, inputCnt, outputCnt);

  // Run tasks
  array<thread, 1> threads = {
      thread(run_CNN, runner.get(), ref(is_running_1)),
      //thread(run_CNN, runner2.get(), ref(is_running_2)),
      //thread(run_CNN, runner2.get(), ref(is_running_3)),
      //thread(run_CNN, runner2.get(), ref(is_running_4)),
      //thread(run_CNN, runner2.get(), ref(is_running_5))
  };

  if (num_threads > 5)
    num_threads = 5;

  for (int i = 0; i < num_threads; ++i)
  {
    threads[i].join();
  }

  return 0;
}
