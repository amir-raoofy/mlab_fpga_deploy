from ctypes import *
from typing import List
import cv2
import numpy as np
import xir
import vart
import os
import math
import threading
import time
import sys

import eval_utils as eu
import csv

Verbose = True
Debug = False

def CPUCalcSigmoid(data, size):
    output = []
    for idx in range(size):
        output.append(1.0/(1.0+np.exp(-data[idx])))
    return output


def CPUZeroPad2d(data, size):
    output = []
    for idx in range(size):
        #output.append ( np.pad(data[idx], 1, 0) )
        #print (data[idx].shape)
        output.append(np.pad(data[idx], ((1, 1), (1, 1), (0, 0)), 'constant'))
    return output


def get_script_directory():
    path = os.getcwd()
    return path


def scale_one_image(image):
    IM_MIN = np.min(image)
    IM_MAX = np.max(image)
    if (IM_MIN != IM_MAX):
        image = (255.0/(IM_MAX-IM_MIN)) * (image - IM_MIN)
    return image


def preprocess_one_image_fn(image_path, width=224, height=224):

    if 'jpg' in image_path:
        #print (image_path)
        image = cv2.imread(image_path)
        image = cv2.resize(image, (width, height))
    else:
        image = np.zeros((width, height, 3))

    #IM_MIN = np.min(image)
    #IM_MAX = np.max(image)
    #image = (2.0/(IM_MAX-IM_MIN)) * (image - IM_MIN) - 1
    return image


SCRIPT_DIR = get_script_directory()
test_image_dir = SCRIPT_DIR + "/dataset/test_data/"
global threadnum
threadnum = 0


def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def get_child_subgraph_cpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if not (cs.has_attr("device") and cs.get_attr("device").upper() == "DPU")
    ]


def runUnetSubgraph(runner: "Runner", img, cnt, start, end, subgraph_id):
    """get tensor"""
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)
    output_ndim = tuple(outputTensors[0].dims)

    n_of_images = end-start #len(img[0])
    count = start
    while count < end:
        runSize = input_ndim[0]
        """prepare batch input/output """
        inputData = [np.ones(tuple(inputTensors[k].dims), dtype=np.float32, order="C")
                     for k in range(len(inputTensors))]
        outputData = [np.empty(tuple(outputTensors[k].dims), dtype=np.float32, order="C")
                      for k in range(len(outputTensors))]

        """init input image to input buffer """
        for k in range(len(img)):
            for j in range(runSize):
                imageRun = inputData[k]
                imageRun[j, ...] = img[k][(
                    count + j) % cnt].reshape(tuple(inputTensors[k].dims)[1:])

        """run with batch """
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)

        for k in range(len(outputTensors)):
            for j in range(runSize):
                output_imgs_dpu[subgraph_id][k][(count + j) %
                                      cnt] = outputData[k][j]

        count = count + runSize

def main(argv):
    global threadnum

    listimage = os.listdir(test_image_dir)
    threadnum = int(argv[1])
    i = 0
    global runTotall
    runTotall = 128  # len(listimage)
    cnt = int(argv[3])

    g = xir.Graph.deserialize(argv[2])
    subgraphs_dpu = get_child_subgraph_dpu(g)
    subgraphs_cpu = get_child_subgraph_cpu(g)

    if (Debug==True):
        print("subgraphs_cpu: ", len(subgraphs_cpu))
        print("subgraphs_dpu: ", len(subgraphs_dpu))
        print("CPU subgraphs:")
        for id, item in enumerate (subgraphs_cpu):
            print("CPU-node: ", id)
            print(item.get_name())
            print(item.get_input_tensors())
            print(item.get_output_tensors())
            for jtem in item.get_input_tensors():
                print(tuple(jtem.dims))
            for jtem in item.get_output_tensors():
                print(tuple(jtem.dims))

        print("DPU subgraphs:")
        for id, item in enumerate (subgraphs_dpu):
            print("DPU-node: ", id)
            print(item.get_name())
            print(item.get_input_tensors())
            print(item.get_output_tensors())
            for jtem in item.get_input_tensors():
                print(tuple(jtem.dims))
            for jtem in item.get_output_tensors():
                print(tuple(jtem.dims))

    def get_name(x):
        return x.name

    # prepare a list of input tensors for DPU subgraphs
    input_tensors_dpu = []
    output_tensors_dpu = []
    for item in subgraphs_dpu:
        tmp = item.get_input_tensors()
        tmp_list = sorted([i for i in tmp], key=get_name)
        input_tensors_dpu.append(tmp_list)

        tmp = item.get_output_tensors()
        tmp_list = sorted([i for i in tmp], key=get_name)
        output_tensors_dpu.append(tmp_list)

    global input_imgs_dpu
    input_imgs_dpu = []
    global output_imgs_dpu
    output_imgs_dpu = []

    # allocating input and output for DPU subgraphs
    for item in input_tensors_dpu:
        input_imgs_dpu.append([])
        for jtem in item:
            data_size = jtem.dims.copy()
            data_size[0] = cnt
            data_size = tuple(data_size)
            input_imgs_dpu[-1].append(np.zeros(data_size, dtype=np.float32))

    for item in output_tensors_dpu:
        output_imgs_dpu.append([])
        for jtem in item:
            data_size = jtem.dims.copy()
            data_size[0] = cnt
            data_size = tuple(data_size)
            output_imgs_dpu[-1].append(np.zeros(data_size, dtype=np.float32))

    if (Debug==True):
        for item in input_tensors_dpu:
            print (item)

    if (Debug==True):
        for item in input_imgs_dpu:
            for jtem in item:
                print(jtem.shape)
            print("\n")
        for item in output_imgs_dpu:
            for jtem in item:
                print(jtem.shape)
            print("\n")

    start_list = []
    end_list =[]
    start = 0
    for i in range(threadnum):
        if (i==threadnum-1):
            end = cnt
        else:
            end = start+(cnt//threadnum)

        start_list.append(start)
        end_list.append(end)

        start=end

    if (Debug==True):
        print (start_list)
        print (end_list)


    width = input_imgs_dpu[0][0].shape[1]
    height = input_imgs_dpu[0][0].shape[2]
    
    BATCH_SIZE = cnt
    RAW_DATASET_DIR = "./dataset"
    TEST_DIR = os.path.join(RAW_DATASET_DIR, 'test_data')
    test_csv_path = os.path.join(TEST_DIR,"calib.csv")

    file =  open(test_csv_path, 'r')
    reader = csv.DictReader(file)

    x_test, y_test = eu.batch_data_get(reader, TEST_DIR, BATCH_SIZE, width, height, augmentation=None)

    file.close()

    """image list to be run """
    for i in range(cnt):
        input_imgs_dpu[0][0][i] = x_test[i]

    if (Verbose==True):
        for idx in range(cnt):
            cv2.imwrite('./rpt/image_'+str(idx)+'.jpg', input_imgs_dpu[0][0][idx])
            cv2.imwrite('./rpt/mask_'+str(idx)+'.jpg', 255*y_test[idx])
    time_start = time.time()
    #######################################################
    # DPU subgraph 0
    #######################################################
    inps = [input_imgs_dpu[0][0]]

    threadAll = []
    all_dpu_runners = []
    subgraph_id = 0
    for i in range(int(threadnum)):
        all_dpu_runners.append(
            vart.Runner.create_runner(subgraphs_dpu[subgraph_id], "run"))

    """run with batch """
    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runUnetSubgraph, args=(
            all_dpu_runners[i], inps, cnt, start_list[i], end_list[i], subgraph_id))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    #######################################################
    # CPU subgraph 1
    #######################################################
    input_imgs_dpu[1] = CPUZeroPad2d(output_imgs_dpu[0][0], cnt)
    if (Debug==True):
        for idx in range(64):
            cv2.imwrite('./rpt/pred_1_'+str(idx)+'.jpg',
                        scale_one_image(input_imgs_dpu[1][0][:, :, idx]))
    #######################################################
    # DPU subgraph 1
    #######################################################
    inps = [input_imgs_dpu[1]]

    threadAll = []
    subgraph_id = 1
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(
            vart.Runner.create_runner(subgraphs_dpu[subgraph_id], "run"))

    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runUnetSubgraph, args=(
            all_dpu_runners[i], inps, cnt, start_list[i], end_list[i], subgraph_id))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    #######################################################
    # CPU subgraph 2
    #######################################################
    input_imgs_dpu[2] = CPUZeroPad2d(output_imgs_dpu[1][0], cnt)
    #######################################################
    # DPU subgraph 2
    #######################################################
    inps = [output_imgs_dpu[1][0], input_imgs_dpu[2]]

    threadAll = []
    subgraph_id = 2
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(
            vart.Runner.create_runner(subgraphs_dpu[subgraph_id], "run"))

    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runUnetSubgraph, args=(
            all_dpu_runners[i], inps, cnt, start_list[i], end_list[i], subgraph_id))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    #######################################################
    # CPU subgraph 3
    #######################################################
    input_imgs_dpu[3] = CPUZeroPad2d(output_imgs_dpu[2][1], cnt)
    #######################################################
    # DPU subgraph 3
    #######################################################
    #inps = [output_imgs_dpu[2][1], output_imgs_dpu[0][0],
    #        output_imgs_dpu[1][0], output_imgs_dpu[2][0], input_imgs_dpu[3]]

    inps = [output_imgs_dpu[0][0], output_imgs_dpu[1][0],
            output_imgs_dpu[2][0], output_imgs_dpu[2][1], input_imgs_dpu[3]]

    threadAll = []
    subgraph_id = 3
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(
            vart.Runner.create_runner(subgraphs_dpu[subgraph_id], "run"))

    for i in range(int(threadnum)):
        t1 = threading.Thread(target=runUnetSubgraph, args=(
            all_dpu_runners[i], inps, cnt, start_list[i], end_list[i], subgraph_id))
        threadAll.append(t1)
    for x in threadAll:
        x.start()
    for x in threadAll:
        x.join()

    del all_dpu_runners

    if (Debug==True):
        for idx in range(cnt):
            cv2.imwrite('./rpt/pred_3_'+str(idx)+'.jpg',
                        scale_one_image(output_imgs_dpu[2][0][idx][:, :, 0]))
    #######################################################
    # CPU subgraph 4
    #######################################################
    prediction = CPUCalcSigmoid(output_imgs_dpu[3][0], cnt)
    if (Verbose==True):
        for idx in range(cnt):
            cv2.imwrite('./rpt/prediction_'+str(idx)+'.jpg',
                        scale_one_image(prediction[idx][:, :, 0]))
    #######################################################

    time_end = time.time()
    timetotal = time_end - time_start
    total_frames = cnt #* int(threadnum)
    fps = float(total_frames / timetotal)
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds"
        % (fps, total_frames, timetotal)
    )

    mIoU = eu.IoU_all (y_test,prediction)
    print ("IoU for model for the test dataset: ", mIoU)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage : python3 script.py <thread_number> <xmodel_file> <num_frames>")

    else:
        main(sys.argv)
