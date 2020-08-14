# Quad GPU Neural Networks

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

1. Dynamic Scheduling with OpenMP Directives
2. Large dual-gpu matrix multiplication using Tensorflow. 
3. Quad GPU model parallelism using PyTorch 

### IDE 

* C Code built and run using CodeBlocks

* Python 3 Code built and run using Visual Studio Code 

### Prerequisites and Running on Discovery Cluster

1. Clone this project directory
    ```sh
    $ git clone https://github.com/srinjoychakravarty/parallelized_gpu_learning.git
    ```

2. Please switch to a multigpu partition with 4 gpus 
    ```sh
    $ srun -p reservation --reservation=CSYE7374-54713-Summer2020 --gres=gpu:4 --mem=16Gb --time=01:00:00 --export=ALL --pty /bin/bash
    ```

3. Recreate the customized environment required to run all the code in this repo:
    ```sh
    $ conda env create -f environment.yaml
    ```
Here _btc_over_mpi_env_ is the name you choose to give your environment. 

**Tip:** to see a list of all of your conda environments, type conda info -e.

4. Activate your imported conda environment:
    ```sh
    $ source activate btc_over_mpi_env
    ```

5. To see the OpenMP directives of Part II run the C code with the following command:
    ```
    $ ./part2
    ```
    
6. To see the Matrix Multiplication over 2 GPUs using Tensorflow run the Python 3 code with the following command: 
    ```sh
    $ python3 part3.py
    ```
    
7. To benchmark 1,2,3,4 GPUs using an Artificial Neural Network with Resnet run the Cuda code with the following command:
    ```sh
    $ python3 bonus.py
    ```

> **Tip:** To simulate a command run of this repo simply run the following command:
    ```sh
    $ scriptreplay --timing=timing.txt session.txt
    ```

License
----

Northeastern University

_Srinjoy Chakravarty_
