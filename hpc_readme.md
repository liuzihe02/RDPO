# Compute: H100 cluster

Specification:

- A reminder they are Cray XD670, 1TB RAM and 8xH100 80G SXM2
- There are 4 added to the cluster called seymour1 - 4 and yours are seymour3 and seymour4.
- cuda 12.4 installed
- We strongly recommend you create a python virtual env using the python in /share/apps using 3.11.9 and create your own source file to reference at the beginning of the job.

## Steps

### 1. SSH into gateway node

knuckles is a gateway machine to access UCL internal CS network
```bash
ssh zachliu@knuckles.cs.ucl.ac.uk
```
### 2. SSH into UCL login node
```bash
ssh zachliu@vic.cs.ucl.ac.uk
```
vic for computer science department

### 3. SSH into compute node
For an interactive session for 12hrs

```bash
qrsh -l tmem=120G,gpu=true,h_rt=12:00:00,gpu_type=h100 -P aihub_ucl

### bash 
```

for shorter time/less memory etc
```bash
qrsh -l tmem=12G,gpu=true,h_rt=00:30:00,gpu_type=h100 -P aihub_ucl
```

### 4. Job Submission Scripts

More info on job submission: https://hpc.cs.ucl.ac.uk/gpus/ (user/pass = hpc/comic)

Example, in ~ directory

```jsx
qsub ./work/DiffRatio/cluster/train_cifar_h100.sub
```

1. `.sub` config template:

```bash
#!/bin/bash

#$ -S /bin/bash

#$ -l gpu=1
#$ -l tmem=120G 
#$ -l h_rt=96:00:00
#$ -P aihub_ucl
#$ -N flow-vae-try ###<- change to your run name 
#$ -l gpu_type=h100
#$ -j y

CODE_DIR=/home/minzhang/work/AVAE
export PYTHONPATH=$PYTHONPATH:$CODE_DIR

#sets up neccessary env variables to use python 3.11.9
source /share/apps/source_files/python/python-3.11.9.source

#sets up relevant path variables for cuda
#these cuda files are located ONLY on the compute nodes, so you have to manually include these in your script!
export PATH=/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/usr/local/cuda-12.4

export WANDB_API_KEY= ####Your wandb api key

```

Another example of a `bash` file for `qsub`:
```bash
#   This is the most basic QSUB file needed for this cluster.
#   Further examples can be found under /share/apps/examples
#   Most software is NOT in your PATH but under /share/apps
#
#   For further info please read http://hpc.cs.ucl.ac.uk
#   For cluster help email cluster-support@cs.ucl.ac.uk
#
#   NOTE hash dollar is a scheduler directive not a comment.


# These are flags you must include - Two memory and one runtime.
# Runtime is either seconds or hours:min:sec

#$ -l tmem=2G
#$ -l h_vmem=2G
#$ -l h_rt=3600

#These are optional flags but you probably want them in all jobs

#$ -S /bin/bash
#$ -j y
#$ -N MyTESTJOBNAME

#The code you want to run now goes here.

hostname
date

sleep 20

date
```

check usage; the `-f` flag lists everything

```bash
qstat -f 
```

Save path, this has 1T to use

```jsx
/SAN/intelsys/GenerativeModel 
```

## Zach's Notes

### Shared Code

Within `/share/apps` is a directory called `examples`. This is where we store source scripts and example submission scripts. For example, within `/share/apps/source_files/python` there is a file called `python-3.8.5.source`, which contains the following:

```bash
#Python 3.8.3 Source
export PATH=/share/apps/python-3.8.5-shared/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/python-3.8.5-shared/lib:$LD_LIBRARY_PATH
```

This sets two bash variables (`PATH` and `LD_LIBRARY_PATH`), which will allow you to simply type python to run python v3.8.5

### Login Nodes Vs Compute Nodes

The HPC setup you're using explicitly separates nodes into two types:

- Login Nodes (`vic`, `wise`, etc.):
  - No CUDA/GPU tools (e.g., no `nvcc`, no `nvidia-smi`)
  - Meant only for job submissions, file transfers, and preparation tasks.
  - No computations allowed to ensure system stability.
- Compute Nodes (seymour4, etc.):
  - GPUs and CUDA tools (like nvcc, nvidia-smi) are installed here.
  - Your code runs here.

### `~/.bashrc`
Note that I have added the following extra lines to the `~/.bashrc` file

```bash
#these source commands uses the existing settings in shared folders to setup relevant packages and environment variables

#sets up neccessary env variables to use python 3.11.9
source /share/apps/source_files/python/python-3.11.9.source

#uses newer version of software like gcc and g++
source /opt/rh/devtoolset-9/enable

#for the rest, there isnt a source file available so we manually set environment variables ourselves

#adds nano to path variable
#export makes the env variable available to all subsequnt commands
#we basically append the path to nano bin to the existing path variable
export PATH=/share/apps/nano-5.8/bin:$PATH
```

The `source` cmd is used to execute a script or load env variables into the current session.

> Since the `~/.bashrc` file is sourced at *every* ssh login (gateway node, login node, compute node), we do not put the `CUDA` variables permanently in the `.bashrc` file. This may cause errors. Put this in the job submission scripts.

### Installing Packages

By default, `pip` creates an isolated environment where it builds dependencies. This process does not use packages from my virtual environment, but rather installs all dependencies separately before adding it to the environment.

I faced into some errors with `torch` not being found, so I did
```bash
pip install -r requirements.txt --no-build-isolation
```
This way `pip` can install the `torch` that is already installed. However this means I'd need to specify all the packages in my `requirements.txt`.

> I also faced into some errors with `wheel`, I had to `pip install` this myself
> Many problems with `flash-attn`, as we need build the wheel for this package ourselves

