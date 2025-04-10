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

`vic` for computer science department

### 3. SSH into compute node

For an interactive session for 12hrs

```bash
qrsh -l tmem=120G,gpu=true,h_rt=12:00:00,gpu_type=h100 -P aihub_ucl
```

for shorter time/less memory etc

```bash
qrsh -l tmem=50G,gpu=true,h_rt=02:00:00,gpu_type=h100 -P aihub_ucl
```

with scratch space (local storage on SSD) requested

```bash
qrsh -verbose -l tmem=20G,gpu=true,h_rt=00:59:00,gpu_type=h100,tscratch=30G -P aihub_ucl
```

without gpu (need to provide `h_vmem`), `-verbose` for debugging

```bash
qrsh -l tmem=14G,h_vmem=14G,h_rt=00:59:59
qrsh -verbose -l tmem=5G,h_vmem=5G,h_rt=01:00:00,gpu=false
```

There is no need to provide `gpu=false` in arguments. `tmem` and `h_vmem` should be set to the same value for CPU jobs, as mismatched values can cause issues.

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
- Compute Nodes (`seymour4`, etc.):
  - GPUs and CUDA tools (like nvcc, nvidia-smi) are installed here.
  - Your code runs here.

### `~/.bashrc`

Note that I have added the following extra lines to the `~/.bashrc` file

```bash
# User specific aliases and functions

 User specific aliases and functions
# Activates nano
export PATH=/share/apps/nano-5.8/bin:$PATH

# Uses newer version of software like gcc and g++
source /opt/rh/devtoolset-9/enable

# Sets up necessary env variables to use newer python
source /share/apps/source_files/python/python-3.10.0.source

#use the right version of cuda! make sure it exists in share apps
export PATH=/share/apps/cuda-12.3/bin:$PATH
export LD_LIBRARY_PATH=/share/apps/cuda-12.3/lib64:$LD_LIBRARY_PATH
export CUDA_HOME=/share/apps/cuda-12.3

# Override cmake *AFTER* all the others to take priority
# newer version needed to compile stuff
export PATH=/share/apps/cmake-3.26.4/bin:$PATH
```

The `source` cmd is used to execute a script or load env variables into the current session.

> Since the `~/.bashrc` file is sourced at _every_ ssh login (gateway node, login node, compute node), we do not put the `CUDA` variables permanently in the `.bashrc` file. This may cause errors. Put this in the job submission scripts.
> Note that the compute nodes may not be able to access `/share/apps` which may result in some errors

### Installing Packages

By default, `pip` creates an isolated environment where it builds dependencies. This process does not use packages from my virtual environment, but rather installs all dependencies separately before adding it to the environment.

```bash
#somehow we need to install these packages manually
pip install packaging wheel torch==2.4.0
#update these tools first especially pip
pip install --upgrade pip wheel setuptools
#you can try no build isolation so it can access the dependencies in our venv
pip install -r requirements.txt
```

If regular install doesn't work, try using `--no-build-isolation` flag. This way `pip` can install the `torch` that is already installed. However this means I'd need to specify all the packages in my `requirements.txt`.

> I faced into some errors with `wheel` and `torch`, I had to `pip install` this myself
> Many problems with `flash-attn`, as we need build the wheel for this package ourselves
> I downloaded the prebuilt wheel from [here](https://github.com/mjun0812/flash-attention-prebuild-wheels/releases), but this is very janky and machine specific! Needs specific version of Python, CUDA, and Pytorch

### Using Scratch Space

Each node has its own local disk or SSD, with typically around 200GB of space available.

This is mounted on `/scratch0` and is typically the best place for large amounts of data to be accessed from by your job. N.B. Scratch means just that it will get deleted on a regular basis. Note that `/scratch0 `is different for each node.

To use `/scratch0` add `#$ -l tscratch=<n>G` to your submission script. Additionally, before you move any data to scratch, create a directory at the top level of `/scratch0` with your username and job ID. For example: `mkdir -p /scratch0/alice/$JOB_ID`

```bash
qrsh -l tmem=80G,gpu=true,h_rt=00:30:00,gpu_type=h100,tscratch=30G -P aihub_ucl
#create directoy in scratch space
mkdir -p /scratch0/$USER
cd /scratch0/$USER
#create new venv
python3 -m venv venv
#activate this venv
source /scratch0/$USER/venv/bin/activate
```

### Memory Management

`tmem` (Total Memory Requested)

- The soft memory allocation for your job.
- It defines how much RAM your job expects to use.
- If your job exceeds this value, it may slow down but will not necessarily be killed immediately.

`h_vmem` (Hard Virtual Memory Limit)

- The absolute maximum memory your job is allowed to use.
- If your job exceeds this value, it will be terminated immediately by the scheduler.
- Setting this value ensures that a job does not consume all available memory on a node.

> When submitting a CPU job, the flag `-l gpu=false` is unnecessary and should be removed. 
> Additionally, please ensure that `tmem` and `h_vmem` are set to the same value for CPU jobs, as mismatched values can cause issues.

### Parallel Environments: SMP

For detailed information on the system's cpu architecture, use `lscpu`. To see the number of CPU cores currently allocated to the running process, use `nproc`.

`-pe smp 8` requests a parallel environment, symmetric multi-processing [with] 8 [cores]”. Use this environment to run multi-threaded or OpenMP applications over multiple cores within a compute node

> The `tmem` and `h_vmem` are multiplied by the number of cores requested! The scheduler interprets these values as being per thread (one thread is equivalent to one core)

## Data

Each time you login into the UCL network, you may need to download data again for some reason. Maybe to load it in some kind of cache.

## Minimal Dependencies

Note the `~/.bashrc` file; I have used `cuda-12.3` and `python-3.10.0`. Most packages require at least `cuda-12.0`. We have manually specified the compatible versions of these packages.

```bash
pip install tqdm packaging numpy wheel torch==2.4.0 transformers==4.49.0 datasets==3.2.0 accelerate==1.2.1 peft==0.12.0 trl==0.9.6 bitsandbytes==0.43.2 deepspeed==0.16.4 triton==3.0.0
```

install llamafactory
```bash
cd LLaMA-Factory
pip install -e ".[torch,metrics]" -v
```

### Troubleshooting

If pip install bitsandbytes doesnt work, we have to build this specific version from source.

```bash
# Clone the repo
git clone https://github.com/bitsandbytes-foundation/bitsandbytes.git
#go to the repo
cd bitsandbytes
#fetch the tag
git fetch --all --tags
# note sometimes the name comes with v0.43.2 sometimes they dont
git checkout 0.43.2
# check if right tag
git describe --tags
# Build with CMake, may need to use newer version of cmake
# need cuda toolkit and nvcc too
# need to specify which cuda architecture we wanna build for
# the H100 is built for compute capability 90
cmake -DCOMPUTE_BACKEND=cuda -DCMAKE_CUDA_ARCHITECTURES=90 -S . -v
make VERBOSE=1
#ensure you are in the bitsandbytes folder, so it can find the setup.py or pyproject.toml
#will install the package into the current activated venv
#we install WITHOUT DEPENDENCIES, ONLY THIS PACKAGE
pip install -e . --no-deps -v  # `-e` for "editable" install, when developing BNB (otherwise leave that out)
```

Then install the regular packages with their versions normally.



