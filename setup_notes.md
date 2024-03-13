# 2024-02-08 setup notes

Need to install TensorFlow with GPU support which depends on CUDA and cuDNN. Versioning really matters here.

## Contents

1. Base system configuration
2. cuDNN installation
3. TensorFlow installation
4. Additional TensorFlow GOTCHYAs
5. Bind mount fast scratch

## 1. Base system configuration

Here is what we are starting with:

- Python: 3.8.10
- GPU: Tesla K80
- Nvidia driver: 470.42.01
- CUDA driver: 11.4
- CUDA runtime: 11.4
- CUDA compute capability: 3.7
- GCC 9.4.0

It looks like there never was a TensorFlow version released for CUDA 11.4 (see [here](https://www.tensorflow.org/install/source#gpu)). According to [this issue](https://github.com/tensorflow/tensorflow/issues/55492) on GitHub, TensorFlow for CUDA 11.2 should work with CUDA 11.4. The thread mentions using TensorFlow-2.8.0 (which was the most current version at the time of the posts) for CUDA 11.4. But, since our GCC is already version 9.4, TensorFlow 2.9.0 - 2.11.0 are the closest matches for us. I believe Bazel is only needed to compile from source an we aren't planning on doing that. The only other prerequisite we need is the same for all three TensorFlow versions of interest: cuDNN 8.1.

## 2. cuDNN installation

Official install instructions are available from the [Nvidia archives](https://docs.nvidia.com/deeplearning/cudnn/archives/cudnn-810/install-guide/index.html). Taking a look at the CUDA compatibility matrix linked on that site, it seems like the recommend configuration for older cards is CUDA 11.8 with cuDNN 9.0.0. Unfortunately, we have only have CUDA 11.4 installed. I could not get 11.8 to install without also pulling in a later, incompatible driver (>500) and borking the system. Maybe it's time to revisit that - the [compatibility matrix](https://docs.nvidia.com/deeplearning/cudnn/reference/support-matrix.html) says CUDA 11.8, cuDNN 9.0.0 and driver >= 450 is the best setup. The latest driver for our K80 is 470, so it seems like the recommended setup should should work. But, in order to avoid reinstalling the whole GPU stack from scratch, let's just stick with the original plan for now and install cuDNN 8.1 as required by TensorFlow.

```text
sudo cp ./cuda/include/cudnn*.h /usr/local/cuda/include/
sudo cp -P ./cuda/lib64/libcudnn* /usr/local/cuda/lib64/
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
```

## 3. TensorFlow installation

Next, let's install TensorFlow-2.11.0 because it's the latest version which has a chance of being compatible with our CUDA 11.4.

Make and activate a python 3.8 venv:

```text
python3 -m venv .venv
source .venv/bin/activate
```

Then install and test Tensorflow 2.11.0:

```text
(.venv)$ pip install tensorflow==2.11.0
(.venv)$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

2024-02-07 21:12:04.839568: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer.so.7'; dlerror: libnvinfer.so.7: cannot open shared object file: No such file or directory
2024-02-07 21:12:04.839699: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libnvinfer_plugin.so.7'; dlerror: libnvinfer_plugin.so.7: cannot open shared object file: No such file or directory
2024-02-07 21:12:04.839726: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Cannot dlopen some TensorRT libraries. If you would like to use Nvidia GPU with TensorRT, please make sure the missing libraries mentioned above are installed properly.
2024-02-07 21:12:08.335738: W tensorflow/compiler/xla/stream_executor/platform/default/dso_loader.cc:64] Could not load dynamic library 'libcudnn.so.8'; dlerror: libcudnn.so.8: cannot open shared object file: No such file or directory
2024-02-07 21:12:08.335782: W tensorflow/core/common_runtime/gpu/gpu_device.cc:1934] Cannot dlopen some GPU libraries. Please make sure the missing libraries mentioned above are installed properly if you would like to use GPU. Follow the guide at https://www.tensorflow.org/install/gpu for how to download and setup the required libraries for your platform.
Skipping registering GPU devices...
[]
```

OK, still not working. Seems like most of these errors and warnings are related to tensorrt. See [here](https://stackoverflow.com/questions/74956134/could-not-load-dynamic-library-libnvinfer-so-7) and [here](https://github.com/tensorflow/tensorflow/issues/57679#issuecomment-1249197802). A quick look inside our venv tells me that we have libnvinfer.so.8 not libnvinfer.so.7 etc. Following the comment in the GitHub issue, let's install the correct version of tensorrt:

```text
(.venv)$ pip install nvidia-tensorrt==7.2.3.4
```

Then add to LD_LIBRARY_PATH

```text
(.venv)$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/arkk/rpm/skylines/skylines/.venv/lib/
(.venv)$ export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/arkk/rpm/skylines/skylines/.venv/lib/python3.8/site-packages/tensorrt/
(.venv)$ python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU')]
```

Made changes to environment 'persistent' by adding the following to the runner script *train.sh*:

```text
# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/mnt/arkk/rpm/skylines/skylines/.venv/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/arkk/rpm/skylines/skylines/.venv/lib/python3.8/site-packages/tensorrt/
```

Holy crap, it worked! Ok, now we just need matplotlib:

```text
pip install matplotlib
```

With that, it's working. I get the sense that we did not need to install cuDNN system wide - looked like tensorrt pulled it into our venv during the pip install. Maybe worth round-tripping this with a clean venv, just to make sure we know what we did that worked. Anyway, still seeing some warnings from TensorFlow during training. Let's take a quick look at those and see if they are anything we need to worry about.

## 4. Additional GOTCHYAs

### 4.1. Lambda functions

```text
WARNING:tensorflow:From /mnt/arkk/rpm/skylines/skylines/.venv/lib/python3.8/site-packages/tensorflow/python/autograph/pyct/static_analysis/liveness.py:83: Analyzer.lamba_check (from tensorflow.python.autograph.pyct.static_analysis.liveness) is deprecated and will be removed after 2023-09-23.
Instructions for updating:
Lambda fuctions will be no more assumed to be used in the statement where they are used, or at least in the same block. https://github.com/tensorflow/tensorflow/issues/56089
```

(typo not mine.)

Not a big deal. See [here](https://github.com/tensorflow/tensorflow/issues/56089) and [here](https://github.com/tensorflow/tensorflow/commit/6197fa37555b710a35e84c1b8e1aab2bcce9d46b). I think if we were using a more current TensorFlow, we wouldn't be seeing this.

To get rid of the warning, needed to set:

```text
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
```

In *dc-gann.py*.

### 4.2. learning_rate

```text
/mnt/arkk/rpm/skylines/skylines/.venv/lib/python3.8/site-packages/keras/optimizers/optimizer_v2/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
```

Easy enough.

### 4.3. Autoshard policy

```text
W tensorflow/core/grappler/optimizers/data/auto_shard.cc:784] AUTO sharding policy will apply DATA sharding policy as it failed to apply FILE sharding policy because of the following reason: Found an unshardable source dataset: name: "TensorDataset/_2"
```

Couldn't find a good solution for this one - luckily, it's just a warning and the model seems to train just fine anyway, though we might be leaving performance on the table. Couple of notes - first, seems like maybe it's not a shardable dataset because it's not 'file based' i.e. our data is not in chunks that can be split up. Instead, each file is a single image. Secondly, according to [this](https://github.com/tensorflow/tensorflow/issues/45157) we don't need to autoshard with a single machine using MirroredStrategy, it's meant for MultiWorkerMirroredStrategy across nodes. Lastly, I tried setting the autoshard policy to DATA or OFF from the get-go, but we still see the warning. Think we need to just ignore it and move on.

OK, I think that's it - turning warnings off via train.sh:

```text
export TF_CPP_MIN_LOG_LEVEL=2
```

## 5. Bind mount fast scratch

This one is pretty specific to our hardware configuration, but might be useful to others. Leaving it here as reference. We have a local NVMe SSD set-up as a fast scratch drive. Place all the data there and then use a bind mount to the project data directory. In */etc/fstab*:

```text
# Fast scratch bind mount for skylines project
/mnt/fast_scratch/skylines /mnt/arkk/rpm/skylines/skylines/skylines/data none x-systemd.requires=/mnt/fast_scratch,x-systemd.requires=/mnt/arkk,x-systemd.automount,bind 0 0
```

## 6. Removing large files from git tracking

```text
git filter-branch --tree-filter 'rm -f path/to/big/file' HEAD
```
