# 30mlp --- Multilayer Perceptron

# Get source code

```
git clone https://github.com/taura/parallel-distributed-programming-code-2024.git
```

You should get directory `parallel-distributed-programming-code-2024`

# Compile

```
cd parallel-distributed-programming-code-2024/30mlp
make
```

You should get `mlp_mnist_infer` in the directory.


# Run

```
./mlp_mnist_infer
```

will run the inference with the default setting, which is to process the 60000 samples in `/home/share/mnist-data/train_images_60000x784_float32.npy` (labels are in `/home/share/mnist-data/train_labels_60000_float32.npy`)

The output should look like this.

```
read data from: /home/share/mnist-data/train_images_60000x784_float32.npy
read labels from: /home/share/mnist-data/train_labels_60000_float32.npy
mini batch size: 256
max mini batch size: 60000
pixels per image: 784
hidden units: 400
output classes: 10
weight read from: /home/share/mnist-data/mlp-mnist-weight.bin
read weights from /home/share/mnist-data/mlp-mnist-weight.bin
reading data from /home/share/mnist-data/train_images_60000x784_float32.npy
read 60000 samples
reading data from /home/share/mnist-data/train_labels_60000_float32.npy
read 60000 samples
draw 60000 samples from [0:60000]
iterations start ...
[0 - 256] ... loss = 0.130931 accuracy = 0.960938
[256 - 512] ... loss = 0.116322 accuracy = 0.953125
[512 - 768] ... loss = 0.125852 accuracy = 0.968750
[768 - 1024] ... loss = 0.142651 accuracy = 0.960938

    ...

[59136 - 59392] ... loss = 0.072811 accuracy = 0.980469
[59392 - 59648] ... loss = 0.043428 accuracy = 0.988281
[59648 - 59904] ... loss = 0.181115 accuracy = 0.960938
[59904 - 60000] ... loss = 0.197375 accuracy = 0.979167
iterations end
60000 samples
loss avg = 0.117300 accuracy avg = 0.965117
23856000000 flops in 14.58120227 sec = 1.64 flops/nsec
```

The value of loss avg (0.117300) and accuracy avg (0.965117) should be exactly these values unless you give command line arguments that change which samples are processed.  See below for details.

# Other ways to run

* You can see available options by giving `--help` option (or any unrecognized option, for that matter)

```
$ ./mlp_mnist_infer --help
./mlp_mnist_infer: unrecognized option '--help'
usage:
  ./mlp_mnist_infer [options]
options:
  -s,--data-start S
    start picking data from S-th sample (default: 0)
  -e,--data-end E
    end picking data at E-th sample (default: -1)
  -n,--n-samples N
    use N samples (default: -1)
  -b,--batch-sz x
    test this many samples at a time. i.e., mini-batch size (default: 256)
  --input-weight F
    read weight from this file (default: /home/share/mnist-data/mlp-mnist-weight.bin)
  --data F
    read input images from this file (default: /home/share/mnist-data/train_images_60000x784_float32.npy)
  --label F
    read true label (class) from this file (default: /home/share/mnist-data/train_labels_60000_float32.npy)
  --verbose N
    show stats and progress (default: 1)
```

* When given `--n-samples N --batch-sz x` the first batch is always sample 0, 1, ..., x-1, the second batch is x, x+1, ..., 2x-1, until you reach the end (sample N-1), after which you wrap around and start processing sample 0.
* Of course, processing the same data twice in inference is meaningless, but this program is written to process arbitrary number of samples for the purpose of performance investigation.

# Your work

* Your job is to make it run faster
* You have to make
  * a program for multicores, using OpenMP
  * a program for GPU, using either an OpenMP or a CUDA
  * a program for CPU, using SIMD
* Optionally, you make
  * a program for CPU, using SIMD AND OpenMP

# Updates are coming 

* This is going to be your next assignment you have to submit (due is not decided yet, but never earlier than December 7th)
* Please anticipate more details will come later about how to submit your work and what must be included in it
* I release this baseline code now so that you can start working on it now and have work to do on Nov 18th, the day I have to cancel the class
* Please start working on it now (or anytime you think you are ready for it)
* Further updates will be given by updating this page
