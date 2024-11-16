# 30mlp --- Multilayer Perceptron

# Get source code

```
cd WHEREVER_YOU_WORK
git clone https://github.com/taura/parallel-distributed-programming-code-2024.git
```

You should get directory `parallel-distributed-programming-code-2024`

# Compile

```
# assuming you are in WHEREVER_YOU_WORK
cd parallel-distributed-programming-code-2024/30mlp
make
```

You should get `mlp_mnist_infer` in the directory.


# Run

```
# assuming you are in WHEREVER_YOU_WORK/parallel-distributed-programming-code-2024/30mlp
./mlp_mnist_infer
```

The output should look like this.

```
read data from: /home/share/mnist-data/train_images_60000x784_float32.npy//home/share/mnist-data/train_labels_60000_float32.npy [0 - -1]
mini batch size: 256
max mini batch size: 256
pixels per image: 784
hidden units: 200
output classes: 10
weight read from: /home/share/mnist-data/mlp-mnist-weight.bin
read weights from /home/share/mnist-data/mlp-mnist-weight.bin
reading data from /home/share/mnist-data/train_images_60000x784_float32.npy
read 60000 samples
reading data from /home/share/mnist-data/train_labels_60000_float32.npy
read 60000 samples
iterations start ...
[0 - 256] ... loss = 0.126792 accuracy = 0.968750
[256 - 512] ... loss = 0.091159 accuracy = 0.980469
[512 - 768] ... loss = 0.106709 accuracy = 0.964844
[768 - 1024] ... loss = 0.155397 accuracy = 0.941406
   ...
[59648 - 59904] ... loss = 0.152160 accuracy = 0.960938
[59904 - 60000] ... loss = 0.230854 accuracy = 0.989583
iterations end
60000 samples
loss avg = 0.111972 accuracy avg = 0.965267
23856000000 flops in 14.58120227 sec = 1.64 flops/nsec
```

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

# Do your work

* Your job is to make it run faster
* You have to make
  * a program for multicores, using OpenMP running on CPU
  * a program for GPU, using either an OpenMP or a CUDA
  * a program for CPU, using SIMD
* Optionally, you make
  * a program for CPU, using SIMD AND muticores

# Updates are coming 

* This is going to be your next assignment you have to submit
* Please anticipate more details will come later about how to submit your work and what must be included in it
* I release this now so that you can start working on it early and have work to do on Nov 18th, the day we would normally have a class but we actually don't
* Please start working on it now (or anytime you think you are ready for it)
