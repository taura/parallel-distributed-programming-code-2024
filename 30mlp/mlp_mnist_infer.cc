/*
 * mlp.cc
 * multilayer perceptron
 */

#include <assert.h>
#include <err.h>
#include <getopt.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>
#include <unistd.h>

/* parameters determined by the data you are dealing with.
   you probably never need to change */

/* settings for tiny MNIST dataset */
enum {
  n_classes    = 10,
  image_height = 28,
  image_width  = 28,
  image_n_channels = 1,
  n_pixels     = image_height * image_width * image_n_channels,
  max_n_data   = 60000, // maximum number of samples you use with this program
};

/* parameters you might want to change to optimize 
   learning perofmance and/or optimize speed, yet
   want to make compile-time constants for performance.
   you can change them below or by giving -Dn_units=X 
   and/or -Dbatch_sz=X on the command to compile this file
   (e.g., clang++ -Dn_units=300 -Dmax_batch_sz=256). */

/* number of units in the hidden layer */
#ifndef n_units
#define n_units 200
#endif

/* the maximum number of samples in a single 
   mini batch (you can adjust the actual number
   of samples at runtime without changing this value) */
#ifndef max_batch_sz
#define max_batch_sz 256
#endif

typedef float real;

/* -----------------------
 * random number generator
 * ----------------------- */
typedef struct  {
  uint64_t x;
} prng_t;

void prng_init(prng_t * rg, uint64_t seed) {
  const uint64_t mask = (1UL << 48) - 1;
  rg->x = seed & mask;
}

/* return a random number 0 <= x < 2^48 */
uint64_t prng_rand(prng_t * rg) {
  const uint64_t a = 0x5deece66dull;
  const uint64_t c = 0xb;
  const uint64_t mask = (1UL << 48) - 1;
  uint64_t x = rg->x;
  uint64_t next = (a * x + c) & mask;
  rg->x = next;
  return next;
}

/* return a random number 0 <= x < L */
uint64_t prng_rand_int(prng_t * rg, uint64_t L) {
  return prng_rand(rg) % L;
}

/* return a random number 0 <= x < 1.0 */
double prng_rand_double(prng_t * rg) {
  return prng_rand(rg) / (double)(1UL << 48);
}

/* draw a random number from normal distribution */
real gen_normal(prng_t * rg, real mu, real sigma) {
  real u = prng_rand_double(rg);
  real v = prng_rand_double(rg);
  real x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v);
  return mu + x * sigma;
}

/* -----------------------
 * get current time in sec
 * ----------------------- */
double get_time() {
  struct timespec ts[1];
  clock_gettime(CLOCK_REALTIME, ts);
  return ts->tv_sec + ts->tv_nsec * 1.0e-9;
}

/* -----------------------
 * command line options
 * ----------------------- */
struct cmdline_opt {
  /* start/end position of data */
  long data_start;
  long data_end;
  /* number of samples */
  long n_samples;
  /* batch size */
  long batch_sz;
  /* filenames for weight and data */
  char * input_weight;
  char * data;
  char * label;
  int verbose;
  
  cmdline_opt() {
    /* default values of command line options */
    data_start = 0;
    data_end = -1;
    n_samples = -1;
    batch_sz = max_batch_sz;
    input_weight = 0;
    data = strdup("/home/share/mnist-data/train_images_60000x784_float32.npy");
    label = strdup("/home/share/mnist-data/train_labels_60000_float32.npy");
    verbose = 1;
  }
  void fini() {
    free(data);
    free(label);
    if (input_weight) free(input_weight);
  }
};

void usage(const char * prog) {
  cmdline_opt opt;
  fprintf(stderr,
	  "usage:\n"
	  "  %s [options]\n", prog);
  fprintf(stderr,
	  "options:\n"
	  "  -s,--data-start S\n"
	  "    start picking data from S-th sample (default: %ld)\n"
	  "  -e,--data-end E\n"
	  "    end picking data at E-th sample (default: %ld)\n"
	  "  -n,--n-samples N\n"
	  "    use N samples (default: %ld)\n"
	  "  -b,--batch-sz x\n"
	  "    test this many samples at a time. i.e., mini-batch size (default: %ld)\n"
	  "  --input-weight F\n"
	  "    read weight from this file (default: %s)\n"
	  "  --data F\n"
	  "    read input images from this file (default: %s)\n"
	  "  --label F\n"
	  "    read true label (class) from this file (default: %s)\n"
	  "  --verbose N\n"
	  "    show stats and progress (default: %d)\n"
	  ,
	  opt.data_start,
	  opt.data_end,
	  opt.n_samples,
	  opt.batch_sz,
	  (opt.input_weight ? opt.input_weight : ""),
	  opt.data,
	  opt.label,
          opt.verbose
          );
  opt.fini();
}

cmdline_opt parse_opt(int argc, char ** argv) {
  cmdline_opt opt;
  static struct option long_options[] = {
    { "data-start",          required_argument, 0, 's' },
    { "data-end",            required_argument, 0, 'e' },
    { "n-samples",           required_argument, 0, 'n' },
    { "batch-sz",            required_argument, 0, 'b' },
    { "input-weight",        required_argument, 0, 0 },
    { "data",                required_argument, 0, 0 },
    { "label",               required_argument, 0, 0 },
    { "verbose",             required_argument, 0, 0 },
  };
  while (1) {
    int option_index = 0;
    int c = getopt_long(argc, argv, "s:e:n:b:",
			long_options, &option_index);
    if (c == -1) break;
    switch (c) {
    case 's':
      opt.data_start = atol(optarg);
      break;
    case 'e':
      opt.data_end = atol(optarg);
      break;
    case 'n':
      opt.n_samples = atol(optarg);
      break;
    case 'b':
      opt.batch_sz = atol(optarg);
      break;
    case 0:
      {
        const char * name = long_options[option_index].name;
        if (strcmp(name, "input-weight") == 0) {
          if (opt.input_weight) { printf("free %s\n", opt.input_weight); free(opt.input_weight); }
          opt.input_weight = strdup(optarg);
        } else if (strcmp(name, "data") == 0) {
          if (opt.data) { printf("free %s\n", opt.data); free(opt.data); }
          opt.data = strdup(optarg);
        } else if (strcmp(name, "label") == 0) {
          if (opt.label) { printf("free %s\n", opt.label); free(opt.label); }
          opt.label = strdup(optarg);
        } else if (strcmp(name, "verbose") == 0) {
          opt.verbose = atoi(optarg);
        } else {
          usage(argv[0]);
          exit(1);
        }
        break;
      }
    default:
      usage(argv[0]);
      exit(1);
      break;
    }
  }
  return opt;
}

/* -------------------------------
 * matrix class
 * M : the (maximum) number of rows
 * N : the number of columns
 * the number of rows can be set dynamically, but
 * always allocate M x N array
 * ------------------------------- */
template<long M,long N>
struct mat {
  long m;			/* the actual number of rows used <= M */
  real a[M][N];
  mat() { m = M; }
  void set_rows(long m) {
    this->m = m;
  }
  real& operator()(int i, int j) {
    assert(0 <= i);
    assert(i < m);
    assert(0 <= j);
    assert(j < N);
    return a[i][j];
  }
  /* initialize weights randomly */
  void random_init_weight(prng_t * rg) {
    mat<M,N>& x = *this;
    assert(m == M);
    real q = sqrt(1.0/N);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N; j++) {
        x(i,j) = gen_normal(rg, 0.0, q);
      }
    }
  }
  /* read weight from file (fp) */
  void read_weight(FILE * fp) {
    long m = 0;
    if (fread(&m, sizeof(m), 1, fp) != 1) err(1, "fread");
    assert(this->m == m);
    if (fread(a, sizeof(real) * m * N, 1, fp) != 1) err(1, "fread");
  }
  /* write weight to file (fp) */
  void write_weight(FILE * fp) {
    long m = this->m;
    if (fwrite(&m, sizeof(m), 1, fp) != 1) err(1, "fread");
    if (fwrite(a, sizeof(real) * m * N, 1, fp) != 1) err(1, "fread");
  }
};


/* ---------------------------------
 * Fc (fully connected) layer
 * input x :  (M x K) matrix
 * weight W : (K x N) matrix
 * output y = x @ W : (M x N) matrix
 *
 * M is the (maximum) number of samples in a batch
 * --------------------------------- */
template<long M,long K,long N>
struct FC {
  mat<K,N> W;    // learned weight
  mat<M,N> y;    // forward output
  mat<M,N>& forward(mat<M,K>& x) {
    /* y = x @ W */
    long m = x.m;
    assert(0 < m);
    assert(m <= M);
    assert(W.m == K);
    y.set_rows(m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N; j++) {
        y(i,j) = 0;
        for (int k = 0; k < K; k++) {
          y(i,j) += x(i,k) * W(k,j);
        }
      }
    }
    return y;
  }
};

/* ------------------------------
 * Relu
 * input x : (M x N)
 * output y = max(0, x) : (M x N)
 * ------------------------------ */
template<long M,long N>
struct Relu {
  mat<M,N> y;                   // output
  mat<M,N>& forward(mat<M,N>& x) {
    /* y = max(0, x) */
    long m = x.m;
    assert(0 < m);
    assert(m <= M);
    y.set_rows(m);
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < N; j++) {
        y(i,j) = x(i,j) > 0.0 ? x(i,j) : 0.0;
      }
    }
    return y;
  }
};

/* ------------------------------
 * SoftmaxCrossEntropy
 * input x : (M x N)
 * output y = CrossEntropy(c, Softmax(x))
 * where
 *  CrossEntropy(p, q) = - ∑_i p[i] log(q[i]),
 *  c is a one-hot vector to represent the correct label, and
 *  Softmax(x) = {exp(x[i])}_i / S, where S = ∑_j exp(x[j])
 *
 * when c is a one-hot vector for class t (i.e., whose only t-th
 * element is one), 
 *   CrossEntropy(c, q) = - log(q_t)
 * therefore,
 *   CrossEntropy(c, Softmax(x)) = log(exp(x[t]) / S) = - x[t] + log S
 *
 * ------------------------------ */
template<long M,long N>
struct SoftmaxCrossEntropy {
  mat<M,N> lsm;                 // store log(softmax(x))
  mat<M,1> y;                   // output 
  mat<M,N>& logsoftmax(mat<M,N> x) {
    long m = x.m;
    assert(N > 0);
    assert(0 < m);
    assert(m <= M);
    lsm.set_rows(m);
    for (int i = 0; i < m; i++) {
      /* t = argmax_j x(i,j) */
      int t = 0;
      for (int j = 1; j < N; j++) {
        t = (x(i,t) < x(i,j) ? j : t);
      }
      double s = 0.0;
      for (int j = 0; j < N; j++) {
        lsm(i,j) = x(i,j) - x(i,t);
        s += exp(lsm(i,j));
      }
      for (int j = 0; j < N; j++) {
        lsm(i,j) -= log(s);
      }
    }
    return lsm;
  }
  mat<M,1>& forward(mat<M,N>& x, mat<M,1>& c) {
    /* y = cross_entropy(1_c, softmax(x)),
       where 1_c is the one-hot vector having the one
       at its c-th element */
    long m = x.m;
    assert(N > 0);
    assert(0 < m);
    assert(m <= M);
    y.set_rows(m);
    mat<M,N>& lsm = logsoftmax(x);
    for (long i = 0; i < m; i++) {
      int t = c(i,0);
      y(i,0) = -lsm(i,t);
    }
    return y;
  }
};

/* a trivial data structure that represents
   the total loss in and
   the number of correctly classified samples
   in a batch */
struct exec_result {
  double loss;  // sum of losses in a batch
  long correct; // count of correctly classified samples in a batch
  exec_result(double loss, long correct) {
    this->loss = loss;
    this->correct = correct;
  }
};

/* Multi-Layer Perceptron
      F0 pixels   -> FC + Relu
   -> F1 features -> FC + Relu
   -> F1 features -> FC
   -> F2 class probability
 */ 
template<long M,long F0,long F1,long F2>
struct MLP {
  mat<M,F0>   x;               /* input to the whole MLP */
  mat<M,1>    c;               /* true class for each sample */
  FC<M,F0,F1> fc0;             /* first FC */
  Relu<M,F1>  relu0;           /* first Relu */ 
  FC<M,F1,F1> fc1;             /* second FC */
  Relu<M,F1>  relu1;           /* second Relu */ 
  FC<M,F1,F2> fc2;             /* last FC */
  SoftmaxCrossEntropy<M,F2> smxe;
  mat<M,1>    g_y;
  long count_correct(mat<M,F2>& x, mat<M,1>& c) {
    /* x(i,j) is the probability sample i belongs to class j.
       c(i) is the true label of sample i.
       count the number of correctly classified samples
       i.e., count i s.t., c(i) = argmax_j x(i,j) */
    long m = x.m;
    assert(F2 > 0);
    assert(0 < m);
    assert(m <= M);
    long correct = 0;
    for (int i = 0; i < m; i++) {
      /* t = argmax_j x(i,j) */
      int t = 0;
      for (int j = 1; j < F2; j++) {
        if (x(i,j) > x(i,t)) {
          t = j;
        }
      }
      correct += (c(i,0) == t);
    }
    return correct;
  }
  double sum_loss(mat<M,1>& e) {
    /* sum of e(i) */
    long m = e.m;
    double s = 0.0;
    for (int i = 0; i < m; i++) {
      s += e(i, 0);
    }
    return s;
  }
  exec_result forward() {
    /* forward phase of the whole MLP */
    mat<M,F1>& x1 = fc0.forward(x);
    mat<M,F1>& x2 = relu0.forward(x1);
    mat<M,F1>& x3 = fc1.forward(x2);
    mat<M,F1>& x4 = relu1.forward(x3);
    mat<M,F2>& x5 = fc2.forward(x4);
    mat<M,1>&   e = smxe.forward(x5, c);
    double      l = sum_loss(e);
    long        o = count_correct(x5, c);
    return exec_result(l, o);
  }
};

/* read the data file (npy format) */
template<int M,int N>
mat<M,N> * read_data(const char * filename) {
  printf("reading data from %s\n", filename);
  const off_t header_sz = 80;
  char header[header_sz];
  FILE * fp = fopen(filename, "rb");
  if (!fp) err(1, "fopen(%s)", filename);
  /* read the header (check a few bytes I know what they should be) */
  if (fread(header, header_sz, 1, fp) != 1) err(1, "fread");
  assert(strncmp(header + 1, "NUMPY", 5) == 0);
  assert(header[header_sz - 1] == '\n');
  /* make a matrix and really read data into it */
  mat<M,N> * A = new mat<M,N>();
  ssize_t m = fread(A->a, sizeof(real) * N, M, fp);
  if (m == -1) err(1, "fread");
  printf("read %ld samples\n", m);
  A->m = m;
  /* make sure we are at the end of the file */
  char dummy[1];
  ssize_t n = fread(dummy, sizeof(dummy), 1, fp);
  if (n == -1) err(1, "fread");
  if (n > 0) {
    fprintf(stderr,
            "WARNING: some data left unread in file %s, due to max_n_data (%d)"
            " consider setting max_n_data in the program to use those data\n",
            filename, max_n_data);
  }
  fclose(fp);
  return A;
}

/* initialize weight randomly */
template<long M,long F0,long F1,long F2>
void init_weight(MLP<M,F0,F1,F2> * mlp, uint64_t seed) {
  prng_t rg[1] = { seed };
  mlp->fc0.W.random_init_weight(rg);
  mlp->fc1.W.random_init_weight(rg);
  mlp->fc2.W.random_init_weight(rg);
}

/* read weight from the file */
template<long M,long F0,long F1,long F2>
void read_weight(MLP<M,F0,F1,F2> * mlp, char * weight_bin) {
  printf("read weights from %s\n", weight_bin);
  FILE * fp = fopen(weight_bin, "rb");
  if (!fp) err(1, "fopen(%s)", weight_bin);
  mlp->fc0.W.read_weight(fp);
  mlp->fc1.W.read_weight(fp);
  mlp->fc2.W.read_weight(fp);
  /* make sure we are at the end of the file */
  char dummy[1];
  ssize_t n = fread(dummy, sizeof(dummy), 1, fp);
  assert(n == 0); (void)n;
  fclose(fp);
}


/* select (b - a) samples either randomly (when rg is not null)
   or deterministically (when rg is null)
   and put them in samples.
   each sample is a number between in [start,end) */
void select_rows(long a, long b, long start, long end,
                 int * samples) {
  for (long i = 0; i < b - a; i++) {
    samples[i] = start + (a + i) % (end - start);
  }
}

/* get a[samples[i],:] for i in 0..m and put them in b */
template<long M,long B,long N>
void get_select_rows(mat<M,N>& a, long m, int * samples, mat<B,N>& b) {
  b.set_rows(m);
  for (int i = 0; i < m; i++) {
    int idx = samples[i];
    for (int j = 0; j < N; j++) {
      b(i,j) = a(idx,j);
    }
  }
}

/*
 * train mlp
 */
template<long M,long F0,long F1,long F2>
void train(MLP<M,F0,F1,F2> * mlp,
           mat<max_n_data,F0> * X, mat<max_n_data,1> * C,
           cmdline_opt opt) {
  long bs = opt.batch_sz;
  long start = opt.data_start;
  long end = opt.data_end;
  long n_samples = opt.n_samples;
  int samples[bs];
  exec_result tot = {0.0, 0};
  printf("iterations start ...\n");
  double t0 = get_time();
  for (long i = 0; i < n_samples; i += bs) {
    /* get samples */
    long m = (i + bs <= n_samples ? bs : n_samples - i);
    if (opt.verbose) {
      printf("[%ld - %ld] ... ", i, i + m); fflush(stdout);
    }
    select_rows(i, i + m, start, end, samples);
    get_select_rows(*X, m, samples, mlp->x);
    get_select_rows(*C, m, samples, mlp->c);
    exec_result r = mlp->forward();
    tot.loss += r.loss;
    tot.correct += r.correct;
    if (opt.verbose) {
      printf("loss = %f accuracy = %f\n",
             r.loss / (double)m, r.correct / (double)m);
    }
  }
  /* report the performance of all test data */
  double t1 = get_time();
  double dt = t1 - t0;
  long flops = 2 * n_samples * (F0 * F1 + F1 * F1 + F1 * F2);
  printf("iterations end\n");
  printf("%ld samples\n", n_samples);
  printf("loss avg = %f accuracy avg = %f\n",
         tot.loss / (double)n_samples, tot.correct / (double)n_samples);
  printf("%ld flops in %.8f sec = %.2f flops/nsec\n",
	 flops, dt, flops / dt * 1.0e-9);
}

/* the main function */
int main(int argc, char ** argv) {
  /* parse command line options (set various options) */
  cmdline_opt opt = parse_opt(argc, argv);
  /* the three matrices in the neural network */
  MLP<max_batch_sz,n_pixels,n_units,n_classes> * mlp
    = new MLP<max_batch_sz,n_pixels,n_units,n_classes>();

  printf("read data from: %s/%s [%ld - %ld]\n",
         opt.data, opt.label, opt.data_start, opt.data_end);
  printf("mini batch size: %ld\n", opt.batch_sz);
  printf("max mini batch size: %d\n", max_batch_sz);
  printf("pixels per image: %d\n", n_pixels);
  printf("hidden units: %d\n", n_units);
  printf("output classes: %d\n", n_classes);
  printf("weight read from: %s\n",
         (opt.input_weight ? opt.input_weight : ""));
  read_weight(mlp, opt.input_weight);
  /* read input images and labels */
  mat<max_n_data,n_pixels> * X = read_data<max_n_data,n_pixels>(opt.data);
  mat<max_n_data,1>        * C = read_data<max_n_data,1>(opt.label);
  if (opt.data_end == -1) opt.data_end = X->m;
  if (opt.n_samples == -1) opt.n_samples = opt.data_end - opt.data_start;

  if (opt.batch_sz > max_batch_sz) {
    printf("error: batch size (%ld) must be <= max_batch_sz (%d)\n",
           opt.batch_sz, max_batch_sz);
    printf("error: consider changing max_batch_sz in the program"
           " to specify that large batch size\n");
    exit(1);
  }
  /* ------------- do the main job ------------- */
  train<max_batch_sz,n_pixels,n_units,n_classes>(mlp, X, C, opt);
  delete X;
  delete C;
  delete mlp;
  opt.fini();
  return 0;
}
    
