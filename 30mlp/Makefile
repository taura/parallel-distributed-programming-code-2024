srcs := mlp_mnist_infer.cc
exes := $(patsubst %.cc,%,$(srcs))

opts := 
#opts += -O0 -g
opts += -O3
# add (uncomment) this option to get rid of array bounds checking
# opts += -DNDEBUG
opts += -Wall
opts += -Wextra

libs :=

all : $(exes)

$(exes) : % : %.cc
	g++ -o $@ $< $(opts) $(libs)

clean :
	rm -f $(exes)
