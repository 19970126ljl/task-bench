DEBUG ?= 0

# Compiler settings
NVCC ?= nvcc
CXX ?= g++

# CUDA STF is header-only, located in CCCL
CCCL_DIR ?= /root/stf_exp/cccl
CUDASTF_INCLUDE = -I$(CCCL_DIR)/cudax/include -I$(CCCL_DIR)/libcudacxx/include

# Basic flags
NVCCFLAGS = -std=c++17 --expt-relaxed-constexpr --extended-lambda -arch=sm_86
CXXFLAGS = -std=c++17 -Wall
LDFLAGS = -lcuda -lcudart

ifeq ($(strip $(DEBUG)),1)
NVCCFLAGS += -g -G -O0
CXXFLAGS += -g -O0
LDFLAGS += -g
else
NVCCFLAGS += -O3
CXXFLAGS += -O3 -march=native
LDFLAGS += -O3
endif

# Include directories
INC = -I../core $(CUDASTF_INCLUDE)
INC_EXT = 

# Location of the libraries
LIB = -L../core -lcore_s
LIB_EXT = -L/opt/nvidia/hpc_sdk/Linux_x86_64/25.3/cuda/12.8/targets/x86_64-linux/lib

INC := $(INC) $(INC_EXT)
LIB := $(LIB) $(LIB_EXT)

NVCCFLAGS += $(INC)
CXXFLAGS += $(INC)

include ../core/make_blas.mk

TARGET = main
all: $(TARGET)

.PRECIOUS: %.cu %.o

main.o: main.cu ../core/timer.h
	$(NVCC) -c $(NVCCFLAGS) $<

main: main.o
	$(CXX) $^ $(LIB) $(LDFLAGS) -o $@

clean:
	rm -f *.o
	rm -f $(TARGET)

.PHONY: all clean
