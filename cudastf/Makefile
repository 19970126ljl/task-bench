DEBUG ?= 0

CCCL_ROOT ?= ../../cccl
NVCC ?= nvcc
HOST_CC ?= gcc
HOST_CXX ?= g++
AR ?= ar
CUDA_ARCH ?= native
CUDA_NATIVE_CC = $(shell nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
ifeq ($(CUDA_ARCH),native)
CUDA_ARCH_RESOLVED ?= $(if $(CUDA_NATIVE_CC),sm_$(CUDA_NATIVE_CC),native)
else
CUDA_ARCH_RESOLVED ?= $(CUDA_ARCH)
endif

TARGET = task_bench
UNIT_TEST = $(BUILD_DIR)/unit_test
SOURCES = main.cu arguments.cc data_access.cc expanded_dag.cc identity.cc \
	  results.cc workload.cc
OBJECTS = $(SOURCES:.cu=.o)
OBJECTS := $(OBJECTS:.cc=.o)
BACKEND_OBJECTS = $(OBJECTS) unit_test.o
BACKEND_DEPFILES = $(BACKEND_OBJECTS:.o=.d)
BUILD_DIR = .build
CORE_CXX_SOURCES = ../core/core.cc ../core/core_c.cc \
		   ../core/core_kernel.cc ../core/timer.cc
CORE_C_SOURCES = ../core/core_random.c ../core/siphash.c
CORE_CXX_OBJECTS = $(patsubst ../core/%.cc,$(BUILD_DIR)/core/%.o,$(CORE_CXX_SOURCES))
CORE_C_OBJECTS = $(patsubst ../core/%.c,$(BUILD_DIR)/core/%.o,$(CORE_C_SOURCES))
CORE_OBJECTS = $(CORE_CXX_OBJECTS) $(CORE_C_OBJECTS)
CORE_LIB = $(BUILD_DIR)/libcore.a
CORE_HEADERS = $(wildcard ../core/*.h)
BUILD_CONFIG_STAMP = $(BUILD_DIR)/build-config

CCCL_REVISION ?= $(shell git -C $(CCCL_ROOT) rev-parse --short HEAD 2>/dev/null || echo unknown)
TASKBENCH_REVISION ?= $(shell git -C .. rev-parse --short HEAD 2>/dev/null || echo unknown)
CCCL_WORKTREE_DIRTY ?= $(shell test -n "$$(git -C $(CCCL_ROOT) status --porcelain --untracked-files=normal 2>/dev/null)" && echo 1 || echo 0)
TASKBENCH_WORKTREE_DIRTY ?= $(shell test -n "$$(git -C .. status --porcelain --untracked-files=normal 2>/dev/null)" && echo 1 || echo 0)

NVCCFLAGS = -std=c++17 -arch=$(CUDA_ARCH) \
	    --expt-relaxed-constexpr --extended-lambda \
	    -I../core -I$(CCCL_ROOT)/cudax/include \
	    -I$(CCCL_ROOT)/libcudacxx/include
BACKEND_CXXFLAGS = -std=c++17 -Wall -Wextra -I../core
CORE_CFLAGS = -std=c11 -fPIC -I../core
CORE_CXXFLAGS = -std=c++11 -fPIC -I../core
LDFLAGS = $(CORE_LIB) -lcuda

ifeq ($(strip $(DEBUG)),1)
NVCCFLAGS += -O0 -g -G
BACKEND_CXXFLAGS += -O0 -g
CORE_CFLAGS += -O0 -g -DDEBUG_CORE
CORE_CXXFLAGS += -O0 -g -DDEBUG_CORE
BUILD_TYPE = debug
else
NVCCFLAGS += -O3
BACKEND_CXXFLAGS += -O3
CORE_CFLAGS += -O3
CORE_CXXFLAGS += -O3
BUILD_TYPE = release
endif

BUILD_METADATA_FLAGS = \
	-DTASKBENCH_REVISION='"$(TASKBENCH_REVISION)"' \
	-DTASKBENCH_WORKTREE_DIRTY=$(TASKBENCH_WORKTREE_DIRTY) \
	-DTASKBENCH_CCCL_REVISION='"$(CCCL_REVISION)"' \
	-DTASKBENCH_CCCL_WORKTREE_DIRTY=$(CCCL_WORKTREE_DIRTY) \
	-DTASKBENCH_BUILD_TYPE='"$(BUILD_TYPE)"' \
	-DTASKBENCH_CUDA_ARCH='"$(CUDA_ARCH)"' \
	-DTASKBENCH_CUDA_ARCH_RESOLVED='"$(CUDA_ARCH_RESOLVED)"'

NVCCFLAGS += $(BUILD_METADATA_FLAGS)
BACKEND_CXXFLAGS += $(BUILD_METADATA_FLAGS)

.PHONY: all clean test FORCE

all: $(TARGET)

FORCE:

test: $(TARGET) $(UNIT_TEST)
	$(UNIT_TEST)
	./test.sh

$(OBJECTS) unit_test.o $(CORE_OBJECTS): $(BUILD_CONFIG_STAMP)
results.o: FORCE

$(BUILD_CONFIG_STAMP): FORCE
	@mkdir -p $(@D)
	@printf '%s\n' \
	  'DEBUG=$(DEBUG)' \
	  'CUDA_ARCH=$(CUDA_ARCH)' \
	  'CUDA_ARCH_RESOLVED=$(CUDA_ARCH_RESOLVED)' \
	  'NVCC=$(NVCC)' \
	  'HOST_CC=$(HOST_CC)' \
	  'HOST_CXX=$(HOST_CXX)' \
	  'CCCL_ROOT=$(CCCL_ROOT)' >$@.tmp
	@if test -r $@ && cmp -s $@.tmp $@; then \
	  rm -f $@.tmp; \
	else \
	  mv -f $@.tmp $@; \
	fi

$(TARGET): $(OBJECTS) $(CORE_LIB)
	$(NVCC) $(NVCCFLAGS) $(OBJECTS) $(LDFLAGS) -o $@

%.o: %.cu Makefile
	$(NVCC) $(NVCCFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

%.o: %.cc Makefile
	$(HOST_CXX) $(BACKEND_CXXFLAGS) -MMD -MP -MF $(@:.o=.d) -c $< -o $@

$(BUILD_DIR)/core/%.o: ../core/%.cc $(CORE_HEADERS)
	@mkdir -p $(@D)
	$(HOST_CXX) $(CORE_CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/core/%.o: ../core/%.c $(CORE_HEADERS)
	@mkdir -p $(@D)
	$(HOST_CC) $(CORE_CFLAGS) -c $< -o $@

$(CORE_LIB): $(CORE_OBJECTS)
	@mkdir -p $(@D)
	$(AR) rcs $@ $^

$(UNIT_TEST): unit_test.o arguments.o data_access.o expanded_dag.o identity.o \
	      results.o workload.o $(CORE_LIB)
	@mkdir -p $(@D)
	$(HOST_CXX) $(BACKEND_CXXFLAGS) $^ -o $@

clean:
	rm -f *.o *.d $(TARGET)
	rm -rf $(BUILD_DIR)

-include $(BACKEND_DEPFILES)
