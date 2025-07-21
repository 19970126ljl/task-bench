/* Copyright 2024 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cuda/experimental/stf.cuh>
#include <stdarg.h>
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <cassert>

#include "core.h"
#include "timer.h"
// #include <nvtx3/nvToolsExt.h>
#include <nvtx3/nvtx3.hpp>
#define VERBOSE_LEVEL 0
#define MAX_NUM_ARGS 10
#define MAX_WIDTH 1024
#define USE_CORE_VERIFICATION

using namespace cuda::experimental::stf;

// CUDA kernels for different Task Bench kernel types
__global__ void empty_kernel() {
  // Do nothing - just for timing overhead measurement
}

__global__ void compute_kernel(slice<char> output, long iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Simple compute-bound work
    double acc = 1.2345;
    for (long iter = 0; iter < iterations; iter++) {
      acc = acc * acc + acc;
    }
    // Store result to prevent optimization
    if (output.size() >= sizeof(double)) {
      *reinterpret_cast<double*>(output.data_handle()) = acc;
    }
  }
}

__global__ void memory_kernel(slice<char> output, slice<char> scratch, long iterations, long timestep) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  // Memory-bound work - copy data patterns
  for (long iter = 0; iter < iterations; iter++) {
    for (size_t i = tid; i < scratch.size(); i += stride) {
      size_t idx = (i + timestep * iter) % scratch.size();
      if (idx < scratch.size() && i < output.size()) {
        output(i) = scratch(idx);
      }
    }
  }
}

__global__ void busy_wait_kernel(slice<char> output, long iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Busy wait computation similar to CPU version
    long long acc = 113;
    for (long iter = 0; iter < iterations; iter++) {
      acc = acc * 139 % 2147483647;
    }
    // Store result to prevent optimization
    if (output.size() >= sizeof(long long)) {
      *reinterpret_cast<long long*>(output.data_handle()) = acc;
    }
  }
}

__global__ void compute2_kernel(slice<char> output, long iterations) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid == 0) {
    // Compute2 kernel - similar to CPU version
    constexpr size_t N = 32;
    double A[N], B[N], C[N];

    for (size_t i = 0; i < N; ++i) {
      A[i] = 1.2345;
      B[i] = 1.010101;
      C[i] = 0.0;
    }

    for (long iter = 0; iter < iterations; iter++) {
      for (size_t i = 0; i < N; ++i) {
        C[i] = C[i] + (A[i] * B[i]);
      }
    }

    // Store result
    if (output.size() >= sizeof(double)) {
      double sum = 0;
      for (size_t i = 0; i < N; ++i) {
        sum += C[i];
      }
      *reinterpret_cast<double*>(output.data_handle()) = sum;
    }
  }
}

typedef struct tile_s {
  float dep;
  char *output_buff;
  // constructors for different access patterns
  tile_s(float dep, char *output_buff) : dep(dep), output_buff(output_buff) {}  // for write access
  tile_s(float dep, const char *output_buff) : dep(dep), output_buff(const_cast<char*>(output_buff)) {}  // for read access
}tile_t;

typedef struct payload_s {
  int x;
  int y;
  TaskGraph graph;
}payload_t;

typedef struct task_args_s {
  int x;
  int y;
}task_args_t;


typedef struct matrix_s {
  tile_t *data;
  int M;
  int N;
  std::vector<logical_data<slice<char, 1>>> handles;
}matrix_t;


// CUDA STF App implementation
struct CudaSTFApp : public App {
  CudaSTFApp(int argc, char **argv);
  ~CudaSTFApp();

  void execute_main_loop();
  void execute_timestep(size_t idx, long t);

private:
  void insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id);
  void debug_printf(int verbose_level, const char *format, ...);

private:
  // STF context
  context ctx;
  matrix_t mat_array[10];
  
};
char **extra_local_memory;

CudaSTFApp::CudaSTFApp(int argc, char **argv)
  : App(argc, argv)
{
  // printf("DEBUG: Initializing CUDA STF App with %zu graphs\n", graphs.size());

  size_t max_scratch_bytes_per_task = 0;
  matrix_t *matrix = mat_array;
  for (unsigned i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * matrix[i].M * matrix[i].N);

    matrix[i].handles.resize(matrix[i].M * matrix[i].N);
    
    {
    nvtx3::scoped_range r{"Data Prepare"};
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      matrix[i].data[j].output_buff = (char *)malloc(sizeof(char) * graph.output_bytes_per_task);
      // register logical data handles
      matrix[i].handles[j] = ctx.logical_data(make_slice(matrix[i].data[j].output_buff, graph.output_bytes_per_task));
    }
    }
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    // execute empty task
    ctx.task(exec_place::host()).set_symbol("empty-task")->*[=](cudaStream_t stream) {
        printf("empty");
    };
    ctx.host_launch().set_symbol("empty-task")->*[=] {
        nvtx3::scoped_range r{"Data Prepare"};
        printf("empty");
    };
    
    // printf("graph id %d, M = %d, N = %d, data %p, nb_fields %d\n", i, matrix[i].M, matrix[i].N, matrix[i].data, graph.nb_fields);
  }
  
  extra_local_memory = (char**)malloc(sizeof(char*) * MAX_WIDTH);
  assert(extra_local_memory != NULL);
  for (int k = 0; k < MAX_WIDTH; k++) {
    if (max_scratch_bytes_per_task > 0) {
      extra_local_memory[k] = (char*)malloc(sizeof(char)*max_scratch_bytes_per_task);
      TaskGraph::prepare_scratch(extra_local_memory[k], sizeof(char)*max_scratch_bytes_per_task);
    } else {
      extra_local_memory[k] = NULL;
    }
  }
 
}


// Destructor
CudaSTFApp::~CudaSTFApp() {
  // printf("DEBUG: Finalizing CUDA STF context\n");
  // ctx.finalize();
  matrix_t *matrix = mat_array;
  for (unsigned i = 0; i < graphs.size(); i++) {
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      free(matrix[i].data[j].output_buff);
      matrix[i].data[j].output_buff = NULL;
    }
    free(matrix[i].data);
    matrix[i].data = NULL;
  }
  
//   free(matrix);
//   matrix = NULL;
  
  for (int j = 0; j < MAX_WIDTH; j++) {
    if (extra_local_memory[j] != NULL) {
      free(extra_local_memory[j]);
      extra_local_memory[j] = NULL;
    }
  }
  free(extra_local_memory);
  extra_local_memory = NULL;

  // printf("DEBUG: CUDA STF cleanup complete\n");
}

static inline void task1(tile_t *tile_out, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_out->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = 0;
  printf("Task1 tid %d, x %d, y %d, out %f\n", payload.x, payload.x, payload.y, tile_out->dep);
#endif
}
static inline void task2(tile_t *tile_out, tile_t *tile_in1, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + 1;
  printf("Task2 tid %d, x %d, y %d, out %f, in1 %f\n", payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep);
#endif
}
static inline void task3(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                     input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + 1;
  printf("Task3 tid %d, x %d, y %d, out %f, in1 %f, in2 %f\n", payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep);
#endif
}
static inline void task4(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + 1;
  printf("Task4 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f\n", payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep);
#endif
}

static inline void task5(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + 1;
  printf("Task5 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f\n", payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep);
#endif
}

static inline void task6(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, tile_t *tile_in5, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in5->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + tile_in5->dep + 1;
  printf("Task6 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f, in5 %f\n", payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep, tile_in5->dep);
#endif
}

static inline void task7(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, tile_t *tile_in5, tile_t *tile_in6, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in5->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in6->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + tile_in5->dep + tile_in6->dep + 1;
  printf("Task7 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f, in5 %f, in6 %f\n",
    payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep, tile_in5->dep, tile_in6->dep);
#endif
}

static inline void task8(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, tile_t *tile_in5, tile_t *tile_in6, tile_t *tile_in7, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in5->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in6->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in7->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + tile_in5->dep + tile_in6->dep + tile_in7->dep + 1;
  printf("Task8 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f, in5 %f, in6 %f, in7 %f\n",
    payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep, tile_in5->dep, tile_in6->dep, tile_in7->dep);
#endif
}

static inline void task9(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, tile_t *tile_in5, tile_t *tile_in6, tile_t *tile_in7, tile_t *tile_in8, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in5->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in6->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in7->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in8->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + tile_in5->dep + tile_in6->dep + tile_in7->dep + tile_in8->dep + 1;
  printf("Task9 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f, in5 %f, in6 %f, in7 %f, in8 %f\n",
    payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep, tile_in5->dep, tile_in6->dep, tile_in7->dep, tile_in8->dep);
#endif
}

static inline void task10(tile_t *tile_out, tile_t *tile_in1, tile_t *tile_in2, tile_t *tile_in3, tile_t *tile_in4, tile_t *tile_in5, tile_t *tile_in6, tile_t *tile_in7, tile_t *tile_in8, tile_t *tile_in9, payload_t payload)
{
#if defined (USE_CORE_VERIFICATION)
  TaskGraph graph = payload.graph;
  char *output_ptr = (char*)tile_out->output_buff;
  size_t output_bytes= graph.output_bytes_per_task;
  std::vector<const char *> input_ptrs;
  std::vector<size_t> input_bytes;
  input_ptrs.push_back((char*)tile_in1->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in2->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in3->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in4->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in5->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in6->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in7->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in8->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);
  input_ptrs.push_back((char*)tile_in9->output_buff);
  input_bytes.push_back(graph.output_bytes_per_task);

  graph.execute_point(payload.y, payload.x, output_ptr, output_bytes,
                      input_ptrs.data(), input_bytes.data(), input_ptrs.size(), extra_local_memory[payload.x], graph.scratch_bytes_per_task);
#else
  tile_out->dep = tile_in1->dep + tile_in2->dep + tile_in3->dep + tile_in4->dep + tile_in5->dep + tile_in6->dep + tile_in7->dep + tile_in8->dep + tile_in9->dep + 1;
  printf("Task10 tid %d, x %d, y %d, out %f, in1 %f, in2 %f, in3 %f, in4 %f, in5 %f, in6 %f, in7 %f, in8 %f, in9 %f\n",
    payload.x, payload.x, payload.y, tile_out->dep,tile_in1->dep, tile_in2->dep, tile_in3->dep, tile_in4->dep, tile_in5->dep, tile_in6->dep, tile_in7->dep, tile_in8->dep, tile_in9->dep);
#endif
}

void CudaSTFApp::execute_main_loop()
{
  display();

  /* start timer */
  Timer::time_start();
  { 
  nvtx3::scoped_range r{"Main Loop"};
  {
    nvtx3::scoped_range r{"Task Submition Loop"};
    for (int i = 0; i < graphs.size(); i++) {
      const TaskGraph &g = graphs[i];

      for (int y = 0; y < g.timesteps; y++) {
        execute_timestep(i, y);
      }
    }
  }
  ctx.finalize();
  }
  double elapsed = Timer::time_end();
  report_timing(elapsed);
}

void CudaSTFApp::execute_timestep(size_t idx, long t) {
  // printf("Debug: Executing timestep %ld for graph %zu\n", t, idx);
  const TaskGraph &g = graphs[idx];
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  long dset = g.dependence_set_at_timestep(t);
  int nb_fields = g.nb_fields;
  
  task_args_t args[MAX_NUM_ARGS];
  payload_t payload;
  int num_args = 0;
  int ct = 0;  
  
  for (int x = offset; x <= offset+width-1; x++) {
    std::vector<std::pair<long, long> > deps = g.dependencies(dset, x);   
    num_args = 0;
    ct = 0;    
    
    if (deps.size() == 0) {
      num_args = 1;
      debug_printf(1, "%d[%d] ", x, num_args);
      args[ct].x = x;
      args[ct].y = t % nb_fields;
      ct ++;
    } else {
      if (t == 0) {
        num_args = 1;
        debug_printf(1, "%d[%d] ", x, num_args);
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct ++;
      } else {
        num_args = 1;
        args[ct].x = x;
        args[ct].y = t % nb_fields;
        ct ++;
        long last_offset = g.offset_at_timestep(t-1);
        long last_width = g.width_at_timestep(t-1);
        for (std::pair<long, long> dep : deps) {
          num_args += dep.second - dep.first + 1;
          debug_printf(1, "%d[%d, %d, %d] ", x, num_args, dep.first, dep.second); 
          for (int i = dep.first; i <= dep.second; i++) {
            if (i >= last_offset && i < last_offset + last_width) {
              args[ct].x = i;
              args[ct].y = (t-1) % nb_fields;
              ct ++;
            } else {
              num_args --;
            }
          }
        }
      }
    }
    
    assert(num_args == ct);
    
    payload.y = t;
    payload.x = x;
    payload.graph = g;
    {
    nvtx3::scoped_range r{"task submit"};
    insert_task(args, num_args, payload, idx);
    }
  }
}

void CudaSTFApp::insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id) {
  // // print the arguments
  // for (int i = 0; i < num_args; i++) {
  //   printf("arg %d: %d, %d\n", i, args[i].x, args[i].y);
  // }
  matrix_t* matrix = mat_array;
  matrix_t* mat = &matrix[graph_id];
  tile_t *tiles = matrix[graph_id].data;
  int nb_fields = matrix[graph_id].M;
  int max_width = matrix[graph_id].N;
  int x0 = args[0].x;
  int y0 = args[0].y;
  switch(num_args) {
  case 1:
  {
    logical_data<slice<char, 1>>& tile_buffer = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile) {
    ctx.host_launch(tile_buffer.write()).set_symbol("task")->*[=](auto dtile) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile.data_handle());
      task1(&t, payload);
    };
    break;
  }

  case 2:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile2.data_handle());
      task2(&t2, &t1, payload);
    };
    break;
  }
  case 3:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile3.data_handle());
      task3(&t3, &t1, &t2, payload);
    };
    break;
  }
  case 4:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile4.data_handle());
      task4(&t4, &t1, &t2, &t3, payload);
    };
    break;
  }
  case 5:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile5.data_handle());
      task5(&t5, &t1, &t2, &t3, &t4, payload);
    };
    break;
  }
  case 6:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    int x5 = args[5].x;
    int y5 = args[5].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y5 % nb_fields) * max_width + x5];
    logical_data<slice<char, 1>>& tile_buffer6 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y5 % nb_fields) * max_width + x5].dep, dtile5.data_handle());
      tile_t t6(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile6.data_handle());
      task6(&t6, &t1, &t2, &t3, &t4, &t5, payload);
    };
    break;
  }
  case 7:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    int x5 = args[5].x;
    int y5 = args[5].y;
    int x6 = args[6].x;
    int y6 = args[6].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y5 % nb_fields) * max_width + x5];
    logical_data<slice<char, 1>>& tile_buffer6 = mat->handles[(y6 % nb_fields) * max_width + x6];
    logical_data<slice<char, 1>>& tile_buffer7 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y5 % nb_fields) * max_width + x5].dep, dtile5.data_handle());
      tile_t t6(tiles[(y6 % nb_fields) * max_width + x6].dep, dtile6.data_handle());
      tile_t t7(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile7.data_handle());
      task7(&t7, &t1, &t2, &t3, &t4, &t5, &t6, payload);
    };
    break;
  }
  case 8:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    int x5 = args[5].x;
    int y5 = args[5].y;
    int x6 = args[6].x;
    int y6 = args[6].y;
    int x7 = args[7].x;
    int y7 = args[7].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y5 % nb_fields) * max_width + x5];
    logical_data<slice<char, 1>>& tile_buffer6 = mat->handles[(y6 % nb_fields) * max_width + x6];
    logical_data<slice<char, 1>>& tile_buffer7 = mat->handles[(y7 % nb_fields) * max_width + x7];
    logical_data<slice<char, 1>>& tile_buffer8 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y5 % nb_fields) * max_width + x5].dep, dtile5.data_handle());
      tile_t t6(tiles[(y6 % nb_fields) * max_width + x6].dep, dtile6.data_handle());
      tile_t t7(tiles[(y7 % nb_fields) * max_width + x7].dep, dtile7.data_handle());
      tile_t t8(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile8.data_handle());
      task8(&t8, &t1, &t2, &t3, &t4, &t5, &t6, &t7, payload);
    };
    break;
  }
  case 9:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    int x5 = args[5].x;
    int y5 = args[5].y;
    int x6 = args[6].x;
    int y6 = args[6].y;
    int x7 = args[7].x;
    int y7 = args[7].y;
    int x8 = args[8].x;
    int y8 = args[8].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y5 % nb_fields) * max_width + x5];
    logical_data<slice<char, 1>>& tile_buffer6 = mat->handles[(y6 % nb_fields) * max_width + x6];
    logical_data<slice<char, 1>>& tile_buffer7 = mat->handles[(y7 % nb_fields) * max_width + x7];
    logical_data<slice<char, 1>>& tile_buffer8 = mat->handles[(y8 % nb_fields) * max_width + x8];
    logical_data<slice<char, 1>>& tile_buffer9 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.read(), tile_buffer9.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8, auto dtile9) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.read(), tile_buffer9.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8, auto dtile9) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y5 % nb_fields) * max_width + x5].dep, dtile5.data_handle());
      tile_t t6(tiles[(y6 % nb_fields) * max_width + x6].dep, dtile6.data_handle());
      tile_t t7(tiles[(y7 % nb_fields) * max_width + x7].dep, dtile7.data_handle());
      tile_t t8(tiles[(y8 % nb_fields) * max_width + x8].dep, dtile8.data_handle());
      tile_t t9(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile9.data_handle());
      task9(&t9, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, payload);
    };
    break;
  }
  case 10:
  {
    int x1 = args[1].x;
    int y1 = args[1].y;
    int x2 = args[2].x;
    int y2 = args[2].y;
    int x3 = args[3].x;
    int y3 = args[3].y;
    int x4 = args[4].x;
    int y4 = args[4].y;
    int x5 = args[5].x;
    int y5 = args[5].y;
    int x6 = args[6].x;
    int y6 = args[6].y;
    int x7 = args[7].x;
    int y7 = args[7].y;
    int x8 = args[8].x;
    int y8 = args[8].y;
    int x9 = args[9].x;
    int y9 = args[9].y;
    logical_data<slice<char, 1>>& tile_buffer1 = mat->handles[(y1 % nb_fields) * max_width + x1];
    logical_data<slice<char, 1>>& tile_buffer2 = mat->handles[(y2 % nb_fields) * max_width + x2];
    logical_data<slice<char, 1>>& tile_buffer3 = mat->handles[(y3 % nb_fields) * max_width + x3];
    logical_data<slice<char, 1>>& tile_buffer4 = mat->handles[(y4 % nb_fields) * max_width + x4];
    logical_data<slice<char, 1>>& tile_buffer5 = mat->handles[(y5 % nb_fields) * max_width + x5];
    logical_data<slice<char, 1>>& tile_buffer6 = mat->handles[(y6 % nb_fields) * max_width + x6];
    logical_data<slice<char, 1>>& tile_buffer7 = mat->handles[(y7 % nb_fields) * max_width + x7];
    logical_data<slice<char, 1>>& tile_buffer8 = mat->handles[(y8 % nb_fields) * max_width + x8];
    logical_data<slice<char, 1>>& tile_buffer9 = mat->handles[(y9 % nb_fields) * max_width + x9];
    logical_data<slice<char, 1>>& tile_buffer10 = mat->handles[(y0 % nb_fields) * max_width + x0];
    // ctx.task(exec_place::host(), tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.read(), tile_buffer9.read(), tile_buffer10.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8, auto dtile9, auto dtile10) {
    ctx.host_launch(tile_buffer1.read(), tile_buffer2.read(), tile_buffer3.read(), tile_buffer4.read(), tile_buffer5.read(), tile_buffer6.read(), tile_buffer7.read(), tile_buffer8.read(), tile_buffer9.read(), tile_buffer10.write()).set_symbol("task")->*[=](auto dtile1, auto dtile2, auto dtile3, auto dtile4, auto dtile5, auto dtile6, auto dtile7, auto dtile8, auto dtile9, auto dtile10) {
    //   nvtx3::scoped_range r{"inner-task"};
      tile_t t1(tiles[(y1 % nb_fields) * max_width + x1].dep, dtile1.data_handle());
      tile_t t2(tiles[(y2 % nb_fields) * max_width + x2].dep, dtile2.data_handle());
      tile_t t3(tiles[(y3 % nb_fields) * max_width + x3].dep, dtile3.data_handle());
      tile_t t4(tiles[(y4 % nb_fields) * max_width + x4].dep, dtile4.data_handle());
      tile_t t5(tiles[(y5 % nb_fields) * max_width + x5].dep, dtile5.data_handle());
      tile_t t6(tiles[(y6 % nb_fields) * max_width + x6].dep, dtile6.data_handle());
      tile_t t7(tiles[(y7 % nb_fields) * max_width + x7].dep, dtile7.data_handle());
      tile_t t8(tiles[(y8 % nb_fields) * max_width + x8].dep, dtile8.data_handle());
      tile_t t9(tiles[(y9 % nb_fields) * max_width + x9].dep, dtile9.data_handle());
      tile_t t10(tiles[(y0 % nb_fields) * max_width + x0].dep, dtile10.data_handle());
      task10(&t10, &t1, &t2, &t3, &t4, &t5, &t6, &t7, &t8, &t9, payload);
    };
    break;
  }
  default:
  assert(false && "unexpected num_args");
  };
}

// Main function
int main(int argc, char **argv) {
  CudaSTFApp app(argc, argv);
  app.execute_main_loop();
  return 0;
}

void CudaSTFApp::debug_printf(int verbose_level, const char *format, ...)
{
  if (verbose_level > VERBOSE_LEVEL) {
    return;
  }
  va_list args;
  va_start(args, format);
  vprintf(format, args);
  va_end(args);
}