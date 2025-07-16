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
  printf("DEBUG: Initializing CUDA STF App with %zu graphs\n", graphs.size());

  size_t max_scratch_bytes_per_task = 0;
  matrix_t *matrix = mat_array;
  for (unsigned i = 0; i < graphs.size(); i++) {
    TaskGraph &graph = graphs[i];
    
    matrix[i].M = graph.nb_fields;
    matrix[i].N = graph.max_width;
    matrix[i].data = (tile_t*)malloc(sizeof(tile_t) * matrix[i].M * matrix[i].N);
  
    for (int j = 0; j < matrix[i].M * matrix[i].N; j++) {
      matrix[i].data[j].output_buff = (char *)malloc(sizeof(char) * graph.output_bytes_per_task);
    }
    
    if (graph.scratch_bytes_per_task > max_scratch_bytes_per_task) {
      max_scratch_bytes_per_task = graph.scratch_bytes_per_task;
    }
    
    printf("graph id %d, M = %d, N = %d, data %p, nb_fields %d\n", i, matrix[i].M, matrix[i].N, matrix[i].data, graph.nb_fields);
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
  // 等待所有任务完成
  printf("DEBUG: Finalizing CUDA STF context\n");
  ctx.finalize();
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

  printf("DEBUG: CUDA STF cleanup complete\n");
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
  printf("Task1, x %d, y %d, out %f\n", payload.x, payload.y, tile_out->dep);
#endif  
}

void CudaSTFApp::execute_main_loop()
{
  display();

  /* start timer */
  Timer::time_start();
  
  for (int i = 0; i < graphs.size(); i++) {
    const TaskGraph &g = graphs[i];

    for (int y = 0; y < g.timesteps; y++) {
      execute_timestep(i, y);
    }
  }

  ctx.finalize();
  double elapsed = Timer::time_end();
  report_timing(elapsed);
}

void CudaSTFApp::execute_timestep(size_t idx, long t) {
  printf("Debug: Executing timestep %ld for graph %zu\n", t, idx);
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
    insert_task(args, num_args, payload, idx);
  }
}

void CudaSTFApp::insert_task(task_args_t *args, int num_args, payload_t payload, size_t graph_id) {
  // print the arguments
  for (int i = 0; i < num_args; i++) {
    printf("arg %d: %d, %d\n", i, args[i].x, args[i].y);
  }
  matrix_t* matrix = mat_array;
  tile_t *mat = matrix[graph_id].data;
  int x0 = args[0].x;
  int y0 = args[0].y;
//  printf("x %d, y %d, mat %p\n", x0, y0, mat);
  switch(num_args) {
  case 1:
  {
    // #pragma omp task depend(inout: mat[y0 * matrix[graph_id].N + x0]) untied mergeable
    logical_data<slice<char>> tile_buffer = ctx.logical_data(mat[y0 * matrix[graph_id].N + x0].output_buff, payload.graph.output_bytes_per_task);
    tile_buffer.set_symbol("out_buffer");
    ctx.task(exec_place::host(), tile_buffer.write()).set_symbol("task(" + std::to_string(x0) + "," + std::to_string(y0) + ")")->*[=](cudaStream_t stream, auto dtile) {
      task1(&mat[y0 * matrix[graph_id].N + x0], payload);
    };
    // task1(&mat[y0 * matrix[graph_id].N + x0], payload);
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