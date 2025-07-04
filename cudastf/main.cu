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
#include <algorithm>
#include <vector>
#include <memory>
#include <cassert>

#include "core.h"
#include "timer.h"

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

// CUDA STF App implementation
struct CudaSTFApp : public App {
  CudaSTFApp(int argc, char **argv);
  ~CudaSTFApp();
  
  void execute_main_loop();
  void execute_timestep(size_t graph_idx, long timestep);

  // Helper methods
  std::shared_ptr<std::vector<char>>& get_or_create_host_data(const std::pair<long, long>& key, size_t size);
  logical_data<slice<char>>& get_or_create_logical_data(const std::pair<long, long>& key, std::vector<char>& host_data);
  void execute_task_no_deps(const TaskGraph& graph, long timestep, long point,
                           logical_data<slice<char>>& output, logical_data<slice<char>>* scratch);
  void execute_task_with_deps(const TaskGraph& graph, long timestep, long point,
                             const std::vector<std::pair<long, long>>& deps,
                             logical_data<slice<char>>& output, logical_data<slice<char>>* scratch);

private:
  // STF context
  context ctx;

  // Data storage for logical data across timesteps
  std::map<std::pair<long, long>, std::shared_ptr<std::vector<char>>> host_data_storage;
  std::map<std::pair<long, long>, logical_data<slice<char>>> logical_data_storage;
};

// Constructor
CudaSTFApp::CudaSTFApp(int argc, char **argv)
  : App(argc, argv)
{
  // Simple constructor - no pre-initialization needed
}

// Destructor
CudaSTFApp::~CudaSTFApp() {
  ctx.finalize();
}

// Main execution loop
void CudaSTFApp::execute_main_loop() {
  display();
  
  Timer::time_start();
  
  // Execute all graphs
  for (size_t i = 0; i < graphs.size(); i++) {
    const TaskGraph &graph = graphs[i];
    for (long t = 0; t < graph.timesteps; t++) {
      execute_timestep(i, t);
    }
  }
  
  // Wait for all tasks to complete
  ctx.finalize();
  
  double elapsed = Timer::time_end();
  report_timing(elapsed);
}

// Execute a single timestep
void CudaSTFApp::execute_timestep(size_t graph_idx, long timestep) {
  const TaskGraph &graph = graphs[graph_idx];
  long offset = graph.offset_at_timestep(timestep);
  long width = graph.width_at_timestep(timestep);
  long dset = graph.dependence_set_at_timestep(timestep);

  for (long point = offset; point < offset + width; point++) {
    // Get or create output data for this task
    auto output_key = std::make_pair(timestep, point);
    auto& output_host_data = get_or_create_host_data(output_key, graph.output_bytes_per_task);
    auto& output_logical_data = get_or_create_logical_data(output_key, *output_host_data);

    // Get dependencies for this point
    std::vector<std::pair<long, long>> deps = graph.dependencies(dset, point);

    // Create scratch data if needed
    logical_data<slice<char>>* scratch_logical_data = nullptr;
    if (graph.scratch_bytes_per_task > 0) {
      auto scratch_key = std::make_pair(timestep, point + 1000000); // Unique key for scratch
      auto& scratch_host_data = get_or_create_host_data(scratch_key, graph.scratch_bytes_per_task);
      scratch_logical_data = &get_or_create_logical_data(scratch_key, *scratch_host_data);

      // Initialize scratch data
      TaskGraph::prepare_scratch(scratch_host_data->data(), scratch_host_data->size());
    }

    if (deps.empty()) {
      // No dependencies - initial task
      execute_task_no_deps(graph, timestep, point, output_logical_data, scratch_logical_data);
    } else {
      // Has dependencies
      execute_task_with_deps(graph, timestep, point, deps, output_logical_data, scratch_logical_data);
    }
  }
}

// Helper method implementations
std::shared_ptr<std::vector<char>>& CudaSTFApp::get_or_create_host_data(const std::pair<long, long>& key, size_t size) {
  auto it = host_data_storage.find(key);
  if (it == host_data_storage.end()) {
    auto data = std::make_shared<std::vector<char>>(size);
    // Initialize with timestep and point info
    if (size >= 2 * sizeof(long)) {
      *reinterpret_cast<long*>(data->data()) = key.first;  // timestep
      *reinterpret_cast<long*>(data->data() + sizeof(long)) = key.second; // point
    }
    host_data_storage[key] = data;
    return host_data_storage[key];
  }
  return it->second;
}

logical_data<slice<char>>& CudaSTFApp::get_or_create_logical_data(const std::pair<long, long>& key, std::vector<char>& host_data) {
  auto it = logical_data_storage.find(key);
  if (it == logical_data_storage.end()) {
    logical_data_storage[key] = ctx.logical_data(host_data.data(), host_data.size());
    return logical_data_storage[key];
  }
  return it->second;
}

void CudaSTFApp::execute_task_no_deps(const TaskGraph& graph, long timestep, long point,
                                     logical_data<slice<char>>& output, logical_data<slice<char>>* scratch) {
  switch (graph.kernel.type) {
    case KernelType::EMPTY:
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        empty_kernel<<<1, 1, 0, s>>>();
      };
      break;

    case KernelType::BUSY_WAIT:
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        busy_wait_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
      };
      break;

    case KernelType::COMPUTE_BOUND:
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
      };
      break;

    case KernelType::COMPUTE_BOUND2:
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        compute2_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
      };
      break;

    case KernelType::MEMORY_BOUND:
      if (scratch) {
        ctx.task(output.write(), scratch->rw())->*[=](cudaStream_t s, auto out, auto scr) {
          memory_kernel<<<16, 128, 0, s>>>(out, scr, graph.kernel.iterations, timestep);
        };
      } else {
        // Fallback to compute if no scratch
        ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;

    case KernelType::LOAD_IMBALANCE:
      // For load imbalance, use compute with variable iterations
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        // Simple imbalance simulation - vary iterations based on point
        long imbalanced_iters = graph.kernel.iterations * (1.0 + (point % 2) * graph.kernel.imbalance);
        compute_kernel<<<1, 1, 0, s>>>(out, imbalanced_iters);
      };
      break;

    default:
      // For other kernel types (COMPUTE_DGEMM, MEMORY_DAXPY, IO_BOUND), use compute as fallback
      ctx.task(output.write())->*[=](cudaStream_t s, auto out) {
        compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
      };
      break;
  }
}

void CudaSTFApp::execute_task_with_deps(const TaskGraph& graph, long timestep, long point,
                                       const std::vector<std::pair<long, long>>& deps,
                                       logical_data<slice<char>>& output, logical_data<slice<char>>* scratch) {
  // Collect input logical data
  std::vector<logical_data<slice<char>>*> input_data;
  for (const auto& dep : deps) {
    auto dep_key = std::make_pair(dep.first, dep.second);
    auto it = logical_data_storage.find(dep_key);
    if (it != logical_data_storage.end()) {
      input_data.push_back(&it->second);
    }
  }

  // Execute task based on kernel type and number of inputs
  // For simplicity, we'll handle up to 2 inputs explicitly, then use a general approach
  if (input_data.empty()) {
    // No inputs - treat as no-dependency task
    execute_task_no_deps(graph, timestep, point, output, scratch);
    return;
  }

  switch (graph.kernel.type) {
    case KernelType::EMPTY:
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          empty_kernel<<<1, 1, 0, s>>>();
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          empty_kernel<<<1, 1, 0, s>>>();
        };
      }
      break;

    case KernelType::BUSY_WAIT:
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          busy_wait_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          busy_wait_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;

    case KernelType::COMPUTE_BOUND:
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;

    case KernelType::COMPUTE_BOUND2:
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          compute2_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          compute2_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;

    case KernelType::MEMORY_BOUND:
      if (scratch && input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write(), scratch->rw())->*[=](cudaStream_t s, auto in, auto out, auto scr) {
          memory_kernel<<<16, 128, 0, s>>>(out, scr, graph.kernel.iterations, timestep);
        };
      } else if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;

    case KernelType::LOAD_IMBALANCE:
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          long imbalanced_iters = graph.kernel.iterations * (1.0 + (point % 2) * graph.kernel.imbalance);
          compute_kernel<<<1, 1, 0, s>>>(out, imbalanced_iters);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          long imbalanced_iters = graph.kernel.iterations * (1.0 + (point % 2) * graph.kernel.imbalance);
          compute_kernel<<<1, 1, 0, s>>>(out, imbalanced_iters);
        };
      }
      break;

    default:
      // Fallback for other kernel types
      if (input_data.size() == 1) {
        ctx.task(input_data[0]->read(), output.write())->*[=](cudaStream_t s, auto in, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      } else if (input_data.size() >= 2) {
        ctx.task(input_data[0]->read(), input_data[1]->read(), output.write())->*[=](cudaStream_t s, auto in1, auto in2, auto out) {
          compute_kernel<<<1, 1, 0, s>>>(out, graph.kernel.iterations);
        };
      }
      break;
  }
}

// Main function
int main(int argc, char **argv) {
  CudaSTFApp app(argc, argv);
  app.execute_main_loop();
  return 0;
}
