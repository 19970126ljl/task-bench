#ifndef TASK_BENCH_CUDASTF_WORKLOAD_CUH
#define TASK_BENCH_CUDASTF_WORKLOAD_CUH

#include <cuda/experimental/stf.cuh>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "arguments.h"
#include "expanded_dag.h"
#include "workload.h"

using cuda::experimental::stf::slice;

struct TaskDataView {
  unsigned char *output = nullptr;
  const unsigned char *inputs[kMaxTaskInputs] = {};
  unsigned char *scratch = nullptr;
  std::size_t input_count = 0;
  std::size_t output_bytes = 0;
  std::size_t scratch_bytes = 0;
};

struct WorkloadKernelLimits {
  int max_threads_per_block = 0;
  std::size_t static_shared_memory_bytes = 0;
};

inline void check_workload_cuda(cudaError_t status, const char *operation)
{
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(operation) + ": " +
                             cudaGetErrorString(status));
  }
}

__device__ __forceinline__ std::uint64_t global_thread_id()
{
  return static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
         threadIdx.x;
}

__device__ __forceinline__ std::uint64_t global_thread_count()
{
  return static_cast<std::uint64_t>(gridDim.x) * blockDim.x;
}

__device__ __forceinline__ uint4 xor_uint4(uint4 lhs, uint4 rhs)
{
  lhs.x ^= rhs.x;
  lhs.y ^= rhs.y;
  lhs.z ^= rhs.z;
  lhs.w ^= rhs.w;
  return lhs;
}

__device__ __forceinline__ uint4 vector_seed(std::uint64_t word)
{
  const unsigned int base =
      static_cast<unsigned int>(word) * 0x9e3779b9U + 1U;
  return make_uint4(base, base + 0x3c6ef372U, base + 0x78dde6e4U,
                    base + 0xb54cda56U);
}

__device__ __forceinline__ void transform_task_data(const TaskDataView &data)
{
  constexpr std::size_t vector_bytes = sizeof(uint4);
  const std::size_t vector_count = data.output_bytes / vector_bytes;
  auto *output_vectors = reinterpret_cast<uint4 *>(data.output);

  for (std::uint64_t word = global_thread_id(); word < vector_count;
       word += global_thread_count()) {
    uint4 value = vector_seed(word);
    for (std::size_t input = 0; input < data.input_count; ++input) {
      const auto *input_vectors =
          reinterpret_cast<const uint4 *>(data.inputs[input]);
      value = xor_uint4(value, input_vectors[word]);
    }
    output_vectors[word] = value;
  }

  const std::size_t vectorized_bytes = vector_count * vector_bytes;
  for (std::uint64_t byte = vectorized_bytes + global_thread_id();
       byte < data.output_bytes;
       byte += global_thread_count()) {
    unsigned char value =
        static_cast<unsigned char>(static_cast<unsigned int>(byte) + 1U);
    for (std::size_t input = 0; input < data.input_count; ++input) {
      value ^= data.inputs[input][byte];
    }
    data.output[byte] = value;
  }
}

// Keep synthetic arithmetic observable without adding data traffic or checks.
__device__ __forceinline__ void retain_workload_result(std::uint64_t value)
{
  if ((value & 1U) != 0) {
    asm volatile("membar.cta;");
  }
}

__device__ __forceinline__ void retain_workload_result(float value)
{
  if ((__float_as_uint(value) & 1U) != 0) {
    asm volatile("membar.cta;");
  }
}

__device__ __forceinline__ void retain_workload_result(double value)
{
  if ((static_cast<std::uint64_t>(__double_as_longlong(value)) & 1U) != 0) {
    asm volatile("membar.cta;");
  }
}

__device__ __forceinline__ void run_busy_wait(std::uint64_t iterations)
{
  const std::uint64_t thread = global_thread_id();
  const std::uint64_t thread_count = global_thread_count();
  std::uint64_t accumulator = 113 + thread % 1024;
  for (std::uint64_t iteration = thread; iteration < iterations;
       iteration += thread_count) {
    accumulator = accumulator * 139 % UINT64_C(2147483647);
  }
  retain_workload_result(accumulator);
}

template <typename T>
__device__ __forceinline__ T compute_step(T accumulator);

template <>
__device__ __forceinline__ float compute_step(float accumulator)
{
  return fmaf(accumulator, 0.999999f, 0.000001f);
}

template <>
__device__ __forceinline__ double compute_step(double accumulator)
{
  return fma(accumulator, 0.999999999999, 0.000000000001);
}

template <typename T>
__device__ __forceinline__ void run_compute_bound(
    std::uint64_t iterations)
{
  constexpr int accumulator_count = 8;
  constexpr int rounds_per_iteration = 8;
  const std::uint64_t thread = global_thread_id();
  const std::uint64_t thread_count = global_thread_count();
  T accumulators[accumulator_count];
#pragma unroll
  for (int accumulator = 0; accumulator < accumulator_count;
       ++accumulator) {
    accumulators[accumulator] =
        static_cast<T>(1) +
        static_cast<T>((thread + accumulator) % 32) *
            static_cast<T>(0.001);
  }
  for (std::uint64_t iteration = thread; iteration < iterations;
       iteration += thread_count) {
#pragma unroll
    for (int round = 0; round < rounds_per_iteration; ++round) {
#pragma unroll
      for (int accumulator = 0; accumulator < accumulator_count;
           ++accumulator) {
        accumulators[accumulator] =
            compute_step(accumulators[accumulator]);
      }
    }
  }
  T result = accumulators[0];
#pragma unroll
  for (int accumulator = 1; accumulator < accumulator_count;
       ++accumulator) {
    result += accumulators[accumulator];
  }
  retain_workload_result(result);
}

__device__ __forceinline__ void copy_scratch_sample(
    const unsigned char *source, unsigned char *destination,
    std::size_t bytes)
{
  const std::uintptr_t addresses =
      reinterpret_cast<std::uintptr_t>(source) |
      reinterpret_cast<std::uintptr_t>(destination);
  const bool word_aligned = (addresses & (alignof(std::uint32_t) - 1)) == 0;
  const std::size_t word_count =
      word_aligned ? bytes / sizeof(std::uint32_t) : 0;

  const auto *source_words =
      reinterpret_cast<const volatile std::uint32_t *>(source);
  auto *destination_words =
      reinterpret_cast<volatile std::uint32_t *>(destination);
  for (std::uint64_t word = global_thread_id(); word < word_count;
       word += global_thread_count()) {
    destination_words[word] = source_words[word];
  }

  const std::size_t copied_bytes = word_count * sizeof(std::uint32_t);
  const auto *source_bytes =
      static_cast<const volatile unsigned char *>(source);
  auto *destination_bytes =
      static_cast<volatile unsigned char *>(destination);
  for (std::uint64_t byte = copied_bytes + global_thread_id(); byte < bytes;
       byte += global_thread_count()) {
    destination_bytes[byte] = source_bytes[byte];
  }
}

__device__ __forceinline__ void run_memory_bound(
    const TaskDataView &data, std::uint64_t iterations,
    std::uint64_t samples, std::uint64_t timestep)
{
  const std::size_t sample_bytes =
      data.scratch_bytes / static_cast<std::size_t>(samples);
  const std::size_t copy_bytes = sample_bytes / 2;
  std::uint64_t sample =
      ((timestep % samples) * (iterations % samples)) % samples;

  for (std::uint64_t iteration = 0; iteration < iterations; ++iteration) {
    const std::size_t sample_offset =
        static_cast<std::size_t>(sample) * sample_bytes;
    const unsigned char *source = data.scratch + sample_offset;
    unsigned char *destination =
        data.scratch + sample_offset + copy_bytes;
    copy_scratch_sample(source, destination, copy_bytes);
    if (++sample == samples) {
      sample = 0;
    }
  }
}

template <GpuWorkload Workload, typename ComputeT>
__global__ void task_workload_kernel(
    TaskDataView data, std::uint64_t iterations, std::uint64_t samples,
    std::uint64_t timestep)
{
  transform_task_data(data);

  if constexpr (Workload == GpuWorkload::busy_wait) {
    run_busy_wait(iterations);
  } else if constexpr (Workload == GpuWorkload::memory_bound) {
    run_memory_bound(data, iterations, samples, timestep);
  } else if constexpr (Workload == GpuWorkload::compute_bound) {
    run_compute_bound<ComputeT>(iterations);
  }
}

template <GpuWorkload Workload, typename ComputeT, typename StfTask>
void attach_typed_workload(
    StfTask &stf_task, std::size_t input_count, std::size_t output_bytes,
    std::size_t scratch_bytes, std::uint64_t iterations,
    std::uint64_t samples, std::uint64_t timestep,
    const GpuKernelConfig &gpu_kernel_config)
{
  const GpuKernelLaunchConfig launch_config = gpu_kernel_config.launch;
  stf_task->*[&stf_task, input_count, output_bytes, scratch_bytes,
              iterations, samples, timestep,
              launch_config](cudaStream_t stream) {
    TaskDataView data;
    data.output =
        stf_task.template get<slice<unsigned char>>(0).data_handle();
    data.input_count = input_count;
    data.output_bytes = output_bytes;
    data.scratch_bytes = scratch_bytes;
    for (std::size_t i = 0; i < input_count; ++i) {
      data.inputs[i] =
          stf_task.template get<slice<unsigned char>>(i + 1).data_handle();
    }
    if (scratch_bytes != 0) {
      data.scratch =
          stf_task.template get<slice<unsigned char>>(input_count + 1)
              .data_handle();
    }

    task_workload_kernel<Workload, ComputeT>
        <<<launch_config.blocks_per_task,
           launch_config.threads_per_block,
           launch_config.dynamic_shared_memory_bytes, stream>>>(
            data, iterations, samples, timestep);
    check_workload_cuda(cudaPeekAtLastError(), "task workload launch");
  };
}

template <typename StfTask>
void attach_gpu_workload(
    StfTask &stf_task, const DagTask &dag_task,
    const TaskGraph &task_graph,
    std::uint64_t iterations,
    const GpuKernelConfig &gpu_kernel_config)
{
  const std::uint64_t samples =
      static_cast<std::uint64_t>(task_graph.kernel.samples);
  const std::size_t input_count = dag_task.predecessors.size();
  const std::size_t output_bytes = task_graph.output_bytes_per_task;
  const std::size_t scratch_bytes = task_graph.scratch_bytes_per_task;
  const std::uint64_t timestep =
      static_cast<std::uint64_t>(dag_task.coordinates.timestep);

  switch (gpu_workload(task_graph.kernel.type)) {
  case GpuWorkload::empty:
    return attach_typed_workload<GpuWorkload::empty, float>(
        stf_task, input_count, output_bytes, 0, iterations, 0, timestep,
        gpu_kernel_config);
  case GpuWorkload::busy_wait:
    return attach_typed_workload<GpuWorkload::busy_wait, float>(
        stf_task, input_count, output_bytes, 0, iterations, 0, timestep,
        gpu_kernel_config);
  case GpuWorkload::memory_bound:
    return attach_typed_workload<GpuWorkload::memory_bound, float>(
        stf_task, input_count, output_bytes, scratch_bytes, iterations,
        samples, timestep,
        gpu_kernel_config);
  case GpuWorkload::compute_bound:
    if (gpu_kernel_config.compute_data_type == ComputeDataType::fp32) {
      return attach_typed_workload<GpuWorkload::compute_bound, float>(
          stf_task, input_count, output_bytes, 0, iterations, 0, timestep,
          gpu_kernel_config);
    }
    return attach_typed_workload<GpuWorkload::compute_bound, double>(
        stf_task, input_count, output_bytes, 0, iterations, 0, timestep,
        gpu_kernel_config);
  }
  throw std::logic_error("unknown GPU workload");
}

template <GpuWorkload Workload, typename ComputeT>
WorkloadKernelLimits typed_workload_kernel_limits()
{
  cudaFuncAttributes attributes{};
  check_workload_cuda(
      cudaFuncGetAttributes(
          &attributes, task_workload_kernel<Workload, ComputeT>),
      "cudaFuncGetAttributes task workload");
  return {attributes.maxThreadsPerBlock,
          static_cast<std::size_t>(
              attributes.sharedSizeBytes)};
}

template <GpuWorkload Workload, typename ComputeT>
GpuKernelResources typed_gpu_kernel_resources(
    const GpuKernelLaunchConfig &launch_config)
{
  cudaFuncAttributes attributes{};
  check_workload_cuda(
      cudaFuncGetAttributes(
          &attributes, task_workload_kernel<Workload, ComputeT>),
      "cudaFuncGetAttributes task workload");
  int max_active_blocks_per_sm = 0;
  check_workload_cuda(
      cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &max_active_blocks_per_sm,
          task_workload_kernel<Workload, ComputeT>,
          launch_config.threads_per_block,
          launch_config.dynamic_shared_memory_bytes),
      "cudaOccupancyMaxActiveBlocksPerMultiprocessor task workload");
  return {attributes.numRegs,
          static_cast<std::size_t>(attributes.sharedSizeBytes),
          max_active_blocks_per_sm};
}

template <GpuWorkload Workload, typename ComputeT>
void set_typed_workload_dynamic_shared_memory_limit(std::size_t bytes)
{
  if (bytes > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "dynamic shared-memory kernel limit exceeds int");
  }
  check_workload_cuda(
      cudaFuncSetAttribute(
          task_workload_kernel<Workload, ComputeT>,
          cudaFuncAttributeMaxDynamicSharedMemorySize,
          static_cast<int>(bytes)),
      "cudaFuncSetAttribute dynamic shared memory");
}

inline WorkloadKernelLimits workload_kernel_limits(
    KernelType type, ComputeDataType compute_data_type)
{
  switch (gpu_workload(type)) {
  case GpuWorkload::empty:
    return typed_workload_kernel_limits<GpuWorkload::empty, float>();
  case GpuWorkload::busy_wait:
    return typed_workload_kernel_limits<GpuWorkload::busy_wait, float>();
  case GpuWorkload::memory_bound:
    return typed_workload_kernel_limits<GpuWorkload::memory_bound, float>();
  case GpuWorkload::compute_bound:
    if (compute_data_type == ComputeDataType::fp32) {
      return typed_workload_kernel_limits<
          GpuWorkload::compute_bound, float>();
    }
    return typed_workload_kernel_limits<
        GpuWorkload::compute_bound, double>();
  }
  throw std::logic_error("unknown GPU workload");
}

inline GpuKernelResources gpu_kernel_resources(
    KernelType type, ComputeDataType compute_data_type,
    const GpuKernelLaunchConfig &launch_config)
{
  switch (gpu_workload(type)) {
  case GpuWorkload::empty:
    return typed_gpu_kernel_resources<GpuWorkload::empty, float>(
        launch_config);
  case GpuWorkload::busy_wait:
    return typed_gpu_kernel_resources<GpuWorkload::busy_wait, float>(
        launch_config);
  case GpuWorkload::memory_bound:
    return typed_gpu_kernel_resources<GpuWorkload::memory_bound, float>(
        launch_config);
  case GpuWorkload::compute_bound:
    if (compute_data_type == ComputeDataType::fp32) {
      return typed_gpu_kernel_resources<
          GpuWorkload::compute_bound, float>(launch_config);
    }
    return typed_gpu_kernel_resources<
        GpuWorkload::compute_bound, double>(launch_config);
  }
  throw std::logic_error("unknown GPU workload");
}

inline void set_workload_dynamic_shared_memory_limit(
    KernelType type, ComputeDataType compute_data_type,
    std::size_t bytes)
{
  switch (gpu_workload(type)) {
  case GpuWorkload::empty:
    return set_typed_workload_dynamic_shared_memory_limit<
        GpuWorkload::empty, float>(bytes);
  case GpuWorkload::busy_wait:
    return set_typed_workload_dynamic_shared_memory_limit<
        GpuWorkload::busy_wait, float>(bytes);
  case GpuWorkload::memory_bound:
    return set_typed_workload_dynamic_shared_memory_limit<
        GpuWorkload::memory_bound, float>(bytes);
  case GpuWorkload::compute_bound:
    if (compute_data_type == ComputeDataType::fp32) {
      return set_typed_workload_dynamic_shared_memory_limit<
          GpuWorkload::compute_bound, float>(bytes);
    }
    return set_typed_workload_dynamic_shared_memory_limit<
        GpuWorkload::compute_bound, double>(bytes);
  }
  throw std::logic_error("unknown GPU workload");
}

#endif
