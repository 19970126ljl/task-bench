# Task Bench 项目分析报告

## 项目概述

Task Bench 是一个用于评估并行和分布式编程模型、运行时和语言效率与性能的可配置基准测试工具。它主要针对基于任务的模型进行评估，其中基本执行单元是任务，但可以在任何并行系统中实现。

### 核心特性
- **可配置的任务图**：可以看作是一个迭代空间，每个点都有任务
- **可配置的依赖关系**：任务之间具有可配置的依赖关系
- **多种内核类型**：支持计算密集型、内存密集型、通信密集型或运行时开销密集型（空任务）
- **多种实现**：支持20+种并行编程框架

### 支持的编程框架
- Charm++, Chapel, Dask, HPX, Legion, MPI, MPI+OpenMP
- OmpSs, OpenMP, PaRSEC, Pygion, Realm, Regent
- Spark, StarPU, Swift/T, TensorFlow, X10

### 依赖模式
- trivial, no_comm, stencil_1d, stencil_1d_periodic
- dom, tree, fft, all_to_all, nearest, spread, random_nearest

### 内核类型
- compute-bound（计算密集型）
- memory-bound（内存密集型）
- load-imbalanced compute-bound（负载不平衡计算密集型）
- empty（空任务）

## 核心架构

### 目录结构
```
task-bench/
├── core/                    # 核心库和接口
│   ├── core.h              # C++接口定义
│   ├── core_c.h            # C接口定义
│   ├── core.cc             # 核心实现
│   ├── timer.h/cc          # 计时器
│   └── Makefile            # 核心库构建
├── [framework]/            # 各种框架实现目录
│   ├── main.cc             # 主实现文件
│   └── Makefile            # 框架特定构建
├── experiments/            # 实验配置
├── scripts/                # 分析脚本
└── build_all.sh           # 统一构建脚本
```

### 核心接口

#### TaskGraph 类
```cpp
struct TaskGraph : public task_graph_t {
  // 时间步相关
  long offset_at_timestep(long timestep) const;
  long width_at_timestep(long timestep) const;
  
  // 依赖关系
  std::vector<std::pair<long, long>> dependencies(long dset, long point) const;
  std::vector<std::pair<long, long>> reverse_dependencies(long dset, long point) const;
  
  // 任务执行
  void execute_point(long timestep, long point,
                     char *output_ptr, size_t output_bytes,
                     const char **input_ptr, const size_t *input_bytes,
                     size_t n_inputs,
                     char *scratch_ptr, size_t scratch_bytes) const;
};
```

#### App 类
```cpp
struct App {
  std::vector<TaskGraph> graphs;
  long nodes;
  int verbose;
  bool enable_graph_validation;
  
  App(int argc, char **argv);  // 命令行解析
  void check() const;          // 验证
  void display() const;        // 显示配置
  void report_timing(double elapsed_seconds) const;  // 报告性能
};
```

## 核心流程

### 1. 初始化阶段
1. **命令行解析**：App构造函数解析参数，配置任务图
2. **任务图构建**：根据参数创建TaskGraph对象
3. **内存分配**：为任务数据和scratch空间分配内存
4. **框架初始化**：初始化特定并行框架

### 2. 执行阶段
1. **时间步迭代**：按时间步顺序执行
2. **依赖分析**：对每个任务点分析其依赖关系
3. **任务提交**：根据依赖关系提交任务到框架
4. **任务执行**：调用TaskGraph::execute_point执行实际计算

### 3. 完成阶段
1. **同步等待**：等待所有任务完成
2. **性能统计**：收集执行时间
3. **结果验证**：可选的正确性验证
4. **报告输出**：输出性能指标

### 典型执行流程（以OpenMP为例）
```cpp
// 主循环
for (unsigned i = 0; i < graphs.size(); i++) {
  const TaskGraph &g = graphs[i];
  for (int y = 0; y < g.timesteps; y++) {
    execute_timestep(i, y);  // 执行时间步
  }
}

// 时间步执行
void execute_timestep(size_t idx, long t) {
  // 获取当前时间步的任务范围
  long offset = g.offset_at_timestep(t);
  long width = g.width_at_timestep(t);
  
  // 为每个任务点分析依赖并提交任务
  for (int x = offset; x <= offset+width-1; x++) {
    std::vector<std::pair<long, long>> deps = g.dependencies(dset, x);
    // 根据依赖数量选择合适的任务函数
    insert_task(args, num_args, payload, idx);
  }
}
```

## 基本用法

### 构建
```bash
# 构建所有实现
./get_deps.sh
./build_all.sh

# 构建特定实现（如Legion）
DEFAULT_FEATURES=0 USE_LEGION=1 ./get_deps.sh
./build_all.sh
```

### 运行示例
```bash
# 基本运行
./legion/task_bench -steps 4 -width 4 -type trivial

# 不同依赖模式
./legion/task_bench -steps 4 -width 4 -type stencil_1d
./legion/task_bench -steps 4 -width 4 -type fft
./legion/task_bench -steps 9 -width 4 -type dom

# 不同内核类型
./legion/task_bench -kernel compute_bound -iter 1024
./legion/task_bench -kernel memory_bound -scratch 8192 -iter 16
./legion/task_bench -kernel load_imbalance -iter 1024
```

### 主要参数
- `-steps N`：时间步数
- `-width N`：每个时间步的任务宽度
- `-type TYPE`：依赖模式类型
- `-kernel TYPE`：内核类型
- `-iter N`：内核迭代次数
- `-scratch N`：scratch空间大小（字节）

## 添加CUDA STF支持的规划

基于对Task Bench架构的分析，我将按以下步骤添加CUDA STF支持：

### 阶段1：基础框架搭建
1. 创建`cudastf`目录
2. 实现基本的Makefile和构建配置
3. 创建主要的实现文件框架

### 阶段2：核心STF集成
1. 集成CUDA STF头文件和库
2. 实现TaskGraph到STF任务的映射
3. 实现基本的数据管理和依赖处理

### 阶段3：任务执行实现
1. 实现STF任务函数
2. 处理不同数量的输入依赖
3. 集成内核执行逻辑

### 阶段4：优化和测试
1. 性能优化
2. 正确性验证
3. 与其他实现的性能对比

这个规划将确保CUDA STF实现与现有框架保持一致的接口和行为模式。
