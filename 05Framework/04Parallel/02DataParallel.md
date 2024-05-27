<!--适用于[License](https://github.com/chenzomi12/AISystem/blob/main/LICENSE)版权许可-->

# 数据并行

<<<<<<< Updated upstream
**数据并行**$^{[1]}$ 是一种将原本在单一设备上进行的数据训练过程，扩展到多个设备并行计算，以达到理想情况下设备数量倍率加速效果的并行算法。数据并行的方式和扩展有很多种，例如：**数据并行（DP）**、**分布式数据并行（DDP）**、**完全分片的数据并行（ZeRO）**$^{[2]}$、**异步的数据并行**、**弹性训练**$^{[3]}$ 等。
=======
<<<<<<< HEAD
**数据并行**是一种广泛应用于分布式 AI 系统中的技术，旨在通过将数据集划分为多个子集并在不同计算节点上并行处理这些子集，以提高计算效率和速度。在大规模机器学习和深度学习训练过程中，数据并行可以显著加快模型训练速度，减少训练时间，提升模型性能。大部分的数据并行模型中，每个计算节点都会接收到完整的模型副本，但处理不同的数据子集。通过这种方法，计算任务可以被分摊到多个节点上，从而显著提高处理速度和效率。数据并行的实现方式多种多样，按照同步方式进行分类，包括**同步数据并行**和**异步数据并行**。同步数据并行要求所有计算节点在每一轮迭代后同步其参数，确保模型的一致性。而异步数据并行则允许节点独立进行计算和参数更新，从而减少等待时间，但也可能带来参数不一致的问题。按照实现方式进行分类，包括**数据并行**、**分布式数据并行**、**完全分片的数据并行**、**异步的数据并行**、**弹性数据并行**以及**参数服务器**。在本章中，我们集中关注与 pytorch 框架相结合的数据并行算法。
=======
**数据并行**$^{[1]}$ 是一种将原本在单一设备上进行的数据训练过程，扩展到多个设备并行计算，以达到理想情况下设备数量倍率加速效果的并行算法。数据并行的方式和扩展有很多种，例如：**数据并行（DP）**、**分布式数据并行（DDP）**、**完全分片的数据并行（ZeRO）**$^{[2]}$、**异步的数据并行**、**弹性训练**$^{[3]}$ 等。
>>>>>>> 3bf4f1ae2c2bb4f1ff287cfbffacd1a6ca85a938
>>>>>>> Stashed changes

## 数据并行

数据并行（Data Parallelism, DP）的核心思想是将大规模的数据集分割成若干个较小的数据子集，并将这些子集分配到不同的计算节点上，每个节点运行相同的模型副本，但处理不同的数据子集。在每一轮训练结束后，各节点会将计算得到的梯度进行汇总，并更新模型参数。这样，每个节点都能在下一轮训练中使用更新后的模型参数，从而保证整个模型在所有节点上保持一致。

数据并行只能在单台机器上运行，采用单进程、多线程的实现方式，将原本在单一设备上进行的数据训练过程，扩展到多个设备并行训练。在某一设备上随机初始化模型和优化器后，就可进行数据并行的训练，算法可分为三个步骤：

- **前向传播**：将 mini-batch 数据平均分配到每个设备上。接下来进行分布式初始化，将模型和优化器复制到每个设备上，保证各设备的模型、优化器完全相同。初始化完成后，各设备根据分配到的数据和模型同时进行前向传播。
- **损失计算与反向传播**：前向传播完成后，每个设备分别计算模型损失并进行反向传播。得到梯度后，将梯度传递到某一设备进行累加，更新模型的参数和优化器状态。更新后的模型参数和优化器将会在下一轮的前向传播中被复制到每个设备上。
- **重复**：上述步骤重复进行，直到模型收敛或者达到预定的训练轮数。

![数据并行](images/02DataParallel01.png)
:width:`650px`

但由于数据并行相对来说还不够完善，造成了许多性能的浪费。如在**语言层面**，使用作为最热门的深度学习开发语言 Python，在数据并行中采用的单进程、多线程并行方式往往受到 GIL（全局解释器锁）限制，CPU 的性能瓶颈使得多线程不能良好的利用多设备的资源。另外在**算法层面**，全局的梯度累积和参数更新发生在一个设备上，会出现明显的单个设备利用率更高，其他设备空闲的情况，造成了资源的浪费。同时如果在数据并行中的 mini-batch 设置过小，将导致设备内并行度不足，从而降低训练速度；在通信开销的影响下，甚至可能出现比单设备慢的情况。

## 分布式数据并行

分布式数据并行（Distributed Data Parallel, DDP）是数据并行的一种高级形式，它综合了多种优化，是当前应用最广的并行算法之一，通常用于大型集群和多 GPU 系统中。DDP 在每个 GPU 上创建一个模型副本，并在每个训练步骤结束时，通过高效的梯度聚合和参数同步机制，确保模型的一致性。除此以外，DDP 针对数据并行的缺点做了许多改进，并拥有良好的的扩展性，如：完全分片的数据并行就是基于分布式数据并行的内存高效扩展版本。具体来说，分布式数据并行使用了**多进程**的实现方式，这避免了开发语言层面 Python GIL 的限制，也将并行规模扩展到多台网络连接的机器，进一步扩大分布式规模和效率；同时，针对通信做了大量优化，如使用**Ring-AllReduce 算法**和**延迟隐藏技术**进行高效的集合通信。分布式数据并行的各设备负载也更均衡，没有单独在某一个设备上工作的情况。

### DDP 基本流程

在分布式数据并行中，程序会启动设备数量个进程，每个进程单独启动一个主训练脚本副本。在开始时，主进程将模型从设备 0 复制到其余设备一次，保证各设备的模型、优化器完全相同，接下来是分布式数据并行的训练过程：

- **前向传播**：每个设备将分别拿到一块完整且不同的 mini-batch 数据，各设备根据分配到的数据同时进行前向传播。
- **损失计算与反向传播**：前向传播完成后，每个设备分别计算模型损失并进行反向传播与梯度更新。值得注意的是，分布式数据并行中反向传播和梯度更新的过程是同时进行的——一旦某些局部梯度准备就绪，它们就会在所有过程中取平均值（默认是使用 Ring-AllReduce 算法做集合通信），然后使用全局梯度更新模型参数和优化器状态。我们将在下一小节具体介绍有关**计算与通信的重叠**的内容。梯度的一致性可确保各设备的模型保持一致，避免使用其他模型的梯度进行参数更新而导致收敛问题。在**异步的数据并行**中，我们还将会接着讨论模型不一致的情况，这将会带来一定的收敛问题，但是可以使整个迭代过程更快，同时设备的利用率更高。
- **重复**：上述步骤重复进行，直到模型收敛或者达到预定的训练轮数。

### DDP 实现分析

数据并行是分布式训练中最基础和常见的并行算法。本节将重点介绍分布式数据并行（DDP）在 PyTorch 中的简单实现示例，并对数据并行的各个关键步骤如前向传播、反向传播、梯度更新等进行详细分析，以更深入地理解数据并行的实现原理和具体执行步骤。

这是在分布式环境下使用 2 块设备训练简单网络的完整例子：

我们首先需要导入了实现分布式数据并行训练所需的库。包括 PyTorch 的核心库 torch、神经网络模块 torch.nn、优化器模块 torch.optim、分布式启动模块 torch.distributed 和多进程模块 torch.multiprocessing。

```python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
```

我们首先指使用 `dist.init_process_group` 初始化进程组。`example` 函数实现了分布式数据并行训练的主要逻辑。该函数首先加载模型并将其复制到当前进程的设备上，并使用 `torch.nn.parallel.DistributedDataParallel` 将其封装为分布式数据并行模型，同步不同进程的参数。对于每个数据，该函数将首先把数据移动到当前设备，然后前向传播计算模型输出，基于损失函数计算损失值并反向传播计算梯度，最后使用优化器更新模型参数。

<<<<<<< Updated upstream
接下来我们来看 Pytorch2.0 $^{[4]}$ 中分布式数据并行具体的实现方式，这里我们先不涉及 Pytorch2.0 或 torchdynamo 引入的编译部分，**分布式系统的编译优化**将在一个单独的章节中进行介绍。首先我们看看 DDP 的**初始化**与**前向传播**，以及在这个过程中是如何**维护模型一致性**的。
=======
<<<<<<< HEAD
```python
def example(rank, world_size):
    # create default process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    # create local model
    model = nn.Linear(10, 10).to(rank)
    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    # define loss function and optimizer
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(rank))
    labels = torch.randn(20, 10).to(rank)
    # backward pass
    loss_fn(outputs, labels).backward()
    # update parameters
    optimizer.step()

def main():
    world_size = 2
    mp.spawn(example,
        args=(world_size,),
        nprocs=world_size,
        join=True)

if __name__=="__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
```

作为分布式的启动函数，`main` 利用 `torch.multiprocessing.spawn` 启动指定数量的进程，并在每个进程中运行传入的函数。在主程序入口处，`main` 被调用并传入了 `example` 函数和进程数 2，因此实现了在 2 个设备上进行分布式数据并行训练。在真实环境中，我们还会使用 `DataLoader` 和 `DistributedSampler` 进行高效的分布式数据加载。接下来我们进行系统的分析。

#### DDP 前向传播

接下来我们来看 Pytorch2.0 中分布式数据并行具体的实现方式，这里我们先不涉及 Pytorch2.0 或 torchdynamo 引入的编译部分，**分布式系统的编译优化**将在一个单独的章节中进行介绍。首先我们看看 DDP 的**初始化**与**前向传播**，以及在这个过程中是如何**维护模型一致性**的。
=======
接下来我们来看 Pytorch2.0 $^{[4]}$ 中分布式数据并行具体的实现方式，这里我们先不涉及 Pytorch2.0 或 torchdynamo 引入的编译部分，**分布式系统的编译优化**将在一个单独的章节中进行介绍。首先我们看看 DDP 的**初始化**与**前向传播**，以及在这个过程中是如何**维护模型一致性**的。
>>>>>>> 3bf4f1ae2c2bb4f1ff287cfbffacd1a6ca85a938
>>>>>>> Stashed changes

模型的一致性要求每次进行的前向传播每个进程的参数需要相同。它依赖于 torch.nn.Module 类和 DistributedDataParallel 类，在 PyTorch 中，所有的模型都会继承 Module 类（包括分布式数据并行类 DistributedDataParallel）。其中我们需要关注的是 Module 类中的两个类变量 `_parameters` 和 `_buffers`，`_parameters` 是指网络的参数，`_buffers`不是参数，但也是会被持久化保存的数据，如 BatchNorm 中的 mean 和 variance。

```python
# torch.nn.modules.py
class Module:
    ...
    _parameters: Dict[str, Optional[Parameter]]
    _buffers: Dict[str, Optional[Tensor]]
    ...
```

DDP 在构建时，会通过 `_sync_module_states` 同步各个进程的模型参数，包括`_parameters` 和 `_buffers`以达到模型的一致性。

```python
# torch.nn.parallel.distributed.py 
class DistributedDataParallel(Module, Joinable):
    ...
    def __init__(
        ...
        # Sync params and buffers. Ensures all DDP models start off at the same value.
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
        )
        ...
```

同时，在每次网络传播开始前，DDP 也都会通过 `_sync_module_states` 同步进程之间的 `buffer`，维持状态的统一。

```python
# torch.nn.parallel.distributed.py 
class DistributedDataParallel(Module, Joinable):
    ...
    def forward(self, *inputs, **kwargs):
        ...
        # Sync params and buffers. Ensures all DDP models start off at the same value.
        _sync_module_states(
            module=self.module,
            process_group=self.process_group,
            broadcast_bucket_size=self.broadcast_bucket_size,
            src=0,
            params_and_buffers_to_ignore=self.parameters_to_ignore,
        )
        ...
```

#### DDP 计算与通信的重叠

在分布式数据并行（DDP）中，一项重要的优化是在反向传播过程中同时进行参数更新，这一过程也被称为计算与通信的重叠。在分布式训练中，每个进程通常会在完成当前网络反向传播的同时进行梯度更新，以隐藏通信延迟。在部分梯度计算完成后，即可立即进行通信，一般通过钩子函数来实现。在通信的同时也会继续计算梯度，这样就无需等待所有计算完成后再集中进行通信，也不必在计算完成后等待通信完成，从而将通信过程覆盖到计算时间内，充分利用设备，提高了设备使用率。

![数据并行](images/02DataParallel02.png)
:width:`650px`

这里我们同样使用 Pytorch2.0 进行举例。在此过程中涉及到钩子函数 `hook`、参数桶 `bucket` 和归约管理器 `reducer` 三个关键部分。

钩子函数 `hook` 是在 `torch.Tensor` 上实现的，每次计算相对于张量的梯度时都会调用该钩子。通过钩子函数，当张量梯度计算完成后，就可以立即进行集合通信。需要注意的是，虽然 DDP 的关键代码是用 C++ 实现的，但在 C++ 和 Python 代码中，`Tensor` 都提供了相似的 hook 接口，实现了类似的功能。

![数据并行](images/02DataParallel03.png)
:width:`650px`

```python
# torch._tensor.py
class Tensor(torch._C._TensorBase):
    ...
    def register_hook(self, hook):
        r"""Registers a backward hook.

        The hook will be called every time a gradient with respect to the
        Tensor is computed. 
        ...
```

Pytorch 使用归约管理器 `reducer` 在反向传播期间进行梯度同步。为提高通信效率，`reducer` 将参数梯度组织到多个桶 `buckets` 中，并对每个桶进行集合通信（可通过在 DDP 构造函数中设置 `bucket_cap_mb` 参数来配置桶大小）。其中参数梯度到桶的映射，在构造时基于桶大小限制和参数大小确定。模型参数按照给定模型 `Model.parameters()` 的大致相反顺序分配到桶中（使用相反顺序的原因是 DDP 期望在反向传播时以大致相同的顺序准备好梯度）。示例图展示了一个场景，其中 $g_{w2}$ 和 $g_{b2}$ 在 bucket1 中，另外两个梯度在 bucket2 中。虽然这种假设可能不总是成立，一旦发生，将损害 DDP 反向传播的速度，因为 reducer 无法在最早可能的时间启动通信。除了分桶，reducer 在构造阶段为每个参数注册了 autograd 钩子，在反向传播时当梯度准备就绪时触发这些钩子。Pytorch 使用 `_ddp_init_helper` 函数，进行参数的 `reducer` 的初始化以及参数的装桶。

```python
# torch.nn.parallel.distributed.py 
class DistributedDataParallel(Module, Joinable):
    ...
    def __init__(
        ...
        # Builds reducer.
        self._ddp_init_helper(
            parameters,
            expect_sparse_gradient,
            param_to_name_mapping,
            static_graph,
        )
        ...
    ...
    def _ddp_init_helper(
        self,
        parameters,
        expect_sparse_gradient,
        param_to_name_mapping,
        static_graph,
    ):
        """
        Initialization helper function that does the following:
        (1) bucketing the parameters for reductions
        (2) resetting the bucketing states
        (3) registering the grad hooks
        (4) Logging construction-time DDP logging data
        (5) passing a handle of DDP to SyncBatchNorm Layer
        """
        ...
```

如果一个参数在前向传播中没有被使用，当前参数的桶会在反向传播时永远等待缺失的梯度。如果设置了 `find_unused_parameters` 为 True，DDP 会分析来自本地模型的输出，从而确定在模型的子图上运行反向传播时哪些参数参与了计算。DDP 通过从模型输出遍历 autograd 图来找出未使用的参数，并将其标记为可供 reduce。在反向传播期间，reducer 只会等待未就绪的参数，但它仍会对所有桶进行 reduce 操作。将参数梯度标记为就绪不会帮助 DDP 跳过桶，但会防止其在反向传播时永远等待缺失的梯度。值得注意的是，遍历 autograd 图会带来额外开销，因此只有在必要时才应将 `find_unused_parameters` 设置为 True。

由于反向传播的函数 `backward` 直接在损失张量上调用，这超出了 DDP 的控制范围。DDP 使用在构造时注册的 autograd 钩子来触发梯度同步。当一个梯度准备就绪时，相应的 DDP 钩子会被触发，DDP 将标记该参数梯度为就绪可供 reduce。当一个桶中的所有梯度都准备就绪时，reducer 将在该桶上启动异步 allreduce 操作以计算所有进程中梯度的平均值。当所有桶都就绪时，reducer 将阻塞等待所有 allreduce 操作完成。完成后，平均梯度将被写入所有参数的 `param.grad` 字段。因此，在反向传播之后，不同 DDP 进程上相同的参数其 `grad` 字段应该是相同的。在之后的优化器步骤中，所有 DDP 进程上的模型副本可以保持同步，因为它们都从同一个状态开始，并且在每次迭代中具有相同的平均梯度。

#### DDP 数据加载

我们所使用的 `DataLoader` 是一个迭代器，在加载 `__iter__` 方法时，会根据进程数量选择对应的迭代器并赋值给类变量 `_iterator`，迭代器种类分为 `_SingleProcessDataLoaderIter` 和 `_MultiProcessingDataLoaderIter`，其中 `_MultiProcessingDataLoaderIter` 负责多进程的数据读取。

```python
# torch.utils.dat.dataLoader.py
class DataLoader(Generic[T_co]):
    ...
    def __iter__(self) -> '_BaseDataLoaderIter':
        ...
        if self.persistent_workers and self.num_workers > 0:
            if self._iterator is None:
                self._iterator = self._get_iterator()
            else:
                self._iterator._reset(self)
            return self._iterator
        else:
            return self._get_iterator()
    ...
    def _get_iterator(self) -> '_BaseDataLoaderIter':
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            self.check_worker_number_rationality()
            return _MultiProcessingDataLoaderIter(self)
```

在获取数据时，这些迭代器会调用使用 `_reset` 初始化 sampler，然后通过 `_next_data` 方法获取数据。

```python
    ...
    def __next__(self) -> Any:
        with torch.autograd.profiler.record_function(self._profile_name):
            if self._sampler_iter is None:
                # TODO(https://github.com/pytorch/pytorch/issues/76750)
                self._reset()  # type: ignore[call-arg]
            data = self._next_data()
            ...
```

在 `_MultiProcessingDataLoaderIter` 中，会加载多个进程，主进程负责维护一个索引队列（index_queue），工作进程从索引队列中获取数据索引，然后从数据集中加载数据并进行预处理。处理后的数据被放入结果队列（worker_result_queue）中，供主进程使用。

```python
# torch.utils.data.dataLoader.py
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        ...
        for i in range(self._num_workers):
            # No certainty which module multiprocessing_context is
            index_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
            # Need to `cancel_join_thread` here!
            # See sections (2) and (3b) above.
            index_queue.cancel_join_thread()
            w = multiprocessing_context.Process(
                target=_utils.worker._worker_loop,
                args=(self._dataset_kind, self._dataset, index_queue,
                        self._worker_result_queue, self._workers_done_event,
                        self._auto_collation, self._collate_fn, self._drop_last,
                        self._base_seed, self._worker_init_fn, i, self._num_workers,
                        self._persistent_workers, self._shared_seed))
            w.daemon = True
            w.start()
            self._index_queues.append(index_queue)
            self._workers.append(w)
        ...
```

其中每一个进程都运行 `_worker_loop` 函数，从 index_queue 中获取 index，而后从 Dataset 中获取对应的数据。

```python
# torch.utils.data._utils.worker.py
def _worker_loop(dataset_kind, dataset, index_queue, data_queue, done_event,
                 auto_collation, collate_fn, drop_last, base_seed, init_fn, worker_id,
                 num_workers, persistent_workers, shared_seed):
    ...
        while watchdog.is_alive():
            try:
                r = index_queue.get(timeout=MP_STATUS_CHECK_INTERVAL)
            except queue.Empty:
                continue
            ...
            idx, index = r
            ...
            try:
                data = fetcher.fetch(index)
            except Exception as e:
                if isinstance(e, StopIteration) and dataset_kind == _DatasetKind.Iterable:
                    data = _IterableDatasetStopIteration(worker_id)
                    # Set `iteration_end`
                    #   (1) to save future `next(...)` calls, and
                    #   (2) to avoid sending multiple `_IterableDatasetStopIteration`s.
                    iteration_end = True
                else:
                    # It is important that we don't store exc_info in a variable.
                    # `ExceptionWrapper` does the correct thing.
                    # See NOTE [ Python Traceback Reference Cycle Problem ]
                    data = ExceptionWrapper(
                        where=f"in DataLoader worker process {worker_id}")
            data_queue.put((idx, data))
```

值得注意的是，每当处理完一个 batch，就需要调用 `_process_data` 将一个待处理的 batch 放入 `_index_queue` 中等待某个进程来处理。这可以使得，在使用当前批次的数据进行训练时，同时加载下一个批次的数据，而不需要在下一次迭代开始使再进行数据的加载，将数据加载的等待时间大大缩减。

```python
# torch.utils.data.dataLoader.py
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def _next_data(self):
        while True:
            ...
            # Check if the next sample has already been generated
            if len(self._task_info[self._rcvd_idx]) == 2:
                data = self._task_info.pop(self._rcvd_idx)[1]
                return self._process_data(data)

    def _process_data(self, data):
        self._rcvd_idx += 1
        self._try_put_index()
        if isinstance(data, ExceptionWrapper):
            data.reraise()
        return data
```

如果设置了 pin_memory=True，则主进程会启动一个内存固定线程，该线程从结果队列中获取数据，并使用 `_pin_memory_loop` 将其复制到设备内存中。复制后的数据被放入数据队列中，供主进程使用。

```python
# torch.utils.data.dataLoader.py
class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
    def __init__(self, loader):
        ...
            pin_memory_thread = threading.Thread(
                target=_utils.pin_memory._pin_memory_loop,
                args=(self._worker_result_queue, self._data_queue,
                      current_device,
                      self._pin_memory_thread_done_event, self._pin_memory_device))
```

在分布式环境下，我们通过 DistributedSampler 可以获取到基于设备索引的数据切分，这样就确保了每个设备可以拿到不同的数据。

```python
# torch.utils.data.distributed.py
class DistributedSampler(Sampler[T_co]):
    def __iter__(self) -> Iterator[T_co]:
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]
        ...

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
```

#### DDP 性能分析

我们使用 `torch.profiler.profile` 对 DDP 的过程进行性能分析。只需要对训练的循环进行简单嵌套，就能得到清晰的具体分析结果。

| Configuration | GPU Summary |
| --- | --- |
| Number of Worker(s): 2 | Name: Tesla V100-SXM2-16GB |
| Device Type: GPU | Compute Capability: 7.0 |

这里使用了两张 V100-SXM2-16GB 作为设备并使用 NV-Link 连接，通过 CIFAR10 训练 ResNet50 网络。

![DDP 性能分析](images\02DataParallel04.png)
:width:`650px`

从 `profile` 对 ResNet50 的性能分析结果可以看到，计算与通信的重叠几乎覆盖了整个反向传播的过程（反向传播的计算时间约为前向传播的两倍，图中重叠的部分约为只计算部分的两倍，只通信的部分可以忽略不记）

![DDP 性能分析](images\02DataParallel05.png)
:width:`650px`

同样，在追踪视图中，我们可以看到反向传播的主要计算函数 `autograd::engine::evaluate_function:ConvolutionBackward0` 与集合通信的函数 `nccl:all_reduce` 执行是重叠的。

DDP 反向传播中计算与通信的重叠导致无需等待所有计算完成后再集中进行通信，也不必在计算完成后等待通信完成，提高了设备使用率。

## 完全分片的数据并行

完全分片的数据并行（Fully Sharded Data Parallel，FSDP）在分布式 AI 系统中具有重要地位，不仅能提高并行效率，还能减少显式内存消耗，这两方面的优势为模型的大规模训练带来了显著的好处。值得注意的是，并行效率和内存消耗之间存在着密切关联关系，降低内存占用可以使我们使用更大的并行度，进而提升整体的并行加速效果。完全分片的数据并行是基于零冗余优化器（ZeRO）的，主要的实现有微软的 **Deepspeed** 和 Meta 的 **Fairscale**，其中 Fairscale 被集成到 Pytorch 中，并作为 FSDP 实现基础。在本节中，我们将从零冗余优化器的常用技术入手，深入剖析如何降低内存开销并提高训练效率。

在开始前，我们需要一些前置知识，混精度训练和显存消耗估算，以帮助我们更好地理解完全分片的数据并行算法。

### 混精度训练

在当今大规模模型训练的背景下，混合精度训练已然成为一种备受推崇的普遍做法。通过采用混合精度训练，我们能够将训练速度显著提升数倍，而又不会对模型的整体性能产生重大影响。在数据科学领域，精度一直是评判的重要考量因素——在传统的科学计算领域，人们通常追求较高的精度，如 FP128 或 FP64 等。然而，在深度学习中，我们所面临的实际上是一个高维函数拟合（或近似）的优化问题，因此并不需要过于精确的数值表示，且使用低精度将会带来显著的计算速度提升：在 NVIDIA A00 SXM 与 NVIDIA H00 SXM 中， FP16 浮点运算能力的理论峰值是 FP32 的近 30 倍。

<center>

| GPU 型号         | NVIDIA H100 SXM 80GB  | NVIDIA A100 SXM 80GB |
| ---------------- | -------------------- | -------------------- |
| FP32             | 67 TFLOPS            | 19.5 TFLOPS          |
| TF32 Tensor Core | 989 TFLOPS           | 312 TFLOPS           |
| FP16 Tensor Core | 1,979 TFLOPS         | 624 TFLOPS           |

</center>

#### 常用精度

在深度学习中，常用的精度包括 **FP32**、**FP16**、**BF16** 和 **TF32**。

![常用精度 FP32 FP16 BF16](images/02DataParallel06.png)
:width:`650px`

**FP32**：这种格式在很长一段时间内都是深度学习的主力，它是 IEEE 754 标准下的单精度浮点数。长期以来，它一直是神经网络计算的标准类型。长期以来，神经网络中的权重、激活和其他值都默认用 FP32 表示。

**FP16**：同样是 IEEE 754 标准下的半精度浮点格式。随着深度学习的发展，FP16 逐渐取代了 FP32 的地位。因为相较于 FP32，更低的精度并不会对神经网络的性能产生重大影响。额外的精度不会带来任何好处，反而会更慢、占用更多内存并降低通信速度。FP16 通常用于混合精度训练（TensorFlow/PyTorch）。也用于训练后量化，以加快推理速度（TensorFlow Lite）。其他用于量化的格式还有整数 INT8（8 位整数）、INT4（4 位）甚至 INT1（二进制值）。

**BF16**：谷歌最初开发的另一种 16 位格式被称为 "Brain Floating Point Format"，简称 "bfloat16"。这个名字源于 "谷歌大脑"（Google Brain），谷歌大脑是谷歌的一个人工智能研究小组，这种格式就是在这个小组构思出来的。最开始是被使用在 Google 芯片 TPU 中，后被广泛使用在 GPU 中。由于具有更多的指数位，常被用于处理 FP16 的溢出问题。

![常用精度 TF32](images/02DataParallel07.png)
:width:`650px`

**TF32**：NVIDIA 在 Ampere GPU 后引入的新数学模式，这是一种十分特殊的格式（无需显示设置，而是自动执行），它将 FP32 数据截断为 TF32 进行计算，然后再转换回 FP32。这一创新的最大优势在于编译器只需在最底层（即 CUDA 编译器内部）提供支持。其他代码部分则可以继续使用动态范围相同但精度较高的 FP32，无需进行修改。TF32 的快速插入性使得利用 Tensor Core 的速度成为可能，而无需过多修改现有代码。

```python
# The flag below controls whether to allow TF32 on matmul. This flag defaults to False
# in PyTorch 1.12 and later.
torch.backends.cuda.matmul.allow_tf32 = True

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = True
```

TF32 采用与半精度（FP16）相同的 10 位尾数，这满足了人工智能工作负载的精度要求，并且使用了与 FP32 相同的 8 位指数，因此具有相同的数值范围。从技术上讲，它可以视为一种 19 位格式，也可以视为扩展精度的 BF16。TF32 的优势在于其格式与 FP32 相同。当使用 TF32 进行计算时，输入 FP32 操作数的尾数从 23 位舍入到 10 位，然后进行精确乘法运算，最后以正常的 FP32 格式进行累加。TF32 Tensor Core 在 FP32 输入上运行并生成 FP32 结果，而不需要修改代码，而非矩阵操作则继续使用 FP32。相比之下，FP16 和 BF16 等格式需要更多工作，因为它们涉及不同的位布局。尽管如此，也值得使用这些格式，因为它们可以减少内存带宽，从而提升执行速度。

#### 混精度训练

在深度学习中，使用半精度（FP16）训练有时会出现下溢出的问题：FP16 的有效的动态范围约为 ${5.96e}^{-8} \sim 65504$，在训练后期，例如激活函数的梯度会非常小，甚至在梯度乘以学习率后，值会更加小。由于 FP16 的精度范围有限，过小的梯度可能导致更新无效——这个时候就需要我们使用混精度训练。混精度训练可以分为两个部分：**半精度** 和 **权重备份**，这里我们拿 FP16 和 FP32 来举例。在训练开始时，我们准备两套模型状态，其中一套为 FP32 类型（优化器状态和模型参数），另一套为 FP16 类型（模型参数），在前向传播、反向传播时，我们都使用 FP16 类型的模型参数进行计算；而在参数更新时，我们将梯度成与学习率 $\eta$ 相乘，更新到 FP32 类型的模型状态上，在新一轮的训练中，我们再次将 FP32 类型的模型拷贝为 FP16 类型的模型。这个过程就是**混精度训练**。由于在计算密集的前向传播、反向传播中，我们使用了半精度进行计算，与单精度相比，训练的速度会大幅度提升。另外，由于激活值在训练过程中占用内存的很大部分，使用 FP16 储存激活值在大批量训练时也会节省内存。同时，在分布式环境下使用 FP16 梯度通信量也会降低。

![混精度训练](images/02DataParallel08.png)
:width:`650px`

为了获得最佳性能，在混精度中我们需要额外选择合适的批量大小。通常建议使用 2 的幂次方作为批量大小，并与输入/输出神经元的数量相匹配，通常为 8 的倍数，但也可能更高，具体取决于所使用的硬件和模型的数据类型。NVIDIA 为全连接层提供了关于输入/输出神经元计数和批量大小选择的建议。根据数据类型和硬件的不同，tensor core 的要求也不尽相同。以 FP16 数据类型为例，通常建议使用 8 的倍数作为批量大小，除非是在 A100 GPU 上，在这种情况下应使用 64 的倍数。这是因为 GPU 对 FP16 数据的并行计算方式决定的。GPU 通常会以 128 位（16 个字节）为一组同时处理多个 FP16 数据。为了保证计算效率，张量在内存中的排布需要与 GPU 的计算方式相匹配，即维度需要是 8 的整数倍。

而在完全分片的数据并行（FSDP）中，我们可以通过在 torch 中指定 `fpSixteen` 进行混精度的自动配置。

```python
fpSixteen = MixedPrecision(
    param_dtype=torch.float16,
    # Gradient communication precision.
    reduce_dtype=torch.float16,
    # Buffer precision.
    buffer_dtype=torch.float16,
)

bfSixteen = MixedPrecision(
    param_dtype=torch.bfloat16,
    # Gradient communication precision.
    reduce_dtype=torch.bfloat16,
    # Buffer precision.
    buffer_dtype=torch.bfloat16,
)

model = FSDP(model,
       auto_wrap_policy=t5_auto_wrap_policy,
       mixed_precision=bfSixteen)
```

#### 损失缩放 （Loss Scale）

解决半精度（FP16）下溢问题的另一个方法是损失缩放（Loss Scale）。刚才提到，训练到了后期，梯度（特别是激活函数平滑段的梯度）会特别小，半精度（FP16）表示容易产生下溢现象。为了解决梯度过小的问题，我们需要对损失进行缩放，由于链式法则的存在，损失的缩放也会作用在梯度上。缩放过后的梯度，就会平移到半精度（FP16）有效的展示范围内。不过缩放并非对于所有网络而言都是必须的，而缩放的取值为也会特别大，一般在 8 - 32k 之间。在 Pytorch 中，可以通过这样的方式实现自动损失缩放：

```python
from torch.cuda.amp import GradScaler, autocast
scaler = GradScaler()

with autocast():
    output = model(input)
    loss = loss_fn(output, target)

scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

其中这种损失缩放的方式是动态的，每当梯度溢出时候减少损失缩放规模，并且间歇性地尝试增加损失规模，从而实现在不引起溢出的情况下使用最高损失缩放因子，更好地恢复精度。

**动态损失缩放**的算法会从比较高的缩放因子开始（如 $2^{24}$），然后开始进行训练，并在迭代中检查数是否会溢出（Infs/Nans）；如果没有梯度溢出，则不调整缩放因子，继续进行迭代；如果检测到梯度溢出，则缩放因子会减半，重新确认梯度更新情况，直到参数不出现在溢出的范围内；在训练的后期，loss 已经趋近收敛稳定，梯度更新的幅度往往小了，这个时候可以允许更高的损失缩放因子来再次防止数据下溢。

#### 内存消耗估算

在深度学习模型的训练中，合理估算和管理内存消耗是非常重要的。我们的内存存储主要分为两大块：**模型状态（Model States）** 和**剩余状态（Residual States）**。

**模型状态**指和模型本身相关的，必须存储的内容，具体包括：

- 优化器状态（Optimizer States）：Adam 优化算法中的 Momentum 和 Variance

- 梯度（Gradients）：模型梯度 G

- 参数（Parameters）：模型参数 W

**剩余状态**是并非模型必须的，但在训练过程中会额外产生的内容，具体包括：

- 激活值（Activation）：在反向传播过程中使用链式法则计算梯度时会用到。有了它算梯度会更快，但它不是必须存储的，因为可以通过重新前向传播来计算。

- 临时存储（Temporary Buffers）: 例如把梯度发送到某个设备上进行 All-Reduce 时产生的存储。

- 碎片化的存储空间（Unusable Fragment Memory）：虽然总存储空间是够的，但是如果取不到连续的存储空间相关的请求也会失败。对这类空间浪费可以通过内存整理来解决。

拿 FP32 与 FP16 的混合精度训练举例，假设模型的参数量是 $\Phi$，那么模型状态所消耗的空间为：

<center>

| Model States                 | Size (Byte) |
|----------------------------- |------------ |
| FP32 Parameters              | 4 $\Phi$    |
| FP32 Adam Optimizer Momentum | 4 $\Phi$    |
| FP32 Adam Optimizer Variance | 4 $\Phi$    |
| FP16 Gradients               | 2 $\Phi$    |
| FP16 Parameters              | 2 $\Phi$    |
| Total                        | 16 $\Phi$   |

</center>

而由于剩余状态和具体模型架构有关，因此需要具体分析。

接下来我们基于 Transformer 的架构进行具体分析，因为所有参数超过 10 亿的 SOTA 模型都遵循这一架构。我们的分析假设使用 Adam 优化器进行混合精度训练，因为此配方是训练基于 Transformer 的模型的事实标准。

**模型状态**：模型状态由优化器状态、梯度和参数组成。基于 Transformer 的模型中的参数总数主要取决于隐藏维度 （$ℎd$） 和 Transformer 层数（$nl$）。 Transformer 块中的几乎所有参数都来自每个块内的四个线性层，其大小分别为：（$ℎd$, $3ℎd$）、（$ℎd$, $ℎd$）、（$ℎd$, $4ℎd$） 和（$4ℎd$, $ℎd$）。因此，基于 Transformer 的模型中的总参数可以近似为：

$$
12 × nl × hd^2
$$

**剩余状态**：剩余状态主要是指激活内存，它取决于模型结构、批量大小（$𝑏𝑠𝑧$）和序列长度（$𝑠𝑒𝑞$），而且可能相当大。不过激活所需的内存可以通过激活检查点（activation checkpointing）大大减少，我们假设 $𝑐𝑖$ 是两个激活检查点之间的 Transformer 块数，$𝑏𝑠𝑧 × 𝑠𝑒𝑞 × ℎ𝑑$ 是每个 Transformer 块的输入大小，激活检查点所需的内存估计为：

$$
2 × 𝑏𝑠𝑧 × 𝑠𝑒𝑞 × ℎ𝑑 × 𝑛𝑙 / 𝑐𝑖
$$

**激活工作内存（AWM）**：激活工作内存是反向传播过程中所需的内存，用于在执行实际反向传播之前重新计算激活。是两个连续激活检查点之间的激活量。例如，如果我们为每个 Transformer 块创建一个激活检查点，那么内存就是每个 Transformer 块的总激活量。其字节数约为：

$$
𝑏𝑠𝑧 × 𝑠𝑒𝑞 × 𝑐𝑖 × (16 × ℎ𝑑 + 2 × 𝑎𝑡𝑡𝑛\_ℎ𝑒𝑎𝑑𝑠 × 𝑠𝑒𝑞)
$$

**模型状态工作内存（MSWM）**：模型状态工作内存是指在将所有模型状态卸载到 CPU 或 NVMe 之后，对模型中最大的单个算子执行前向或后向传播所需的 GPU 内存最小量。这大约是由模型中该算子的参数和梯度的大小决定的，因为必须至少有足够的内存来保存向后传播的参数及其梯度。Transformer 的最大的算子是将隐藏状态从 $ℎ$ 转换为 $4ℎ$ 的线性层。该线性层的参数和梯度的大小为：

$$
4 × ℎ𝑑 × 4ℎ𝑑
$$

### 零冗余优化器（Zero Redundancy Optimizer，ZeRO）

在数据并行（DP）中，每个设备都需要保存一份完整的参数（模型状态和剩余状态），而不是所有的参数在训练的整个过程中都会被使用到，而是在特定的阶段中（某个层的前向或反向传播），因此我们可以在不需要使用的时候将它转移到其他地方节省内存空间。ZeRO 有两套优化方案：ZeRO-DP，旨在减少模型状态的内存占用。ZeRO-R，旨在减少剩余状态内存消耗。我们将详细阐述这些优化及其背后的启示，这些优化使 ZeRO 能够在保持高效的同时减少内存占用。

### ZeRO-DP

ZeRO-DP 对模型状态进行切分，具体来说，每个设备都只会会存储 $\frac{1}{N_d}$ 的模型状态（其中 $N_d$ 为并行度），在需要时通过集合通讯 All-Gather 获取参数。ZeRO-DP 保留了数据并行训练（DP）的高效率，同时实现了模型并行（MP）的内存效率优势。由于数据并行的模型状态在所有数据并行进程中冗余存储，因此内存效率低下，但数据并行具有更高的计算粒度和更低的通信量，从而具有更高的训练效率。模型并行的通信开销很大，因此可扩展性比数据并行低，但模型并行对模型状态进行分区，获得了较高的内存效率。ZeRO-DP 对模型状态进行分区而不是复制它们，并使用动态通信调度最小化通信量。通过这样做，ZeRO-DP 随着数据并行程度的增加线性减少模型在每块设备的内存占用，同时保持通信量接近默认数据并行的通信量，从而保持效率。
ZeRO-DP 有三个主要优化阶段，分别对应于优化器状态、梯度和参数的划分，在累积启用时：

1) **优化状态分区**（Partition optimizer states，$P_{os}$）：又称为 ZeRO-1，将优化器状态按并行度均匀分区，每个进程只需存储 $\frac{1}{N_d}$ 的优化器状态（其中 $N_d$ 为并行度）。这可将内存消耗减少到 1 / 4，且无额外通信开销。

2) **添加梯度分区**（Partition gradients，$P_{os+g}$）：又称为 ZeRO-2，在优化器状态分区的基础上，对梯度也进行分区。每个进程只需存储用于更新自身参数分区所需的梯度。这可减少 8 倍的内存消耗，且无额外通信开销。

3) **添加参数分区**（Partition parameters，$P_{os+g+p}$）：又称为 ZeRO-3，在优化器状态和梯度分区的基础上，对参数也进行分区。每个进程只存储自身的参数分区，在前向反向传播时需要从其他进程收集所需的参数分区。这会使通信量增加约 50%，但可以实现与并行度 $N_d$ 成正比的内存减少。

![数据并行](images/02DataParallel09.png)
:width:`650px`

通过这三个阶段的优化，ZeRO-DP 最终能够在保持数据并行高效的同时，将每个设备的内存消耗降低至 $\frac{1}{N_d}$ 的水平，使得利用少量硬件资源训练万亿参数等超大模型成为可能，接下来我们进行每个阶段的详细介绍。这里我们假设模型使用混精度训练，模型参数量为 4 $\Psi$。

#### ZeRO-1 计算内存分析

根据之前的介绍我们知道，优化器状态是训练过程中设备内存中的主要保存内容，但只有在参数更新的时候会被用到。ZeRO-1 的核心思想是将优化器状态分布到多个设备上，减少每个设备所需的显存，在需要参数更新时进行聚合。

![数据并行](images/02DataParallel10.png)
:width:`650px`

1. **数据分片（a）**：从优化器状态分片开始，将优化器状态分成 N 份，每个设备保存一份分片，并将训练批次数据（batch data）分成 N 份，每个设备处理一份数据。
   
2. **前向与后向计算（b）**：每个设备执行一步前向（forward）和后向（backward）计算，得到局部梯度 $ G_i $。

3. **梯度聚合（b）**：对各个设备上的局部梯度 $ G_i $ 执行 All-Reduce 操作，得到完整梯度 $ G $。这一步的单个设备的通信量为 $ 2\Phi $。

4. **权重更新（c）**：使用完整梯度 $ G $ 和优化器状态更新权重 $ W $。每个设备保存一部分权重 $ W $，并通过 All-Gather 操作从其他设备获取更新后的部分权重，完成权重更新。此时的单个设备通信量为 $ \Phi $。

在 $ P_{os} $ 阶段，将 Adam 优化器状态根据数据并行维度 $ N_d $ 分成等份。每个设备只需存储和更新总优化器状态的 $ 1/N_d $，并更新对应参数。通过分片和聚合操作，显存占用从 $ 4\Psi + K\Psi $ 降低到 $ 4\Psi + K\Psi / N_d $。当 $ N_d $ 很大时，显存占用接近于 $ 4\Psi $，带来约 4 倍显存节约。

#### ZeRO-2 计算内存分析

ZeRO-2 在 ZeRO-1 的基础上进一步优化，通过对梯度（Grad）也进行切分，减少显存占用并提高通信效率。

![数据并行](images/02DataParallel11.png)
:width:`650px`

1. **数据分片**：从优化器状态和梯度分片开始，将优化器状态和梯度分成 N 份，每个设备保存一份分片，并将训练批次数据（batch data）分成 N 份，每个设备处理一份数据。

2. **前向与后向计算**：每个设备执行一步前向（forward）和后向（backward）计算，得到局部梯度 $ G_i $。

3. **梯度分片与聚合**：对各块设备上的局部梯度 $ G_i $ 执行 Reduce-Scatter 操作，确保每个设备只维护自己负责的梯度部分。比如，设备1负责维护梯度 $ G_1 $，其他设备只需要将 $ G_1 $ 对应位置的梯度发送给设备1。聚合完毕后，设备1释放无用的显存部分，单卡通信量为 $ \Phi $。

4. **权重更新**：每个设备使用自身维护的梯度 $ G_i $ 和优化器状态 $ O_i $ 更新相应的权重 $ W_i $。

5. **权重聚合**：对权重 $ W_i $ 执行 All-Gather 操作，将其他设备的权重 $ W_i $ 同步至完整权重 $ W $，单卡通信量为 $ \Phi $。

在 $ P_{os+g} $ 阶段，梯度与优化器强相关，因此优化器可以更新其独立的梯度参数。更新梯度参数时使用 Reduce-Scatter 操作，梯度参数更新后立即释放。具体实现中使用分桶（Bucket）技术，将梯度分到不同的桶中并在桶上进行 Reduce-Scatter 操作。通过移除梯度和优化器状态冗余，显存占用从 $ 4\Psi + K\Psi $ 降低到 $ 2\Psi + K\Psi / N_d $。当 $ N_d $ 很大时，显存占用接近于 $ 2\Psi $，带来约 8 倍显存节约。

#### ZeRO-3 计算内存分析

ZeRO-3 在 ZeRO-1 和 ZeRO-2 的基础上进一步优化，通过对优化器状态、梯度和权重进行全面切分，最大化显存节约，这种优化使得训练超大规模模型成为可能。

![数据并行](images/02DataParallel12.png)
:width:`650px`

1. **数据分片**：对优化器状态、梯度和权重进行全面切分，每个设备保存一份分片，将训练批次数据（batch data）分成 N 份，每块设备处理一份数据。

2. **前向计算**：在前向计算过程中，对权重 $ W $ 执行 All-Gather 操作，从各设备获取分布在不同设备上的权重，得到完整的权重 $ W $。不属于自身的权重 $ W_{others} $ 被抛弃，单卡通信量为 $ \Phi $。

3. **后向计算**：在后向计算过程中，再次对权重 $ W $ 执行 All-Gather 操作，取回完整权重，并抛弃不属于自身的部分 $ W_{others} $，单卡通信量为 $ \Phi $。

4. **梯度聚合**：后向计算得到各自的梯度 $ G_i $ 后，对梯度 $ G_i $ 执行 Reduce-Scatter 操作，从其他设备聚合自身维护的梯度 $ G_i $。聚合操作结束后，立刻抛弃不是自己维护的梯度，单卡通信量为 $ \Phi $。

5. **权重更新**：每块设备只保存其权重参数 $ W_i $，由于只维护部分参数 $ W_i $，因此无需对 $ W_i $ 执行 All-Reduce 操作。

在 $ P_{os+g+p} $ 阶段，优化器状态、梯度和权重均进行划分。在前向和后向计算过程中，通过广播从其他设备中获取参数，减少每块设备中的显存占用。通过移除梯度、优化器状态和权重的冗余，将显存占用从 $ 4\Psi + K\Psi $ 降低到 $ (4\Psi + K\Psi)/ N_d $。这种方法通过增加通信开销，以通信换显存，使得显存占用与 $ N_d $ 成正比。显存占用的优化带来了1.5倍单卡通信量的增加。

### ZeRO-R

除了优化模型状态（优化器状态、梯度和参数）的内存利用率，ZeRO 还专门针对剩余状态（如激活数据、临时缓冲区和内存碎片等）进行了优化，以进一步减少内存开销。ZeRO-R 对剩余状态进行了切分和优化，主要包括以下几个策略:

1) **分区激活检查点**（Partitioned Activation Checkpointing，$P_{a}$）：解决了模型并行时激活内存冗余的问题。在模型并行中，每个设备需要保存完整的输入激活数据才能计算自己分到的模型部分。ZeRO-R 将激活检查点按模型并行度 $N_m$ 进行分区，每个设备只需存储 $\frac{1}{N_m}$ 的激活检查点。在需要时通过 All-Gather 操作重构出完整激活数据，从而按 $N_m$ 的比例减少激活内存。 在极端情况下，当模型规模很大时，ZeRO-R 甚至可以将分区后的激活检查点卸载到 CPU 内存（$P_{a+cpu}$），再次降低设备内存占用，代价是额外的主机-设备通信开销。该策略在大规模模型训练时会自动开启，以保证足够的设备内存用于计算。

2) **恒定大小的缓冲区**（Constant Size Buffer，$C_{b}$）：一些操作如 All-reduce 需要将张量拼成连续的临时缓冲区，使用恒定大小的缓冲区来避免临时缓冲区随着模型大小的增加而爆炸，同时使它们足够大以保持效率。

3) **内存碎片化整理**（Memory Defragmentation，$M_{d}$）：在训练过程中，由于激活检查点、梯度等张量生命周期的差异，会产生大量内存碎片。ZeRO-R 通过预分配和动态管理这些张量的内存，减少了内存碎片和内存分配器的开销，提高了内存利用率。

通过以上优化策略，ZeRO-R 很好地补充和完善了 ZeRO-DP 优化模型状态内存的功能。两者相结合，ZeRO 优化器能最大限度减少大规模模型训练的内存占用，为未来万亿参数级别的深度学习模型铺平了道路。

### ZeRO-Infinity

ZeRO-Infinity 是 ZeRO 的扩展，可以将深度学习训练扩展到前所未有的规模。具体来说它突破了 GPU 内存壁垒的限制，并使得能够训练具有数万亿个参数的模型成为可能，这是迄今为止最先进系统所无法企及的量级。此外，它为训练具有一千万亿个参数的模型铺平了道路——充分利用系统的全部内存容量，利用 GPU、CPU 和 Non-Volatile Memory Express（NVMe）等所有异构内存组件的能力。

![ZeRO-Infinity](images/02DataParallel13.png)
:width:`650px`

在 ZeRO-Infinity 中，参数从较慢的内存源（如 CPU 和 NVMe）无缝迁移到 GPU，其中它们被合并为完整的层。梯度计算完成后，这些参数被聚合、重新分区，然后重新卸载回较慢的内存组件。其中内存资源的编排确保了最佳利用和最小的开销。这种创新的方法不仅克服了 GPU 内存的常规限制，而且提升了分布式框架的可扩展性。

### ZeRO 通讯分析

无论是零冗余优化，还是卸载到 CPU 和 NVMe 内存，一个关键问题是，它们有限的带宽是否会影响训练效率。我们很自然地会问是否在用通信量来换取内存效率。换句话说，与标准 DP 方法相比，ZeRO 驱动的 DP 方法的通讯量是多少？

#### ZeRO-DP

最先进的 All-Reduce 实现采用两步法，第一步是 Reduce-Scatte 操作，一个是 All-Gather 操作，每个流程的总数据移动量为 $\Psi$ 个元素（对于 $\Psi$ 个元素的数据）。因此，标准 DP 在每个训练步骤中会产生 2 $\Psi$ 次数据移动。

通过梯度分区（$P_{os+g}$），每个进程只存储更新相应参数分区所需的梯度部分。因此，ZeRO 只需要对梯度先进行 Reduce-Scatte 操作，产生的通信量为 $\Psi$。在每个进程更新完自己负责的参数分区后，会执行一次 All-Gather，从所有数据并行进程中收集所有更新的参数。这也会产生 $\Psi$ 的通信量。因此，每个训练步骤的总通信量为 $\Psi$ + $\Psi$ = 2 $\Psi$，与标准 DP 相同。

在参数分区（$P_{os+g+p}$）后，每个数据并行进程只存储其更新的参数。因此，在前向传播过程中，它需要接收所有其他分区的参数。不过，这可以通过流水线操作来避免内存开销——在对模型中与特定分区对应的部分进行前向传播计算之前，负责该分区的数据并行进程可以向所有数据并行进程广播权重。一旦该分区的前向传播计算完成，参数就可以被丢弃。因此，总通信量为 $\Psi × N_d / N_d = \Psi$。我们通过在整个前向传播中通过 All-gather 传播参数已重新获取参数，并在使用完参数后将其丢弃。而在后向传播时，需要以相反的顺序再次进行参数获取。参数的通信为 2 $\Psi$，在参数更新时只需要执行一个 Reduce-Scatte 操作，通信量为 $\Psi$，因此总通信量是 3 $\Psi$，是标准 DP 的 1.5 倍。

#### ZeRO-R

ZeRO-R 的通信开销取决于模型大小、检查点策略和模型并行（MP）策略。与标准模型并行相比（其中没有对激活进行分区），ZeRO-R $P_{a}$ 的通信开销通常不到标准模型并行的十分之一。

在使用激活检查点的 Megatron-LM 中，每个 Transformer 块在前向传播中执行两次大小为 $batch × seq × length × hidden\_dim$ 的 All-Reduce 操作，然后在反向传播中再执行两次。在使用激活检查点的 ZeRO-R 中，每个前向重计算激活之前需要执行一个额外的 All-Gather 操作。通常情况下，对于每个 Transformer 块的输入激活进行检查点，因此每个 Transformer 块需要一个 All-Gather 操作。因此，ZeRO-R $P_{a}$ 的通信开销为 $seq\_length × hidden\_dim$，仅增加不到 10%。

当 MP 与 DP 一起使用时， ZeRO-R $P_{a}$ 可以将数据并行通信量减少一个数量级，而模型并行通信量只增加 10%，并且当数据并行通信是性能瓶颈时，可以显着提高效率。通过模型并行可以减少数据并行的内存消耗，从而可以成比例地增加批处理大小。对于大型模型，MP 可以增加到 16（DGX-2 节点上的 GPU 数量），从而可以将批处理大小增加多达 16 倍。数据并行训练的通信量与批处理大小成反比，由于 $P_{a}$ 导致批处理大小增加一个数量级，可能会导致数据并行通信量减少一个数量级。

如果应用 $P_{a+cpu}$，则分区激活检查点会被卸载到 CPU，将激活内存需求减少到接近零，但与 $P_{a}$ 相比，往返 CPU 内存的数据移动增加了 2 倍。如果 DP 通信量是主要瓶颈，由于批处理大小较小，$P_{a+cpu}$ 也可以通过增加批处理大小来提高效率，只要 CPU 数据传输开销小于 DP 通信量开销。

#### ZeRO-Infinity

我们可以使用**峰值计算吞吐量（$peak_{tp}$）**、**数据移动带宽（$𝑏𝑤$）** 及其**算术强度（$𝑎𝑖𝑡$）** 来估算 ZeRO-Infinity 
的训练效率，因为它还涉及到了设备之间的数据移动。工作负载的**算术强度（AIT）** 是总计算量与计算所需数据量之间的比率。它描述了每次数据移动所需的计算量。AIT 越高，意味着对数据移动带宽的要求越低，因为每加载一个数据，加速器就能完成更多计算。

$$
𝑎𝑖𝑡 = \frac{𝑡𝑜𝑡𝑎𝑙\_𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛}{𝑡𝑜𝑡𝑎𝑙\_𝑑𝑎𝑡𝑎\_𝑚𝑜𝑣𝑒𝑚𝑒𝑛𝑡}
$$

因此我们的效率可以大致估算为：

$$
\begin{aligned}

compute\_time 
&= \frac{𝑡𝑜𝑡𝑎𝑙\_𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛}{𝑝𝑒𝑎𝑘_{tp}} \\

𝑐𝑜𝑚𝑚𝑢𝑛𝑖𝑐𝑎𝑡𝑖𝑜𝑛\_𝑡𝑖𝑚𝑒 
&= \frac{𝑡𝑜𝑡𝑎𝑙\_𝑑𝑎𝑡𝑎\_𝑚𝑜𝑣𝑒𝑚𝑒𝑛𝑡}{bw} \\
&= \frac{𝑡𝑜𝑡𝑎𝑙\_𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛}{ait × bw} \\

𝑒𝑓𝑓𝑖𝑐𝑖𝑒𝑛𝑐𝑦 
&= \frac{𝑐𝑜𝑚𝑝𝑢𝑡𝑒\_𝑡𝑖𝑚𝑒}{𝑐𝑜𝑚𝑝𝑢𝑡𝑒\_𝑡𝑖𝑚𝑒+𝑐𝑜𝑚𝑚𝑢𝑛𝑖𝑐𝑎𝑡𝑖𝑜𝑛\_𝑡𝑖𝑚𝑒} \\
&= \frac{𝑎𝑖𝑡 × 𝑏𝑤}{𝑎𝑖𝑡 × 𝑏𝑤 + 𝑝𝑒𝑎𝑘_{tp}}

\end{aligned}
$$

我们同样以 Transformer 为例：每次迭代的总计算量可以由参数数量、序列长度和批量大小估算，即对于前向传播为 $2 × bsz × seq × 𝑝𝑎𝑟𝑎𝑚𝑠$，反向传播的成本大约是正向传播的两倍。因此我们可以估算计算量：

$$
𝑐𝑜𝑚𝑝𝑢𝑡𝑎𝑡𝑖𝑜𝑛\_𝑝𝑒𝑟\_𝑖𝑡𝑒𝑟 = 2 × 4 × 𝑏𝑠𝑧 × 𝑠𝑒𝑞 × 𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠 = 2 × 4 × 12 × 𝑏𝑠𝑧 × 𝑠𝑒𝑞 × 𝑛𝑙 × ℎ𝑑^2
$$

在前向和反向传播期间，模型参数必须从源位置加载到 GPU 寄存器至少两次（前向传播期间和实际后向传播期间），导致 2 次的数据移动。在存在激活检查点的情况下，可以在向后传递过程中额外加载一次参数以进行重新计算。此外，梯度必须至少从 GPU 寄存器存储到其最终位置一次。因此，假设参数和梯度存储在相同的最终位置，则前向和后向传递期间的总数据移动将为 $4 × 𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠$，即 $2 × 4 × 𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠$（以字节为单位）。因此参数和梯度的 ait 为：

$$
𝑠𝑒𝑞 × 𝑏𝑠𝑧
$$

在优化器迭代期间，必须至少读取一次优化器状态，​​并且必须至少写入一次优化器状态。因此，总数据移动量为 $2 × 𝑜𝑝𝑡𝑖𝑚𝑖𝑧𝑒𝑟_𝑠𝑡𝑎𝑡𝑒𝑠$，大约为 $2 × 16 × 𝑝𝑎𝑟𝑎𝑚𝑒𝑡𝑒𝑟𝑠$ 字节。因此，在完整的训练迭代期间，优化器状态的 ait 为：

$$
𝑠𝑒𝑞 × 𝑏𝑠𝑧/4
$$

在前向传播期间，激活检查点必须保存到其最终位置，并且必须在后向传播期间获取。因此，激活检查点的总数据移动量（以字节为单位）为 $2 × 𝑡𝑜𝑡𝑎𝑙\_𝑎𝑐𝑡𝑖𝑣𝑎𝑡𝑖𝑜𝑛\_𝑐ℎ𝑒𝑐𝑘𝑝𝑜𝑖𝑛𝑡𝑠\_𝑖𝑛\_𝑏𝑦𝑡𝑒𝑠$，带入我们之间计算的激活检查点大小，可以得到总数据移动量 $4 × 𝑛𝑙/𝑐𝑖 × ℎ𝑑 × 𝑠𝑒𝑞 × 𝑏𝑠𝑧$。所以激活检查点的 ait 为：

$$
24 × ℎ𝑑 × 𝑐𝑖
$$

模型状态和激活检查点对带宽的要求大不相同。前者只取决于**批量大小和序列长度**，而后者只取决于**激活检查点的频率和模型的隐藏维度大小**。在实际中，参数和梯度的带宽超过 70 GB/s，即使是最小的批处理量，也能实现超过 50% 的效率。在这种带宽下，数据移动理论上可以与计算完全重叠，从而实现 100% 的效率。与参数和梯度相比，优化器状态需要高出近 4 倍的带宽才能达到 50% 的效率。此外，优化器状态在前向和后向传播结束时更新，不能与计算重叠。因此，它们需要更大的带宽来保持整个 DL 工作负载的效率。例如，在每个 GPU 的批处理量为 2 的情况下，要达到 90% 的效率，需要近 1.5 TB/s 的有效带宽，甚至超过了 GPU 内存带宽。启用激活检查点后，即使隐藏大小为 2K，2 GB/s 的微薄带宽也能维持 50% 以上的效率。 当隐藏大小超过 8K 时，带宽需求降至 1 GB/s 以下。

### 完全分片的数据并行简单实现

和 DDP 一样，我们可以通过简单的嵌套使用 ZeRO 优化器来实现完全分片的数据并行。将模型的参数、梯度和优化器状态分片，从而显著减少单个 GPU 的内存占用，实现更大模型的训练。以下是一个简单的代码示例，展示了如何使用 `torch.distributed.fsdp` 中的 `FullyShardedDataParallel`（FSDP）类进行完全分片的数据并行：

```python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

torch.cuda.set_device(device_id)
sharded_module = FSDP(my_module)

optim = torch.optim.Adam(sharded_module.parameters(), lr=0.0001)
x = sharded_module(x, y=3, z=torch.Tensor([1]))

loss = x.sum()
loss.backward()
optim.step()
```

我们首先设置当前的 GPU 设备，然后将 `my_module` 包装成一个 FSDP 模块。这会将模型的参数、梯度和优化器状态在多个 GPU 之间进行分片，从而减少每个 GPU 上的内存占用。

再来看一个更详细的例子，我们使用 `fsdp_main` 函数，用于分布式训练 T5 模型。函数通过 `setup_model` 函数加载模型和分词器，并设置分布式训练的相关环境变量，包括 `local_rank`、`rank` 和 `world_size`。在数据集和数据加载器设置之后，通过 `functools.partial` 函数部分应用了 `transformer_auto_wrap_policy`，并指定 `T5Block` 为要自动包装的变压器层。这一步的目的是定义一个自动包装策略，用于后续的模型分片和并行处理。

接下来，定义了 `sharding_strategy` 变量，并将其设为 `ShardingStrategy.SHARD_GRAD_OP`。这表示使用 Zero2 分片策略，如果要使用 Zero3 策略，可以将其设为 `FULL_SHARD`。分片策略决定了在分布式训练中如何管理和分配模型参数，以优化内存使用和计算性能。
为了支持混精度训练（bf16），如果当前 CUDA 版本和 NCCL 版本支持 bf16，并且 CUDA 版本大于或等于 11.0，那么 `bf16_ready` 变量为 `True`，并将 `mp_policy` 设为 `bfSixteen`。否则，`mp_policy` 设为 `None`，默认使用 fp32（单精度）训练。

在模型仍在 CPU 上时，代码将模型传入 `FSDP`（Fully Sharded Data Parallel）模块，配置自动包装策略、混合精度策略（`mixed_precision`），以及当前设备 ID（`device_id`）。这一步是将模型转换为分布式训练的模式，以便在多个 GPU 之间分片和并行计算。

```python
def fsdp_main(args):
    model, tokenizer = setup_model("t5-base")

    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Set dataset and dataloader here
    
    t5_auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={
            T5Block,
        },
    )
    sharding_strategy: ShardingStrategy = ShardingStrategy.SHARD_GRAD_OP #for Zero2 and FULL_SHARD for Zero3
    torch.cuda.set_device(local_rank)

    bf16_ready = (
    torch.version.cuda
    and torch.cuda.is_bf16_supported()
    and LooseVersion(torch.version.cuda) >= "11.0"
    and dist.is_nccl_available()
    and nccl.version() >= (2, 10)
    )

    if bf16_ready:
        mp_policy = bfSixteen
    else:
        mp_policy = None # defaults to fp32

    # model is on CPU before input to FSDP
    model = FSDP(model,
        auto_wrap_policy=t5_auto_wrap_policy,
        mixed_precision=mp_policy,
        #sharding_strategy=sharding_strategy,
        device_id=torch.cuda.current_device())

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1)
        scheduler.step()
```


当然，如果需要更简单而全面的配置，我们可以通过 `deepspeed` 库进行便捷的 ZeRO 配置。

## 异步的数据并行

前面的介绍都是基于**同步的数据并行**的，同步的数据并行特别适用于计算资源相对均衡的情况。在同步数据并行中，每个设备都处理数据的一个子集，并独立地计算梯度。在每次迭代中，所有设备都将它们的梯度汇总，并通过一致的规则来更新模型参数。这样，所有设备上的模型都保持一致，不会出现不同步的情况。由于所有设备在每个训练步骤中都执行相同的更新操作，模型的收敛性更容易得到保证。且所有设备都参与到梯度更新的计算中，整体计算效率也相对较高。此外，同步数据并行还易于实现，因为所有设备的操作都是同步的，不需要复杂的同步机制。

![数据并行](images\02DataParallel14.png)
:width:`650px`

但是同步数据并行也有一些局限性。当集群中的某些设备性能较差或者出现故障时，整体的训练效率会受到影响，所有设备都需要等待最慢的设备完成计算。又或是当设备数量过多时，集合通信的时间可能会成为训练的瓶颈，从而限制整体的扩展性。

**异步的数据并行（Asynchronous Data Parallelism, ADP）** 可以在一定程度上解决这些问题。在异步数据并行中，不同设备的计算过程相互独立，不再需要等待其他设备完成计算。每个设备都按照自己的速度进行前向和反向传播，随时将计算得到的梯度更新到模型参数中。这样，快速的设备不再受到慢速设备的影响，整体计算效率得到提高。异步数据并行的步骤为：

![数据并行](images\02DataParallel15.png)
:width:`650px`

- **前向传播**：将 mini-batch 数据平均分配到每个设备上。接下来进行分布式初始化，将模型和优化器复制到每个设备上，保证各设备的模型、优化器完全相同。初始化完成后，各设备根据分配到的数据和模型同时进行前向传播。

- **损失计算与反向传播**：前向传播完成后，每个设备分别计算模型损失并进行反向传播。得到梯度后，各设备将自己的梯度发送到主进程。主进程接收到多个梯度后，进行梯度累加和平均，根据累加后的梯度更新模型参数。由于是异步更新，不需要等待所有设备都完成计算后再进行更新，这样快速的设备可以频繁更新参数，而慢速的设备则不影响整体更新进度。

- **参数更新**：主进程将更新后的模型参数广播给各个设备。每个设备接收到更新后的参数后，进行下一轮的训练。

- **重复**：上述步骤重复进行，直到模型收敛或者达到预定的训练轮数。

异步数据并行的优点之一是它可以充分利用集群中每个设备的计算能力，快速的设备不会受到慢速设备的影响，从而提高了整体的训练速度。此外，由于每个设备都独立地进行计算和参数更新，异步数据并行也具有较好的扩展性，能够适应不同规模的集群和不同数量、类型的设备。但是异步数据并行也存在一些挑战。由于计算过程是异步的，可能会出现梯度更新之间的竞争条件，需要采取一些机制来解决，如：**参数服务器**。同时由于计算过程不再同步，模型的收敛性可能会受到影响，需要通过调整学习率或者采用一些优化算法来弥补。

## 弹性数据并行

弹性训练是一种分布式机器学习训练方法，旨在提高系统在动态环境中的容错性和灵活性。其核心理念是通过动态调整训练过程中的资源分配和任务调度，以应对节点故障、资源变化等不可预测的情况，从而保证训练过程的连续性和高效性。弹性训练主要通过以下方法实现其目标：

1. **动态调度**：系统根据当前资源状况动态分配任务，并在资源变化时进行重新调度。这种方法能够有效利用可用资源，提高训练效率。
2. **检查点机制**：在训练过程中定期保存模型状态和训练进度，以便在故障发生时能够从最近的检查点继续训练，减少因故障导致的训练时间损失。
3. **故障检测和恢复**：系统持续监控各个节点的状态，及时检测故障并采取相应的恢复措施，如重新启动失败的任务或重新分配资源。这种机制保证了训练过程的鲁棒性。

PyTorch 提供了 Torchelastic 组件，用于支持分布式训练过程中的弹性调度和故障恢复。它使得在大规模分布式训练环境中，可以动态调整参与训练的节点数量，并在节点发生故障时进行自动恢复，从而提高训练过程的鲁棒性和效率。Torchelastic 包括 Elastic Agent 服务器、Rendezvous 等组件。

### Elastic Agent 服务器

Elastic Agent 是 Torch Elastic 的控制面板。它是一个进程，负责启动和管理底层的工作进程。其主要职责包括：

1. **与分布式 Torch 的集成**：启动工作进程，并提供所有必要信息，使其能够成功且轻松地调用 `torch.distributed.init_process_group()` 进行分布式环境管理。
2. **故障恢复**：监控工作进程，一旦检测到工作进程失败或不健康，立即终止所有工作进程并重新启动。
3. **弹性调度**：响应成员变化，重新启动包含新成员的工作进程。

最简单的 Agent 部署在每个节点上，管理本地进程。更高级的 Agent 可以远程启动和管理工作进程。Agent 可以完全去中心化，根据其管理的工作进程独立做出决策，也可以通过与其他管理同一作业的 Agent 协作，做出集体决策。

![数据并行](images\02DataParallel16.png)
:width:`650px`

以上是一个管理本地工作进程组的 Agent 的示意图。每个 Agent 都会管理多个 Worker，并运行一个 Rendezvous 模块，用于在分布式环境中实现节点的同步和发现，阻塞会持续到至少 Min 个 Elastic Agent 加入后返回。当有新的节点加入或现有节点退出时（即成员变更），Rendezvous 过程会重新开始。Rendezvous 过程包括两个关键步骤：屏障操作（barrier）和排名分配（rank assignment）。屏障操作确保所有节点在达到最小节点数量之前都处于等待状态，并在达到最大节点数量后立即完成。排名分配则为每个节点分配一个唯一的 rank，确保每个节点在分布式训练中的 rank 是明确的。

Elastic Agent 持续监控本地的工作进程状态。如果检测到任何工作进程失败或不健康，Elastic Agent 会立即终止所有工作进程，并重新启动。这一过程通过重新启动（respawn）工作进程来实现，确保训练任务的持续进行。监控工作进程的健康状态是一个持续的过程，Elastic Agent 不断检查工作进程的状态，并在必要时触发故障恢复机制。

Elastic Agent 还具备弹性调度功能。它能够动态调整参与训练的节点数量，响应成员变更，重新启动包含新成员的工作进程。这样，即使在训练过程中有节点故障或新增节点，Elastic Agent 也能够及时调整，保证训练过程的鲁棒性和灵活性。

### Rendezvous

在 Torch Distributed Elastic 中，Rendezvous 是一种功能，结合了分布式同步原语和节点发现。它用于在分布式训练作业中聚集节点，确保所有节点对节点列表及其角色达成一致，并一致决定何时开始或恢复训练。Rendezvous 的关键功能有：

1. **Barrier**：执行 Rendezvous 的节点将阻塞，直到至少达到最小节点数量为止。这意味着 Barrier 的大小不固定。达到最小节点数量后，会有一个额外的等待时间，以防止过快完成，可能排除同时尝试加入的其他节点。如果达到最大节点数量，Rendezvous 会立即完成。若在指定时间内未达到最小节点数量，Rendezvous 将失败，以释放部分分配的作业资源。

2. **排他性**：确保在任何时候只有一个节点组存在（对于同一作业）。新的节点若试图晚加入，只能宣布等待，直到现有的 Rendezvous 被销毁。

3. **一致性**：Rendezvous 完成后，所有成员将同意作业成员和各自的角色。角色由一个整数表示，称为 rank，范围在 0 到 world size 之间。需要注意的是，rank 不是固定的，同一节点在下一次（重新）Rendezvous 中可以分配不同的 rank。

4. **故障恢复**：在 Rendezvous 过程中容忍节点故障。如果在加入 Rendezvous 和其完成之间发生进程崩溃（或失去网络连接等），将自动进行重新 Rendezvous。若节点在完成 Rendezvous 后故障，将由 Torch Distributed Elastic 的 train_loop 处理，并触发重新 Rendezvous。

5. **共享键值存储**：Rendezvous 完成后，将创建并返回一个共享键值存储，实现 `torch.distributed.Store` API。此存储仅在已完成的 Rendezvous 成员间共享，用于初始化作业控制和数据平面所需的信息交换。

Torch Distributed Elastic 提供了 `DynamicRendezvousHandler` 类，实现上述 Rendezvous 机制。它是一种与后端无关的类型，需在构造时指定特定的 `RendezvousBackend` 实例。Torch 分布式用户可以实现自己的后端类型，或使用 PyTorch 提供的实现：

1. **C10dRendezvousBackend**：使用 C10d 存储（默认 TCPStore）作为 Rendezvous 后端。其主要优势是不需要第三方依赖（如 etcd）来建立 Rendezvous。
2. **EtcdRendezvousBackend**：取代了旧的 `EtcdRendezvousHandler` 类。将 `EtcdRendezvousBackend` 实例传递给 `DynamicRendezvousHandler` 功能等同于实例化 `EtcdRendezvousHandler`。

以下是使用 `DynamicRendezvousHandler` 的示例代码：

```python
store = TCPStore("localhost")

backend = C10dRendezvousBackend(store, "my_run_id")

rdzv_handler = DynamicRendezvousHandler.from_backend(
    run_id="my_run_id",
    store=store,
    backend=backend,
    min_nodes=2,
    max_nodes=4
)
```

下面是描述 Rendezvous 工作流程的状态图。

![数据并行](images\02DataParallel17.png)
:width:`650px`

1. **Version Counter**：在流程开始时，Rendezvous 机制会创建一个版本号（如果不存在则创建初始值为“0”）。这个版本号由 /rdzv/version_counter 跟踪，并使用原子操作 fetch-add(1) 来确保版本号的唯一性和一致性。当新的节点加入或现有节点重新启动时，版本号会递增，从而标识新的 Rendezvous 过程。

2. **Active Version**：初始状态为非存在状态（non-existent），表示当前没有活跃的 Rendezvous。当版本号更新后，状态切换为活跃版本（active version），并记录在 /rdzv/active_version。这个活跃版本标识了当前正在进行的 Rendezvous 实例。

3. **Setup**：当达到最小工作节点数（min_workers）时，Rendezvous 进入设置（setup）阶段，并切换到临时状态（ephemeral）。这个阶段确保有足够的节点参与训练，使得训练过程能够顺利进行。

4. **Join Phase**：在加入阶段（join phase），节点开始加入当前的 Rendezvous。此时，Rendezvous 处于 joinable 状态，允许新的节点加入。每个节点加入后，参与者列表（participants）会更新，记录当前已加入的节点。例如，当第一个节点加入时，参与者列表为 [0]；当第二个节点加入时，列表更新为 [0, 1]；以此类推。这个阶段会持续到达到最大工作节点数（max_workers）或最后调用超时（last-call timeout）。

5. **Confirm Phase**：当达到最大工作节点数或最后调用超时时，Rendezvous 进入确认阶段（confirm phase），状态变为 frozen。在这个阶段，参与者列表和保活键（keep-alives）被记录，以确保所有已加入节点的状态一致。每次更新后，参与者和保活键列表会逐步更新，直到所有节点都达到 frozen 状态。

6. **Final State**：在最终状态（final state），Rendezvous 被认为是有效的。所有参与者和保活键被记录在案，确保训练过程的一致性和稳定性。如果有晚加入的节点，它们会增加 num_pending，表示正在等待下一次机会加入训练。这些晚加入的节点不会立即参与当前的训练，而是等待现有的 Rendezvous 完成。

### 弹性数据并行简单实现

弹性数据并行和数据并行的启动方式一致，但有以下区别：

1. 无需手动传递 `RANK`、`WORLD_SIZE`、`MASTER_ADDR` 和 `MASTER_PORT`。

2. 确保在脚本中包含 `load_checkpoint(path)` 和 `save_checkpoint(path)` 逻辑。当任何数量的工作进程失败时，我们会使用相同的程序参数重新启动所有工作进程，因此您会丢失最近一次检查点之前的所有进度（见 elastic launch）。

以下是一个训练脚本的示例，它在每个 epoch 上进行检查点操作，因此在失败时最坏情况下会丢失一个完整 epoch 的训练进度。

```python
def main():
    args = parse_args(sys.argv[1:])
    state = load_checkpoint(args.checkpoint_path)
    initialize(state)

    torch.distributed.init_process_group(backend=args.backend)

    for i in range(state.epoch, state.total_num_epochs):
        for batch in iter(state.dataset):
            train(batch, state.model)

        state.epoch += 1
        save_checkpoint(state)
```

我们可以通过 torchrun 启动分布式和弹性训练，在启动弹性作业时，需要在至少 MIN_SIZE 节点和最多 MAX_SIZE 节点上运行以下命令。

```bash
torchrun --nnodes=MIN_SIZE:MAX_SIZE \
    --nproc-per-node=TRAINERS_PER_NODE \
    --max-restarts=NUM_ALLOWED_FAILURES_OR_MEMBERSHIP_CHANGES \
    --rdzv-id=JOB_ID \
    --rdzv-backend=c10d \
    --rdzv-endpoint=HOST_NODE_ADDR \
    YOUR_TRAINING_SCRIPT.py (--arg1 ... train script args...)
```

## 本节视频

<html>
<iframe src="https:&as_wide=1&high_quality=1&danmaku=0&t=30&autoplay=0" width="100%" height="500" scrolling="no" border="0" frameborder="no" framespacing="0" allowfullscreen="true"> </iframe>
</html>
