
# world_size = 16
# tensor_model_parallel_size = 2
# pipeline_model_parallel_size = 4
world_size = 192
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 128
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 64
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8

world_size = 160
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4

world_size = 224
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4

world_size = 96
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 4

data_parallel_size = world_size // (tensor_model_parallel_size *
                                    pipeline_model_parallel_size) # 2
num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size # 8
num_pipeline_model_parallel_groups = world_size // pipeline_model_parallel_size # 4
num_data_parallel_groups = world_size // data_parallel_size # 8

# Build the data-parallel groups.
print("Build DP Groups :")
all_data_parallel_group_ranks = []
for i in range(pipeline_model_parallel_size):
    start_rank = i * num_pipeline_model_parallel_groups
    end_rank = (i + 1) * num_pipeline_model_parallel_groups
    for j in range(tensor_model_parallel_size):
        ranks = range(start_rank + j, end_rank,
                      tensor_model_parallel_size)
        all_data_parallel_group_ranks.append(list(ranks))
print(all_data_parallel_group_ranks)

# Build the model-parallel groups.
print("Build MP Group:")
for i in range(data_parallel_size):
    ranks = [data_parallel_group_ranks[i]
             for data_parallel_group_ranks in all_data_parallel_group_ranks]
    print(list(ranks))

# Build the tensor model-parallel groups.
print("Build TP Groups:")
for i in range(num_tensor_model_parallel_groups):
    ranks = range(i * tensor_model_parallel_size,
                  (i + 1) * tensor_model_parallel_size)
    print(list(ranks))

# Build the pipeline model-parallel groups and embedding groups
print("Build PP Groups :")
for i in range(num_pipeline_model_parallel_groups):
    ranks = range(i, world_size,
                  num_pipeline_model_parallel_groups)
    print(list(ranks))
