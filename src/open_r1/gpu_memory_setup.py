import math

import torch


def setup_gpu_memory(model_size_gb=7, per_gpu_batch_size=1, num_gpus=8):
    """
    Allocate memory across available GPUs
    model_size_gb: approximate size of model in GB
    per_gpu_batch_size: batch size per GPU
    num_gpus: number of GPUs to use
    """
    # Calculate approximate memory needed per GPU
    memory_per_gpu = model_size_gb + (
        2 * model_size_gb * per_gpu_batch_size
    )  # Model + gradients + buffer

    available_gpus = []
    for i in range(torch.cuda.device_count()):
        total_memory = torch.cuda.get_device_properties(i).total_memory / (
            1024**3
        )  # Convert to GB
        if total_memory >= memory_per_gpu:
            available_gpus.append(i)

        if len(available_gpus) >= num_gpus:
            break

    if not available_gpus:
        raise RuntimeError("Not enough GPU memory available")

    # Set up device map
    device_map = {i: torch.device(f"cuda:{i}") for i in available_gpus[:num_gpus]}

    return device_map


def print_gpu_memory():
    """Print current GPU memory usage"""
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
        props = torch.cuda.get_device_properties(i)
        print(f"Total Memory: {props.total_memory / 1024**3:.2f} GB")


# Example usage
if __name__ == "__main__":
    # For DeepSeek-7B model, approximate size
    MODEL_SIZE_GB = 7
    BATCH_SIZE_PER_GPU = 1
    NUM_GPUS = 8

    try:
        # Setup GPU allocation
        device_map = setup_gpu_memory(
            model_size_gb=MODEL_SIZE_GB,
            per_gpu_batch_size=BATCH_SIZE_PER_GPU,
            num_gpus=NUM_GPUS,
        )

        print("\nInitial GPU Memory Status:")
        print_gpu_memory()

        # Simulate model allocation (for testing)
        test_tensors = []
        for gpu_id in device_map:
            # Allocate approximately 7GB per GPU
            tensor = torch.zeros(
                (1, math.ceil(7 * 1024 * 1024 * 1024 / 4)),
                dtype=torch.float32,
                device=f"cuda:{gpu_id}",
            )
            test_tensors.append(tensor)

        print("\nAfter Allocation:")
        print_gpu_memory()

        # Optional: Clear memory
        del test_tensors
        torch.cuda.empty_cache()

        print("\nAfter Clearing:")
        print_gpu_memory()

    except RuntimeError as e:
        print(f"Error: {e}")
