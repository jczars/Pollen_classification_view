import logging
import os
import psutil
import nvidia_smi
import gc
import tensorflow as tf
from tensorflow.keras.backend import clear_session

def get_cpu_memory_usage():
    """Retrieves CPU memory usage information."""
    memory_info = psutil.virtual_memory()
    return memory_info.used / (1024 ** 3), memory_info.available / (1024 ** 3)  # Returns in GB

def get_gpu_memory_usage():
    """Retrieves GPU memory usage information using nvidia-smi."""
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)  # Assuming using GPU 0
    info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
    used_memory = info.used / (1024 ** 2)  # Convert to MB
    free_memory = info.free / (1024 ** 2)  # Convert to MB
    total_memory = info.total / (1024 ** 2)  # Convert to MB
    return used_memory, free_memory, total_memory

def reset_keras():
    """Resets Keras backend and attempts to free GPU memory."""
    clear_session()
    print("Cleared Keras session.")
    
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            for device in physical_devices:
                tf.config.experimental.reset_memory_stats(device.name)
                print(f"Memory stats reset for device: {device.name}")
        except Exception as e:
            print(f"Error resetting memory stats: {e}")

    gc.collect()  # Trigger garbage collection
    print("Garbage collection complete.")

def monitor_memory_and_run():
    """Monitors GPU memory usage and initiates execution if memory conditions are met."""
    used_gpu_mem, free_gpu_mem, total_gpu_mem = get_gpu_memory_usage()
    used_cpu_mem, free_cpu_mem = get_cpu_memory_usage()

    print(f"CPU Memory - Used: {used_cpu_mem:.2f} GB, Available: {free_cpu_mem:.2f} GB")
    print(f"GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB, Total: {total_gpu_mem:.2f} MB")

    # Check if GPU memory is at least 60% free
    if free_gpu_mem / total_gpu_mem * 100 < 60:
        print("GPU memory usage is above the threshold. Attempting to clear memory.")
        reset_keras()

        # Recheck GPU memory usage after cleanup
        used_gpu_mem, free_gpu_mem, total_gpu_mem = get_gpu_memory_usage()
        print(f"Post-Cleanup GPU Memory - Used: {used_gpu_mem:.2f} MB, Free: {free_gpu_mem:.2f} MB")

        if free_gpu_mem / total_gpu_mem * 100 < 60:
            print("Insufficient GPU memory available. Exiting.")
            return

    print("Sufficient GPU memory available. Proceeding with program execution.")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', filename='memory_log.log', filemode='w')

def log_memory_usage(variable_name):
    """
    Log the memory usage before and after removing a variable.
    
    Args:
        variable_name (str): The name of the variable being deleted.
    """
    # Get memory usage before
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
    
    logging.info(f"[INFO] Memory usage before deleting {variable_name}: {mem_before:.2f} MB")
    
    # Garbage collection and memory cleanup
    gc.collect()
    
    # Get memory usage after
    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    logging.info(f"[INFO] Memory usage after deleting {variable_name}: {mem_after:.2f} MB")


if __name__ == "__main__":
    monitor_memory_and_run()
