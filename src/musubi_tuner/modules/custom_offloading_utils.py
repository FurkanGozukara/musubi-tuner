from concurrent.futures import ThreadPoolExecutor
import gc
import platform
import time
from typing import Optional
import torch
import torch.nn as nn


# Keep these functions here for portability, and private to avoid confusion with the ones in device_utils.py
def _clean_memory_on_device(device: torch.device):
    r"""
    Clean memory on the specified device, will be called from training scripts.
    """
    gc.collect()

    # device may "cuda" or "cuda:0", so we need to check the type of device
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if device.type == "xpu":
        torch.xpu.empty_cache()
    if device.type == "mps":
        torch.mps.empty_cache()


def _synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "xpu":
        torch.xpu.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# Use pinned memory for faster transfer between CPU and GPU, but it requires more memory. Keep these functions here for portability
def swap_weight_devices_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    # start_time = time.perf_counter()

    weight_swap_jobs = []

    # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules
    # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
    #     print(module_to_cpu.__class__, module_to_cuda.__class__)
    #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
    #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
    for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
        if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
            module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
            if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
            else:
                if module_to_cuda.weight.data.device.type != device.type:
                    # print(
                    #     f"Module {module_to_cuda_name} not found in CPU model or shape mismatch, so not swapping and moving to device"
                    # )
                    module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)

    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # cuda to cpu
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.record_stream(stream)
            module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

        stream.synchronize()

        # cpu to cuda
        for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
            cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
            module_to_cuda.weight.data = cuda_data_view

    stream.synchronize()
    torch.cuda.current_stream().synchronize()  # this prevents the illegal loss value

    # print(f"Swapped weights in {time.perf_counter() - start_time:.2f}s")


def swap_weight_devices_no_cuda(device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
    """
    not tested
    """
    assert layer_to_cpu.__class__ == layer_to_cuda.__class__

    weight_swap_jobs = []
    for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
            weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

    # device to cpu
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        module_to_cpu.weight.data = cuda_data_view.data.to("cpu", non_blocking=True)

    _synchronize_device(device)

    # cpu to device
    for module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view in weight_swap_jobs:
        cuda_data_view.copy_(module_to_cuda.weight.data, non_blocking=True)
        module_to_cuda.weight.data = cuda_data_view

    _synchronize_device(device)


def weighs_to_device(layer: nn.Module, device: torch.device):
    for module in layer.modules():
        if hasattr(module, "weight") and module.weight is not None:
            module.weight.data = module.weight.data.to(device, non_blocking=device.type != "cpu")


class Offloader:
    """
    common offloading class
    """

    def __init__(self, block_type: str, num_blocks: int, blocks_to_swap: int, device: torch.device, debug: bool = False):
        self.block_type = block_type
        self.num_blocks = num_blocks
        self.blocks_to_swap = blocks_to_swap
        self.device = device
        self.debug = debug

        # Detect platform for Windows-specific optimizations
        self.is_windows = platform.system() == 'Windows'
        
        # Disable threading on Windows due to higher overhead
        if self.is_windows:
            self.thread_pool = None
            # Print platform-specific optimization status
            print(f"[{self.block_type}] 🪟 WINDOWS detected: Pinned memory=OFF, Threading=OFF, Batched operations=ON")
        else:
            self.thread_pool = ThreadPoolExecutor(max_workers=1)
            print(f"[{self.block_type}] 🐧 LINUX detected: Pinned memory=ON, Threading=ON, Interleaved operations=ON")
        
        self.futures = {}
        self.cuda_available = device.type == "cuda"

        # Staging buffers for cuda offloading
        # Note: Pinned memory is disabled on Windows due to poor performance (3-5x slower than Linux)
        self.staging_buffer_a = None
        self.staging_buffer_b = None
        self.use_pinned_memory = not self.is_windows  # Only use pinned memory on non-Windows platforms

    def swap_weight_devices_cuda(self, device: torch.device, layer_to_cpu: nn.Module, layer_to_cuda: nn.Module):
        assert layer_to_cpu.__class__ == layer_to_cuda.__class__

        overall_start = time.perf_counter()
        timings = {}

        weight_swap_jobs = []

        # This is not working for all cases (e.g. SD3), so we need to find the corresponding modules. kept here for reference:
        # for module_to_cpu, module_to_cuda in zip(layer_to_cpu.modules(), layer_to_cuda.modules()):
        #     print(module_to_cpu.__class__, module_to_cuda.__class__)
        #     if hasattr(module_to_cpu, "weight") and module_to_cpu.weight is not None:
        #         weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))

        t0 = time.perf_counter()
        modules_to_cpu = {k: v for k, v in layer_to_cpu.named_modules()}
        for module_to_cuda_name, module_to_cuda in layer_to_cuda.named_modules():
            if hasattr(module_to_cuda, "weight") and module_to_cuda.weight is not None:
                module_to_cpu = modules_to_cpu.get(module_to_cuda_name, None)
                if module_to_cpu is not None and module_to_cpu.weight.shape == module_to_cuda.weight.shape:
                    weight_swap_jobs.append((module_to_cpu, module_to_cuda, module_to_cpu.weight.data, module_to_cuda.weight.data))
                else:
                    if module_to_cuda.weight.data.device.type != device.type:
                        module_to_cuda.weight.data = module_to_cuda.weight.data.to(device)
        timings['build_jobs'] = time.perf_counter() - t0

        # Removed redundant synchronization before stream creation (Windows optimization)
        # torch.cuda.current_stream().synchronize()  # Not needed - stream creation handles this
        
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            if self.staging_buffer_a is None:
                t0 = time.perf_counter()
                # Create staging buffers - use pinned memory only on Linux for better performance
                if self.use_pinned_memory:
                    # Linux: pinned memory for optimal GPU<->CPU transfer speed
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu").pin_memory(device=device)
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                else:
                    # Windows: regular CPU tensors (pinned memory is 3-5x slower on Windows)
                    self.staging_buffer_a = [
                        torch.empty_like(cuda_data_view, device="cpu")
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                    self.staging_buffer_b = [
                        torch.empty_like(cuda_data_view, device="cpu")
                        for _, _, cuda_data_view, _ in weight_swap_jobs
                    ]
                timings['create_buffers'] = time.perf_counter() - t0

            events = [torch.cuda.Event() for _ in weight_swap_jobs]  # Waiting events for staging buffer A to CPU non-blocking copy

            # Copy weights to staging buffers and record events
            t0 = time.perf_counter()
            gpu_to_staging_time = 0
            cpu_to_staging_time = 0
            for event, sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                events, self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
            ):
                # CUDA to staging buffer A, non-blocking copy
                t1 = time.perf_counter()
                sbuf_a.copy_(cuda_data_view.data, non_blocking=True)
                event.record(stream)
                gpu_to_staging_time += time.perf_counter() - t1

                # CPU to staging buffer B, CPU to pinned CPU, synchronous copy. Can overlap with CUDA to staging buffer A
                # Making this multithreaded does not help, and 'non_blocking=True' does not help either.
                t1 = time.perf_counter()
                sbuf_b.copy_(module_to_cuda.weight.data)  # BOTTLENECK
                cpu_to_staging_time += time.perf_counter() - t1
            timings['gpu_to_staging'] = gpu_to_staging_time
            timings['cpu_to_staging'] = cpu_to_staging_time

        with torch.cuda.stream(stream):
            event_sync_time = 0
            staging_to_gpu_time = 0
            staging_to_cpu_time = 0
            
            # Windows optimization: Batch event syncs instead of one-by-one
            if self.is_windows:
                # Sync all events at once (reduces overhead on Windows)
                t1 = time.perf_counter()
                for event in events:
                    event.synchronize()
                event_sync_time = time.perf_counter() - t1
                
                # Then do all copies without interleaved syncs
                t1 = time.perf_counter()
                for sbuf_b, (_, _, cuda_data_view, _) in zip(self.staging_buffer_b, weight_swap_jobs):
                    cuda_data_view.copy_(sbuf_b, non_blocking=True)
                staging_to_gpu_time = time.perf_counter() - t1
                
                t1 = time.perf_counter()
                for sbuf_a, (_, _, _, cpu_data_view) in zip(self.staging_buffer_a, weight_swap_jobs):
                    cpu_data_view.copy_(sbuf_a)
                staging_to_cpu_time = time.perf_counter() - t1
                
                # Update references
                for sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    module_to_cuda.weight.data = cuda_data_view
                    module_to_cpu.weight.data = cpu_data_view
            else:
                # Linux: Keep original interleaved pattern (works well there)
                for event, sbuf_a, sbuf_b, (module_to_cpu, module_to_cuda, cuda_data_view, cpu_data_view) in zip(
                    events, self.staging_buffer_a, self.staging_buffer_b, weight_swap_jobs
                ):
                    t1 = time.perf_counter()
                    event.synchronize()
                    event_sync_time += time.perf_counter() - t1

                    t1 = time.perf_counter()
                    cuda_data_view.copy_(sbuf_b, non_blocking=True)
                    staging_to_gpu_time += time.perf_counter() - t1

                    t1 = time.perf_counter()
                    cpu_data_view.copy_(sbuf_a)
                    staging_to_cpu_time += time.perf_counter() - t1

                    module_to_cuda.weight.data = cuda_data_view
                    module_to_cpu.weight.data = cpu_data_view
                    
            timings['event_sync'] = event_sync_time
            timings['staging_to_gpu'] = staging_to_gpu_time
            timings['staging_to_cpu'] = staging_to_cpu_time

        t0 = time.perf_counter()
        stream.synchronize()  # Synchronize staging buffer B to CUDA
        # Removed redundant current_stream sync - stream.synchronize() is sufficient
        timings['final_sync'] = time.perf_counter() - t0

        total_time = time.perf_counter() - overall_start
        
        # Print detailed timing statistics on ALL platforms for comparison
        if not hasattr(self, '_swap_count'):
            self._swap_count = 0
        
        self._swap_count += 1
        if self._swap_count % 10 == 0:  # Print every 10th swap
            platform_indicator = "🪟 WINDOWS" if self.is_windows else "🐧 LINUX"
            pinned_status = "pinned" if self.use_pinned_memory else "unpinned"
            print(f"\n[{self.block_type}] {platform_indicator} BLOCK SWAP #{self._swap_count} ({pinned_status}):")
            print(f"  Total: {total_time*1000:.1f}ms | Modules: {len(weight_swap_jobs)}")
            print(f"  Build jobs: {timings.get('build_jobs', 0)*1000:.1f}ms")
            print(f"  Create buffers: {timings.get('create_buffers', 0)*1000:.1f}ms")
            print(f"  GPU→Staging: {timings.get('gpu_to_staging', 0)*1000:.1f}ms")
            print(f"  CPU→Staging: {timings.get('cpu_to_staging', 0)*1000:.1f}ms")
            print(f"  Event sync: {timings.get('event_sync', 0)*1000:.1f}ms")
            print(f"  Staging→GPU: {timings.get('staging_to_gpu', 0)*1000:.1f}ms")
            print(f"  Staging→CPU: {timings.get('staging_to_cpu', 0)*1000:.1f}ms")
            print(f"  Final sync: {timings.get('final_sync', 0)*1000:.1f}ms")

    def swap_weight_devices(self, block_to_cpu: nn.Module, block_to_cuda: nn.Module):
        if self.cuda_available:
            self.swap_weight_devices_cuda(self.device, block_to_cpu, block_to_cuda)
        else:
            swap_weight_devices_no_cuda(self.device, block_to_cpu, block_to_cuda)

    def _submit_move_blocks(self, blocks, block_idx_to_cpu, block_idx_to_cuda):
        def move_blocks(bidx_to_cpu, block_to_cpu, bidx_to_cuda, block_to_cuda):
            if self.debug:
                start_time = time.perf_counter()
                print(
                    f"[{self.block_type}] Move block {bidx_to_cpu} to CPU and block {bidx_to_cuda} to {'CUDA' if self.cuda_available else 'device'}"
                )

            self.swap_weight_devices(block_to_cpu, block_to_cuda)

            if self.debug:
                print(
                    f"[{self.block_type}] Moved blocks {bidx_to_cpu} and {bidx_to_cuda} in {time.perf_counter() - start_time:.2f}s"
                )
            return bidx_to_cpu, bidx_to_cuda  # , event

        block_to_cpu = blocks[block_idx_to_cpu]
        block_to_cuda = blocks[block_idx_to_cuda]

        if self.thread_pool is None:
            # Windows: synchronous execution (no threading overhead)
            result = move_blocks(block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda)
            # Create a simple object that mimics Future.result()
            class SyncResult:
                def __init__(self, value):
                    self._value = value
                def result(self):
                    return self._value
            self.futures[block_idx_to_cuda] = SyncResult(result)
        else:
            # Linux/Mac: asynchronous execution with threading
            self.futures[block_idx_to_cuda] = self.thread_pool.submit(
                move_blocks, block_idx_to_cpu, block_to_cpu, block_idx_to_cuda, block_to_cuda
            )

    def _wait_blocks_move(self, block_idx):
        if block_idx not in self.futures:
            return

        if self.debug:
            print(f"[{self.block_type}] Wait for block {block_idx}")
            start_time = time.perf_counter()

        future = self.futures.pop(block_idx)
        _, bidx_to_cuda = future.result()

        assert block_idx == bidx_to_cuda, f"Block index mismatch: {block_idx} != {bidx_to_cuda}"

        if self.debug:
            print(f"[{self.block_type}] Waited for block {block_idx}: {time.perf_counter() - start_time:.2f}s")


class ModelOffloader(Offloader):
    """
    supports forward offloading
    """

    def __init__(
        self,
        block_type: str,
        blocks: list[nn.Module],
        num_blocks: int,
        blocks_to_swap: int,
        supports_backward: bool,
        device: torch.device,
        debug: bool = False,
    ):
        super().__init__(block_type, num_blocks, blocks_to_swap, device, debug)

        self.supports_backward = supports_backward
        self.forward_only = not supports_backward  # forward only offloading: can be changed to True for inference

        if self.supports_backward:
            # register backward hooks
            self.remove_handles = []
            for i, block in enumerate(blocks):
                hook = self.create_backward_hook(blocks, i)
                if hook is not None:
                    handle = block.register_full_backward_hook(hook)
                    self.remove_handles.append(handle)

    def set_forward_only(self, forward_only: bool):
        self.forward_only = forward_only

    def __del__(self):
        if self.supports_backward:
            for handle in self.remove_handles:
                handle.remove()

    def create_backward_hook(self, blocks: list[nn.Module], block_index: int) -> Optional[callable]:
        # -1 for 0-based index
        num_blocks_propagated = self.num_blocks - block_index - 1
        swapping = num_blocks_propagated > 0 and num_blocks_propagated <= self.blocks_to_swap
        waiting = block_index > 0 and block_index <= self.blocks_to_swap

        if not swapping and not waiting:
            return None

        # create  hook
        block_idx_to_cpu = self.num_blocks - num_blocks_propagated
        block_idx_to_cuda = self.blocks_to_swap - num_blocks_propagated
        block_idx_to_wait = block_index - 1

        def backward_hook(module, grad_input, grad_output):
            if self.debug:
                print(f"Backward hook for block {block_index}")

            if swapping:
                self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
            if waiting:
                self._wait_blocks_move(block_idx_to_wait)
            return None

        return backward_hook

    def prepare_block_devices_before_forward(self, blocks: list[nn.Module]):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        if self.debug:
            print(f"[{self.block_type}] Prepare block devices before forward")

        for b in blocks[0 : self.num_blocks - self.blocks_to_swap]:
            b.to(self.device)
            weighs_to_device(b, self.device)  # make sure weights are on device

        cpu_device = torch.device("cpu")
        for b in blocks[self.num_blocks - self.blocks_to_swap :]:
            b.to(self.device)  # move block to device first. this makes sure that buffers (non weights) are on the device
            weighs_to_device(b, cpu_device)  # make sure weights are on cpu

        _synchronize_device(self.device)
        _clean_memory_on_device(self.device)

    def wait_for_block(self, block_idx: int):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self._wait_blocks_move(block_idx)

    def submit_move_blocks_forward(self, blocks: list[nn.Module], block_idx: int):
        # check if blocks_to_swap is enabled
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return

        # if backward is enabled, we do not swap blocks in forward pass more than blocks_to_swap, because it should be on GPU
        if not self.forward_only and block_idx >= self.blocks_to_swap:
            return

        block_idx_to_cpu = block_idx
        block_idx_to_cuda = self.num_blocks - self.blocks_to_swap + block_idx
        block_idx_to_cuda = block_idx_to_cuda % self.num_blocks  # this works for forward-only offloading
        self._submit_move_blocks(blocks, block_idx_to_cpu, block_idx_to_cuda)
