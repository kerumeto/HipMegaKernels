#pragma once

#include "kittens.cuh"
#include "config.cuh"
#include "util.cuh"
#include "controller/controller.cuh"
#include "launcher.cuh"
#include "storer.cuh"
#include "loader.cuh"
#include "consumer.cuh"
#include "noop.cuh"

namespace megakernel {

template <typename config, typename globals, typename... ops>
__device__ inline void mk_internal(const globals &g) {
    uint64_t start_time = (uint64_t)clock64();
#ifdef MK_DEBUG
    // if (threadIdx.x == 0)
    //     printf("Thread %d: Kernel launched\n", threadIdx.x);
    // group<config::NUM_WARPS>::sync(15);
#endif
    __shared__ instruction_state_t<config>
    instruction_state[config::INSTRUCTION_PIPELINE_STAGES]
    __attribute__((aligned(128)));
    __shared__ kittens::hip_semaphore
        page_finished[config::NUM_PAGES]
                     [config::INSTRUCTION_PIPELINE_STAGES_BITS],
        instruction_arrived[config::INSTRUCTION_PIPELINE_STAGES],
        instruction_finished[config::INSTRUCTION_PIPELINE_STAGES],
#ifdef KITTENS_BLACKWELL
        tensor_finished,
#endif
        semaphores_ready;
    extern __shared__ int __shm[];
    void *aligned_shm_addr =
        (void *)((1023 + (uint64_t)&__shm[0]) & ~(uint64_t)1023);
    typename state<config>::page_array_t &pages =
        *reinterpret_cast<typename state<config>::page_array_t *>(
            aligned_shm_addr);
#ifdef KITTENS_BLACKWELL
    // typename state<config>::tensor_allocator_t tensor_alloc{};
#endif

#ifdef MK_DEBUG
    // if (threadIdx.x == 0)
    //     printf("Thread %d: Pre-MKS creation\n", threadIdx.x);
    // group<config::NUM_WARPS>::sync(15);
#endif
    state<config> mks{instruction_state,
                      instruction_arrived,
                      instruction_finished,
                      0,
                      0,
                      {/* ... */},
                      pages,
                      page_finished,
#ifdef KITTENS_BLACKWELL
                    //   tensor_finished,
#endif
                      semaphores_ready,
                      start_time
#ifdef KITTENS_BLACKWELL
                    //   ,
                    //   tensor_alloc
#endif
    }; // megakernel state

#ifdef MK_DEBUG
    // if (threadIdx.x == 0)
    //     printf("Thread %d: Created MKS\n", threadIdx.x);
    // group<config::NUM_WARPS>::sync(15);
#endif

    // Zero initial timings memory.
    if (threadIdx.x < config::TIMING_WIDTH) {
#pragma unroll
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
            instruction_state[i].timings[threadIdx.x] = 0;
        }
    }

    if (threadIdx.x < config::INSTRUCTION_PIPELINE_STAGES) {
        init_semaphore(instruction_arrived[threadIdx.x], 1);
        init_semaphore(instruction_finished[threadIdx.x],
                       config::NUM_WARPS - 1);
    }
    if (threadIdx.x < config::NUM_PAGES) {
        for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES_BITS; i++) {
            auto count = config::NUM_CONSUMER_WARPS * (1 << i);
            init_semaphore(page_finished[threadIdx.x][i], count);
            arrive(page_finished[threadIdx.x][i], count);
        }
    }
    if (threadIdx.x == 0) {
#ifdef KITTENS_BLACKWELL
        // init_semaphore(tensor_finished, config::NUM_CONSUMER_WARPS);
        // arrive(tensor_finished,
        //        config::NUM_CONSUMER_WARPS); // Flip to state 0, to mark that it
        //                                     // starts as available.
#endif
        init_semaphore(semaphores_ready, 1);
    }

    // dont neeed this for md
    // asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    // Use a memory fence to ensure that semaphores are actually initialized in
    // and can be seen by other threads. This DOES NOT sync threads and wait
    // for work to be completed. It simply ensures we point to new memory
    // values created/modified by threads.
    __threadfence_block();

    if (config::CLUSTER_BLOCKS == 1){
        // kittens::everyone::sync(15); // all warps must arrive here, confirming semaphore
        //                     // initialization is visible to all threads.
        // The above code used ID 15 to wait on barrier id 15, but these
        // barrier ids are not exposed in AMD. Instead just sync all threads.
        // Remmeber that syncing all threads within a thread block implicitly
        // syncs all warps.
        // Hierarchy is thread block -> warps (x warps; however many fit in a block) -> threads (32 per warp)
        __syncthreads();
    }
    else {
        // On modern NVIDIA GPUs, you are able to group multiple thread blocks
        // into clusters that can share memory and sync with each other.
        // AMD does not support this behavior and would instead require 
        // synchronization using global memory atomics. For now, we just ensure
        // that we run with cluster size 1, and sync threads within the block.
        // kittens::everyone::tma::cluster::sync();
        static_assert(false, "CLUSTER_BLOCKS > 1 is not supported on AMD GPUs");
    }

#ifdef MK_DEBUG
    // if (blockIdx.x == 0 && threadIdx.x == 0)
    //     mks.print();
#endif

    if (kittens::warpid() < config::NUM_CONSUMER_WARPS) {
        // Not supported in AMD, register usage is cannot be dynamically changed
	    //kittens::warpgroup::increase_registers<config::CONSUMER_REGISTERS>();
        ::megakernel::consumer::main_loop<config, globals, ops...>(g, mks);
    } else {
        // Not supported in AMD, register usage is cannot be dynamically changed
        // kittens::warpgroup::decrease_registers<config::NON_CONSUMER_REGISTERS>();
        switch (kittens::warpgroup::warpid()) {
        case 0:
            ::megakernel::loader::main_loop<config, globals, ops...>(g, mks);
            break;
        case 1:
            ::megakernel::storer::main_loop<config, globals, ops...>(g, mks);
            break;
        case 2:
            ::megakernel::launcher::main_loop<config, globals, ops...>(g, mks);
            break;
        case 3:
            ::megakernel::controller::main_loop<config, globals, ops...>(g,
                                                                         mks);
            break;
        default:
            // asm volatile("trap;");
            __builtin_trap();
        }
    }

#ifdef MK_DEBUG
    printf("Thread %d arriving at final barrier\n", threadIdx.x);
#endif

    // See above comments on why CLUSTER_BLOCKS == 1 is required for AMD GPUs.
    if (config::CLUSTER_BLOCKS > 1) {
        // kittens::everyone::tma::cluster::sync();
        __syncthreads();
    }
    else {
        // kittens::everyone::sync(15);
        static_assert(false, "CLUSTER_BLOCKS > 1 is not supported on AMD GPUs");
    }

#ifdef MK_DEBUG
    uint64_t end_time = (uint64_t)clock64();
    if (threadIdx.x == 0)
        printf("Overall VM execution time: %lu\n", end_time - start_time);
#endif
}

// Forward a NoOp to the VM, to ensure that the VM can support zeros.
template <typename config, typename globals, typename... ops>
struct megakernel_wrapper {
    __device__ inline static void run(const globals &g) {
        mk_internal<config, globals, NoOp<config>, ops...>(g);
    }
};

template <typename config, typename globals, typename... ops>
__launch_bounds__(config::NUM_THREADS, 1)
  //  __cluster_dims__(config::CLUSTER_BLOCKS) __global__
  // no such thing as grid_constant in amd
       	__global__
       	void mk(const globals g) {
    megakernel_wrapper<config, globals, ops...>::run(g);
}

} // namespace megakernel
