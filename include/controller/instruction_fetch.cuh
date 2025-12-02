#pragma once

#include "kittens.cuh"

// HIP: Include HIP runtime
#include <hip/hip_runtime.h>

#include "../util.cuh"

namespace megakernel {
namespace controller {

// New AMD-compatible load_instructions function replacing TMA/inline PTX
template<typename config> 
__device__ void inline load_instructions(int *instructions, int task_iter, const typename config::globals &globals, kittens::hip_semaphore &bar) {
    if(::kittens::laneid() == 0) {
        constexpr int ints_per_instruction = config::INSTRUCTION_WIDTH; // Or globals.instructions.cols() if static dim available
        constexpr int bytes = ints_per_instruction * sizeof(int);
        
        // AMD: No tma::expect_bytes equivalent needed for simple async copy.
        // We use the pointer-based async copy intrinsic.
        
        int *dst_ptr = instructions; // Shared memory destination
        const int *src_ptr = &globals.instructions[kittens::coord<>{(int)(get_worker_id()), task_iter, 0}]; // Global memory source

        // AMD: Async copy from Global to Shared (LDS)
        // __memcpy_async is a standard HIP/CUDA intrinsic for this.
        // It maps to cp.async on NVIDIA and potentially buffer_load_lds on AMD.
        // Alternatively, use the HipKittens intrinsic if exposed: 
        // kittens::copy_async(dst_tile, src_tile, bar) -> but here we have raw pointers.
        
        // Using standard HIP async copy:
        __memcpy_async(dst_ptr, src_ptr, bytes, __builtin_amdgcn_s_memtime()); 
        
        // NOTE: The original code passed a 'bar' (mbarrier).
        // On AMD, we don't attach the barrier to the instruction.
        // Instead, we just issue the load. The caller typically waits for completion 
        // using a wave barrier or s_waitcnt. 
        // The controller logic we ported earlier calls 'load_instructions' without the semaphore
        // and then waits on a wave_barrier, which is correct for this implementation.
    }
}

// Fallback / Overload for the specific signature used in main_loop
template <typename config, typename globals>
__device__ void inline load_instructions(int *instruction,
                                         int instruction_index,
                                         const globals &g) {
    auto laneid = ::kittens::laneid();

    auto src_ptr = &g.instructions[kittens::coord<>{(int)(get_worker_id()),
                                                    instruction_index, 0}];
    // static assert it's an int*
    static_assert(std::is_same<decltype(src_ptr), int *>::value,
                  "src_ptr is not an int*");

    static_assert(config::INSTRUCTION_WIDTH <= 32);

    if (laneid < config::INSTRUCTION_WIDTH) {
        instruction[laneid] = src_ptr[laneid];
    }
}

template <typename config, typename globals>
__device__ void inline instruction_fetch_loop(
    const globals &g, ::megakernel::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16,
                  "This would be an absurd thing to do.");
    int num_iters = g.instructions.rows();
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {
        
        // AMD: Monotonic wait target calculation
        // If reusing the slot (index >= stages), we wait for the *previous* completion.
        // Wait target = how many times this slot has been fully consumed before.
        // target = index / STAGES
        int wait_target = kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES;

        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES) {
            kvms.instruction_finished[kvms.instruction_ring].wait(wait_target);
        }

        // Call load_instructions (removed semaphore arg to match definition)
        load_instructions<config, globals>(
            &kvms.instruction()[0], kvms.instruction_index, g);
        
        // AMD: Barrier to ensure instructions are loaded before signaling arrival
        __builtin_amdgcn_wave_barrier();

        // AMD: Manually signal arrival (replaces the semaphore arg passed in NVIDIA version)
        if (::kittens::laneid() == 0) {
            kvms.instruction_arrived[kvms.instruction_ring].arrive();
        }
    }
}

} // namespace controller
} // namespace megakernel