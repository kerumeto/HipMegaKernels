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
        constexpr int ints_per_instruction = config::INSTRUCTION_WIDTH;
        constexpr int bytes = ints_per_instruction * sizeof(int);
        
        int *dst_ptr = instructions; // Shared memory destination
        const int *src_ptr = &globals.instructions[kittens::coord<>{(int)(get_worker_id()), task_iter, 0}]; // Global memory source

        // FIX: The intrinsic __memcpy_async is not universally supported/declared on AMD.
        // Replace with a synchronous copy since the calling function provides a wave barrier.
        // If an async copy is strictly required, you must use a compiler-defined intrinsic 
        // or a framework function guaranteed to work on AMD, or a simple loop copy.
        
        // Option 1: Synchronous copy using standard library function
        memcpy(dst_ptr, src_ptr, bytes);  
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
