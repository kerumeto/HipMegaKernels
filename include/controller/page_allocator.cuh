#pragma once

#include "kittens.cuh"

// HIP: Include HIP runtime
#include <hip/hip_runtime.h>

#include "../util.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals>
struct page_allocator_op_dispatcher {
    template <typename op> struct dispatcher {
        __device__ static inline int
        run(const globals &g, typename config::instruction_t &instruction,
            int &query) {
            return op::controller::release_lid(g, instruction, query);
        }
    };
};

template <typename config, typename globals, typename... ops>
__device__ void inline page_allocator_loop(const globals &g,
                                           ::megakernel::state<config> &kvms) {
    static_assert(config::INSTRUCTION_PIPELINE_STAGES <= 16,
                  "This would be an absurd thing to do."); // real
    
    // AMD: wave_barrier syncs the whole wavefront, so mask is implicitly ignored/handled
    // by the fact that we expect the warp to be converged here.
    // constexpr uint32_t membermask = 0xFFFFFFFF >> (32 - config::NUM_PAGES);
    
    int num_iters = g.instructions.rows();
    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {

        // AMD: Monotonic wait for cleanup/slot reuse
        // If index >= STAGES, we are reusing a slot. 
        // We wait for the *previous* usage (index - STAGES) to be finished.
        // Generation count = (prev_index / STAGES) + 1
        // (index - STAGES) / STAGES + 1  ==>  index / STAGES
        // IMPORTANT: Double check this logic for waiting on counters
        int wait_target_finished = kvms.instruction_index / config::INSTRUCTION_PIPELINE_STAGES;

        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES)
            kvms.instruction_finished[kvms.instruction_ring].wait(wait_target_finished);

        int next_pid;
        if (kvms.instruction_index == 0)
            next_pid = kittens::laneid();
        else {
            int last_instruction_ring =
                (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES -
                 1) %
                config::INSTRUCTION_PIPELINE_STAGES;
            
            // AMD: Monotonic wait for previous instruction arrival
            // We are waiting for instruction (index - 1) to arrive.
            // Generation count = ((index - 1) / STAGES) + 1
            int wait_target_arrived = ((kvms.instruction_index - 1) / config::INSTRUCTION_PIPELINE_STAGES) + 1;
            
            kvms.instruction_arrived[last_instruction_ring].wait(wait_target_arrived);

            int lane = kittens::laneid();
            int opcode =
                kvms.all_instructions[last_instruction_ring].instructions[0];
            // int lid = dispatch_op<
            //     page_allocator_op_dispatcher<config, globals>::dispatcher,
            //     ops...>::template run<int, config, globals,
            //                           config::instruction_t, int>(
            //     opcode, g,
            //     kvms.all_instructions[last_instruction_ring].instructions,
            //     lane);
            int lid = dispatch_op<
                page_allocator_op_dispatcher<config, globals>::template dispatcher, // FIXED
                ops...>::template run<int, config, globals,
                                    config::instruction_t, int>(
                    opcode, g,
                    kvms.all_instructions[last_instruction_ring].instructions,
                    lane
                );
            next_pid =
                kvms.all_instructions[last_instruction_ring].pid_order[lid];
        }
        kvms.pid_order()[kittens::laneid()] = next_pid;
        
        // AMD: Replaces bar.warp.sync
        __builtin_amdgcn_wave_barrier();

        if (kittens::laneid() == 0) {
            // AMD: Replaces kittens::arrive(..., 1)
            kvms.instruction_arrived[kvms.instruction_ring].arrive();
        }
    }
}

} // namespace controller
} // namespace megakernel
