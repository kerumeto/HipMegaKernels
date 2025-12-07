#pragma once

#include "kittens.cuh"

// HIP: Include HIP runtime
#include <hip/hip_runtime.h>

#include "../util.cuh"
#include "instruction_fetch.cuh"
#include "timings_store.cuh"
#include "semaphore_constructor.cuh"
#include "page_allocator.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals, typename... ops>
__device__ void main_loop(const globals &g, ::megakernel::state<config> &kvms) {
    auto laneid = ::kittens::laneid();
    int num_iters = g.instructions.rows();
    int num_semaphores[config::INSTRUCTION_PIPELINE_STAGES];

    // for warps
    static_assert(config::DYNAMIC_SEMAPHORES <= 32);
    static_assert(config::NUM_PAGES <= 32);

    for (kvms.instruction_index = 0, kvms.instruction_ring = 0;
         kvms.instruction_index < num_iters;
         kvms.instruction_index++,
        kvms.instruction_ring =
             ring_advance<config::INSTRUCTION_PIPELINE_STAGES>(
                 kvms.instruction_ring)) {

        // Step 0. if the slot was used in the previous iteration, wait for the
        // previous instruction to complete & invalidate its semaphores
        if (kvms.instruction_index >= config::INSTRUCTION_PIPELINE_STAGES) {
            auto last_slot_instruction_index =
                kvms.instruction_index - config::INSTRUCTION_PIPELINE_STAGES;

            // AMD: Replaced phase-bit logic with monotonic wait target.
            // Calculate how many times this specific ring slot has been used previously.
            // +1 because we are waiting for the completion of the previous usage.
            // -----------------------------------------------------------
            // IMPORTANT: Make sure that the semaphore counter is maintained
            // in increasing style. We are not using a phase bit anymore but
            // rather a monotonic counter.
            // -----------------------------------------------------------
            int wait_target = (last_slot_instruction_index / config::INSTRUCTION_PIPELINE_STAGES) + 1;
            
            kvms.instruction_finished[kvms.instruction_ring].wait(wait_target);

            // AMD: We enable this now. Resetting internal semaphores to 0 is safe/required
            // even if the outer loop uses monotonic counters, because the internal 
            // ops (matmul, etc) likely expect semaphores to start at 0.
            if (laneid < num_semaphores[kvms.instruction_ring]) {
                invalidate_semaphore(
                    kvms.all_instructions[kvms.instruction_ring]
                        .semaphores[laneid]); 
            }

            // AMD: Wave barrier to ensure visibility of resets before reuse
            __builtin_amdgcn_wave_barrier();

            if (laneid == 0) {
                if constexpr (config::TIMING_RECORD_ENABLED) {
                    kvms.record(TEVENT_CONTROLLER_END);
                    store_timings_and_reset<config, globals>(
                        &kvms.all_instructions[kvms.instruction_ring]
                             .timings[0],
                        last_slot_instruction_index, g);
                }
            }
        }

        if (laneid == 0) {
            kvms.record(TEVENT_CONTROLLER_START);
        }

        // Step 1. Load instructions (no semaphores used)
        load_instructions<config, globals>(&kvms.instruction()[0],
                                           kvms.instruction_index, g);

        if (laneid == 0) {
            kvms.record(TEVENT_IFETCH_DONE);
        }

        // Step 2. Establish physical page order
        int last_instruction_ring =
            (kvms.instruction_ring + config::INSTRUCTION_PIPELINE_STAGES - 1) %
            config::INSTRUCTION_PIPELINE_STAGES;

        if (kvms.instruction_index == 0) {
            if (laneid < config::NUM_PAGES) {
                kvms.pid_order()[laneid] = laneid;
            }
        } else {
            auto last_opcode =
                kvms.all_instructions[last_instruction_ring].instructions[0];

            if (laneid < config::NUM_PAGES) {
                // Original:
//                 int lid = dispatch_op<
//                     page_allocator_op_dispatcher<config, globals>::dispatcher,
//                     ops...>::template run<int, config, globals,
//                                           config::instruction_t, int>(
//                     last_opcode, g,
//                     kvms.all_instructions[last_instruction_ring].instructions,
//                     laneid);

                int lid = dispatch_op<
                    page_allocator_op_dispatcher<config, globals>::template dispatcher,
                    ops...>::template run<int, config, globals,
                                          config::instruction_t, int>(
                    last_opcode, g,
                    kvms.all_instructions[last_instruction_ring].instructions,
                    laneid);

                kvms.pid_order()[laneid] =
                    kvms.all_instructions[last_instruction_ring].pid_order[lid];
            }
        }

        if (laneid == 0) {
            kvms.record(TEVENT_PAGE_ALLOC_DONE);
        }

        // Step 3. Construct semaphores
        int opcode = kvms.instruction()[0];
        if (opcode == 0) {
            num_semaphores[kvms.instruction_ring] = 0;
        } else {
            // Original:
//             if (laneid == 0) {
//                 num_semaphores[kvms.instruction_ring] = dispatch_op<
//                     semaphore_constructor_op_dispatcher<config,
//                                                         globals>::dispatcher,
//                     ops...>::template run<int, config, globals,
//                                           ::megakernel::state<config>>(opcode,
//                                                                        g, kvms);
//             }
            if (laneid == 0) {
                num_semaphores[kvms.instruction_ring] = dispatch_op<
                    semaphore_constructor_op_dispatcher<config,
                                                        globals>::template dispatcher,
                    ops...>::template run<int, config, globals,
                                          ::megakernel::state<config>>(opcode,
                                                                       g, kvms);
            }

            // HIP supports __shfl_sync, mask is usually -1 or 0xffffffff
            auto shfl_val = __shfl_sync(
                0xffffffff, num_semaphores[kvms.instruction_ring], 0);

            // broadcast the result to all lanes
            num_semaphores[kvms.instruction_ring] = shfl_val;
        }

        if (laneid == 0) {
            kvms.record(TEVENT_SEMS_SETUP);
            // Step 4. Let the rest of the world know that next instruction is
            // ready to roll!
            
            // AMD: Call .arrive() on the custom semaphore struct
            kvms.instruction_arrived[kvms.instruction_ring].arrive();
        }
    }

    // invalidate remaining semaphores and write out remaining timings
    // Remember that our pipeline is config::INSTRUCTION_PIPELINE_STAGES (2) deep
    // When the above loop ended, it finished preparing the last 2 instructions,
    // but they have not yet been waited on.
    for (int i = 0; i < config::INSTRUCTION_PIPELINE_STAGES; i++) {
        auto instruction_index =
            num_iters - config::INSTRUCTION_PIPELINE_STAGES + i;
        if (instruction_index < 0) {
            continue;
        }

        auto instruction_ring =
            instruction_index % config::INSTRUCTION_PIPELINE_STAGES;

        // AMD: Monotonic wait calculation for cleanup loop
        int wait_target = (instruction_index / config::INSTRUCTION_PIPELINE_STAGES) + 1;
        kvms.instruction_finished[instruction_ring].wait(wait_target);

        if (laneid < num_semaphores[instruction_ring]) {
            invalidate_semaphore(
                kvms.all_instructions[instruction_ring].semaphores[laneid]);
        }

        kvms.instruction_index = instruction_index;
        kvms.instruction_ring = instruction_ring;
        // record using the current ring
        if (laneid == 0)
            kvms.record(TEVENT_CONTROLLER_END);

        // technically don't need to reset, whatevs?
        store_timings_and_reset<config, globals>(
            &kvms.all_instructions[instruction_ring].timings[0],
            instruction_index, g);
    }
}

} // namespace controller
} // namespace megakernel
