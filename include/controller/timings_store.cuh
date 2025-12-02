#pragma once

#include "kittens.cuh"

#include "../util.cuh"

namespace megakernel {
namespace controller {

template <typename config, typename globals>
__device__ void inline store_timings(int *timings, int instruction_index,
                                     const globals &g) {
    // constexpr int bytes = config::TIMING_WIDTH * sizeof(int);
    // uint32_t src_ptr = static_cast<uint32_t>(__cvta_generic_to_shared(timings));
    // uint64_t dst_ptr = (uint64_t)(&g.timings[kittens::coord<>{
    //     (int)(get_worker_id()), instruction_index, 0}]);
    // asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
    // asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n"
    //              :
    //              : "l"(dst_ptr), "r"(src_ptr), "n"(bytes)
    //              : "memory");
    // kittens::tma::store_commit_group();

    int* dst_ptr = &g.timings[kittens::coord<>{
        (int)(get_worker_id()), instruction_index, 0}];

    int lane = kittens::laneid();
    for (int i = lane; i < config::TIMING_WIDTH; i += kittens::WARP_THREADS) {
        dst_ptr[i] = timings[i];
    }
    __threadfence();
    __builtin_amdgcn_s_waitcnt(0);
}

template <typename config, typename globals>
__device__ void inline store_timings_and_reset(int *timings,
                                               int instruction_index,
                                               const globals &g) {
//     if (kittens::laneid() == 0) {
//         store_timings<config, globals>(timings, instruction_index, g);
//         kittens::tma::store_async_read_wait();
// #ifdef KITTENS_BLACKWELL
//         uint32_t src_ptr =
//             static_cast<uint32_t>(__cvta_generic_to_shared(timings));
//         asm volatile("st.bulk.weak [%0], %1, 0;\n" ::"r"(src_ptr),
//                      "n"(config::TIMING_WIDTH *
//                          sizeof(int))); // Reinitialize timing memory as zeros.
// #endif
//     }
// #ifndef KITTENS_BLACKWELL
//     __syncwarp();
//     for (int i = kittens::laneid(); i < config::TIMING_WIDTH; i += kittens::WARP_THREADS) {
//         timings[i] = 0;
//     }
// #endif
    store_timings<config, globals>(timings, instruction_index, g);
    
	__builtin_amdgcn_wave_barrier()

    int lane = kittens::laneid();
    for (int i = lane; i < config::TIMING_WIDTH; i += kittens::WARP_THREADS) {
        timings[i] = 0;
    }
    
    __builtin_amdgcn_wave_barrier()
}

} // namespace controller
} // namespace megakernel
