#pragma once

#include "llama.cuh"
#include <hip/hip_runtime.h>

template <typename Config, kittens::ducks::sv::all sv_t>
__device__ static inline auto
rms_norm(const sv_t &rms_scale_smem, const sv_t &activations_smem,
         float rms_norm_eps, void *scratch_memory) {
    using rv_t = kittens::rv_fl<sv_t::length>;
    rv_t activations_vec, sq_activations_vec, rms_scale_vec;

    kittens::load(activations_vec, activations_smem);
    kittens::copy(sq_activations_vec, activations_vec);
    kittens::mul(sq_activations_vec, sq_activations_vec, sq_activations_vec);
    float partial_sum = kittens::sum(sq_activations_vec);

    float *smem_rms_partial_sums = (float *)scratch_memory;
    if (kittens::laneid() == 0) {
        smem_rms_partial_sums[kittens::warpid()] = partial_sum;
    }
    kittens::group<Config::NUM_CONSUMER_WARPS>::sync(0);

    float full_sum = 0;
#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {
        full_sum += smem_rms_partial_sums[i];
    }

    float variance = full_sum / 2048.0f;
    float rms_scale = rsqrtf(variance + rms_norm_eps);

    kittens::mul(activations_vec, activations_vec, rms_scale);
    kittens::load(rms_scale_vec, rms_scale_smem);
    kittens::mul(activations_vec, activations_vec, rms_scale_vec);

    return activations_vec;
}

template <typename GlobalTile, typename SharedTile, typename Coord>
__device__ inline void manual_hip_store_add(GlobalTile& global_dest, const SharedTile& shared_src, const Coord& coord) {
    constexpr int Size = SharedTile::length;

    // shared registers
    kittens::rv_bf<Size> computed_reg;
    // from global register
    kittens::rv_bf<Size> residual_reg; 

    // to store in global register
    kittens::rv_bf<Size> result_reg;  

    kittens::rv_fl<Size> comp_f, res_f; // intermediate rgisters

    kittens::load(computed_reg, shared_src);
    kittens::load(residual_reg, global_dest, coord);

    // convert to float
    kittens::copy(comp_f, computed_reg);
    kittens::copy(res_f, residual_reg);

    // add and store in res_F
    kittens::add(res_f, res_f, comp_f);

    // store as bf16 again
    kittens::copy(result_reg, res_f);

    kittens::store(global_dest, result_reg, coord);

    __threadfence();
    __builtin_amdgcn_s_waitcnt(0);
}

#ifdef KITTENS_BLACKWELL
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_bf<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename kittens::rt_fl<16, 16>::col_vec;
    // using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::broadcast_col(broadcast_activations, row_activations);
    kittens::load(weights, weights_smem);
    kittens::rt_fl<16, 16> out_activations;
    kittens::zero(out_activations);
    kittens::mma_ABt(out_activations, weights, broadcast_activations,
                  out_activations);
    rcv_t sum_col_vec;
    kittens::row_max(sum_col_vec, out_activations);

    rv_t sum_vec;
    kittens::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    // kittens::sync();
     __builtin_amdgcn_wave_barrier();
}
#else
template <kittens::ducks::st::all st_t>
__device__ static inline void matvec(kittens::sv_fl<st_t::rows> &out_smem,
                                     st_t &weights_smem,
                                     kittens::rv_fl<st_t::cols> &activations) {
    using rt_t = kittens::rt_fl<st_t::rows, st_t::cols>;
    using rrv_t = typename rt_t::row_vec;
    using rcv_t = typename rt_t::col_vec;
    using rv_t = kittens::rv_fl<st_t::rows>;
    using sv_t = kittens::sv_bf<st_t::rows>;

    rrv_t row_activations;
    kittens::copy(row_activations, activations);

    rt_t broadcast_activations, weights;
    kittens::broadcast_col(broadcast_activations, row_activations);
    kittens::load(weights, weights_smem);
    kittens::mul(broadcast_activations, broadcast_activations, weights);
    rcv_t sum_col_vec;
    kittens::row_sum(sum_col_vec, broadcast_activations);

    rv_t sum_vec;
    kittens::copy(sum_vec, sum_col_vec);

    if (kittens::laneid() < 16) {
        out_smem[kittens::laneid()] = sum_vec[0][0];
    }
    // kittens::sync();
     __builtin_amdgcn_wave_barrier();
}
#endif

template <typename Config, kittens::ducks::sv::all sv_t, typename rv_t,
          int SCRATCH_BYTES_PER_WARP>
__device__ static inline void matvec_reduce(uint8_t *scratch, rv_t &sum_vec) {
    rv_t part_vec;
    kittens::zero(sum_vec);

#pragma unroll
    for (int i = 0; i < Config::NUM_CONSUMER_WARPS; i++) {

        // TODO: for now, deliberately not using sizeof(sv_t) here because we've
        // had alignment issues before.
        sv_t &part =
            *reinterpret_cast<sv_t *>(scratch + (i * SCRATCH_BYTES_PER_WARP));

        kittens::load(part_vec, part);
        kittens::add(sum_vec, sum_vec, part_vec);
    }
}

