#pragma once

#include "llama.cuh" // Assuming this and subsequent includes are HIP-compatible
#include <limits>
// HIP: Add HIP runtime header if not already included by kittens
#include <hip/hip_runtime.h>
// HIP: Include the kittens utilities for async loads/stores
#include "kittens.cuh" 

// Explicit declaration of the AMD LLVM intrinsic for load_Q_async
extern "C" __device__ void llvm_amdgcn_raw_buffer_load_lds(
    i32x4 rsrc,             // 128-bit buffer resource
    __attribute__((address_space(3))) void* lds_ptr, // Local memory pointer
    int size,                        // Bytes to load (e.g., 4, 16)
    int voffset,                     // Thread-variable offset
    int soffset,                     // Scalar offset
    int offset,                      // Immediate offset
    int aux                          // Auxiliary flags (cache policy)
);

using namespace kittens;
using namespace megakernel;

template <typename config, typename globals> struct attention_partial {
    static constexpr int opcode = OPCODE_PartialAttention;
    static constexpr int NUM_STAGES = 3;
    static constexpr int GQA_RATIO =
        LLAMA_1B_NUM_ATTENTION_HEADS / LLAMA_1B_NUM_KV_HEADS;
    static constexpr int QOL_PAGE = 0;
    static constexpr int KV_PAGE = 1;

    static_assert(GQA_RATIO == 4, "GQA_RATIO must be 4.");
    static_assert(NUM_STAGES <= 4, "Modify page allocation for KVs.");

    using q_rt = kittens::rt_bf<16, LLAMA_1B_HEAD_DIM>; // only 4 rows are used
    using q_st = kittens::st_bf<16, LLAMA_1B_HEAD_DIM>; // only 4 rows are used
    using k_rt = kittens::rt_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM>;
    using v_rt = kittens::rt_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM, col_l>;
    using kv_st = kittens::st_bf<LLAMA_1B_KV_BLOCK_SIZE, LLAMA_1B_HEAD_DIM>;
    using attn_fl_rt =
        kittens::rt_fl<16, LLAMA_1B_KV_BLOCK_SIZE, kittens::ducks::rt_layout::col>; // only 4 values are used. mma_ABt requires column major layout.
    using attn_bf_rt =
        kittens::rt_bf<16, LLAMA_1B_KV_BLOCK_SIZE>; // only 4 values are used
    using max_vec_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using max_vec_sv = kittens::sv_fl<16>;              // only 4 values are used
    using norm_vec_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using norm_vec_sv = kittens::sv_fl<16>;             // only 4 values are used
    using l_rv =
        col_vec<kittens::rt_fl<16, LLAMA_1B_HEAD_DIM>>; // only 4 values are used
    using l_sv = kittens::sv_fl<16>;                    // only 4 values are used
    using o_rt = kittens::rt_fl<16, LLAMA_1B_HEAD_DIM, kittens::ducks::rt_layout::col>; // only 4 rows are used. mma_AB requires column major layout
    using o_sv = kittens::sv_fl<LLAMA_1B_HEAD_DIM>;
    using o_sv_bf = kittens::sv_bf<LLAMA_1B_HEAD_DIM>;

    struct parsed_instruction {
        int layer_idx;
        int kv_head_idx;
        int num_partials;
        int partial_idx;
        __device__ inline parsed_instruction(
            typename config::instruction_t &instruction) {
            layer_idx = instruction[1];
            kv_head_idx = instruction[2];
            num_partials = instruction[3];
            partial_idx = instruction[4];
        }
        __device__ inline parsed_instruction(megakernel::state<config> &s)
            : parsed_instruction(s.instruction()) {}
    };

    // We have 32 dynamic kittens::semaphores total
    __device__ static inline kittens::hip_semaphore &Q_arrived(megakernel::state<config> &s) {
        return s.semaphores()[0];
    }
    __device__ static inline kittens::hip_semaphore &O_arrived(megakernel::state<config> &s) {
        return s.semaphores()[1];
    }
    __device__ static inline kittens::hip_semaphore &L_arrived(megakernel::state<config> &s) {
        return s.semaphores()[2];
    }
    __device__ static inline kittens::hip_semaphore &K_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2];
    }
    __device__ static inline kittens::hip_semaphore &V_arrived(megakernel::state<config> &s, int stage) {
        return s.semaphores()[3 + stage * 2 + 1];
    }
    __device__ static inline kittens::hip_semaphore &K_finished(megakernel::state<config> &s,
                                                   int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2];
    }
    __device__ static inline kittens::hip_semaphore &V_finished(megakernel::state<config> &s,
                                                   int stage) {
        return s.semaphores()[3 + NUM_STAGES * 2 + stage * 2 + 1];
    }

    __device__ static inline void wait_QOL_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(QOL_PAGE));
    }
    __device__ static inline void wait_KV_page(megakernel::state<config> &s) {
        s.wait_page_ready(s.pid(KV_PAGE));
    }
    __device__ static inline void finish_QOL_page(megakernel::state<config> &s) {
        if (kittens::laneid() == 0)
            s.finish_page(s.pid(QOL_PAGE), config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline void finish_KV_page(megakernel::state<config> &s) {
        if (kittens::laneid() == 0)
            s.finish_page(s.pid(KV_PAGE), config::NUM_CONSUMER_WARPS);
    }
    __device__ static inline q_st &get_Q_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<q_st *>(s.pages[pid].data);
    }
    __device__ static inline o_sv (&get_O_smem(megakernel::state<config> &s))[4] {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<o_sv(*)[4]>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st));
    }
    __device__ static inline l_sv &get_L_smem(megakernel::state<config> &s) {
        int pid = s.pid(QOL_PAGE);
        return *reinterpret_cast<l_sv *>(
            reinterpret_cast<char *>(s.pages[pid].data) + sizeof(q_st) +
            sizeof(o_sv) * 4);
    }
    __device__ static inline kv_st &get_K_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (stage * 2));
    }
    __device__ static inline kv_st &get_V_smem(megakernel::state<config> &s, int stage) {
        int pid = s.pid(KV_PAGE);
        return *reinterpret_cast<kv_st *>(
            reinterpret_cast<char *>(s.pages[pid].data) +
            sizeof(kv_st) * (1 + stage * 2));
    }

    template <ducks::sv::all SV, ducks::rt::all RT>
    __device__ static inline void
    store_4_rows(SV (&dst)[4], const RT &src, int row4idx /*= 0, 1, 2, or 3*/) {
        static_assert(RT::rows == 16, "src rows must be 16.");
        static_assert(SV::length == src.cols,
                      "dst length must match src cols.");

        using T2 = typename RT::dtype;
        using U = typename SV::dtype;
        using U2 = typename base_types::packing<U>::packed_type; // e.g., float2 or bf16_2

        int laneid = kittens::laneid(); // Use HipKittens laneid
        int local_row_idx = (laneid % 16) / 4;
        int local_col_idx = laneid % 4;

        // Address logic: Select the correct shared vector based on the lane's local row index
        // We write directly to the pointer provided by the shared vector .data array
        U* dst_base_ptr = &dst[local_row_idx].data[0];

        if (row4idx % 2 == 0 && laneid < 16) { // rows 0~3 or 8~11
            if (row4idx / 2 == 0) {            // rows 0~3
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    // Convert and extract from register tile
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[2]); // note 2, not 1
                    int col_idx = local_col_idx * 2 + j * 16;
                    
                    // Store to shared memory using pointer assignment
                    // Compiles to ds_write instructions on AMD
                    // See shared_to_register.cuh::store - Line 99 to see 
                    // a similar pattern of storing directly into shared memory
                    // DEBUG: the compiler should know that this is a shared
                    // address, make sure to look at the asm to double check
                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx]) = tmp[0];
                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx + 8]) = tmp[1];
                }
            } else { // rows 8~11
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(
                        src.tiles[0][j].data[3]);
                    int col_idx = local_col_idx * 2 + j * 16;

                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx]) = tmp[0];
                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx + 8]) = tmp[1];
                }
            }
        } else if (row4idx % 2 == 1 && laneid >= 16) { // rows 4~7 or 12~15
            if (row4idx / 2 == 0) {                    // rows 4~7
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[0]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[2]);

                    int col_idx = local_col_idx * 2 + j * 16;

                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx]) = tmp[0];
                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx + 8]) = tmp[1];
                }
            } else { // rows 12~15
                for (int j = 0; j < src.width; j++) {
                    U2 tmp[2];
                    tmp[0] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[1]);
                    tmp[1] = base_types::convertor<U2, T2>::convert(src.tiles[0][j].data[3]);

                    int col_idx = local_col_idx * 2 + j * 16;

                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx]) = tmp[0];
                    *reinterpret_cast<U2*>(&dst_base_ptr[col_idx + 8]) = tmp[1];
                }
            }
        }
    }
    // right_fill only accepts row_layout BUT attn_fl_rt and o_rt in column major
    template <ducks::rt::all RT>
    __device__ static inline void right_fill(
        RT &dst, const RT &src, const int col_idx,
        const typename base_types::packing<typename RT::dtype>::unpacked_type
            &val = 0) {
        if (col_idx >= dst.cols)
            return;
#pragma unroll
        for (int i = 0; i < dst.height; i++) {
#pragma unroll
            for (int j = 0; j < dst.width; j++) {
#pragma unroll
                for (int k = 0; k < dst.packed_per_tile; k++) {
                    const int col_idx_x = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::laneid() % 4) * 2);
                    const int col_idx_y = (j * dst.tile_size_col) +
                                          ((k / 2) * 8) +
                                          ((kittens::laneid() % 4) * 2) + 1;
                    if (col_idx_x >= col_idx) {
                        dst.tiles[i][j].data[k].x = val;
                    } else {
                        dst.tiles[i][j].data[k].x = src.tiles[i][j].data[k].x;
                    }
                    if (col_idx_y >= col_idx) {
                        dst.tiles[i][j].data[k].y = val;
                    } else {
                        dst.tiles[i][j].data[k].y = src.tiles[i][j].data[k].y;
                    }
                }
            }
        }
    }
    // This is super specific to loading Q in a single kittens::warp
    // Mainly two things are different:
    //   1. Ignores Q global dimensions
    //   2. Only loads 4 rows of Q, not 16 (assumes GQA_RATIO == 4) --> only 32
    //   calls needed, so single call per thread
    __device__ static inline void
    load_Q_async(q_st &dst, const globals::activations_t &src,
                 const int q_head_start_idx /*0, 4, 8, ...*/) {
        static_assert(LLAMA_1B_HEAD_DIM == 64 && GQA_RATIO == 4,
                      "Fix this function.");
        using T = typename q_st::dtype;
        static_assert(sizeof(typename q_st::dtype) == 2, "Edit load_Q_async");

        // loading 16 bytes (128bits) per thread per instruction to match CUDA original
        // IMPORTANT: may need to change this to 32 bytes since AMD has 64 threads per warp
        constexpr int elem_per_memcpy =
            sizeof(float4) / sizeof(typename q_st::dtype);                  // 8
        constexpr int memcpy_per_row = LLAMA_1B_HEAD_DIM / elem_per_memcpy; // 8

        int laneid = kittens::laneid();
        // thread mapping 64 threads in wavefront to 8 rows of 8 threads each
        // CHECK IMPORTANT: will this change algo correctness later on 

        int row_offset = laneid / memcpy_per_row;
        int col_chunk = laneid % memcpy_per_row;
        // CHECK: assumes 16 x 16 tile
        int tile_row = (q_head_start_idx % 16) + row_offset;
        int tile_col = col_chunk * elem_per_memcpy;

        // Global Memory Address (Source)
        // Create a resource descriptor for the activations buffer
        // CHECK: used a huge limit
        i32x4 srsrc = make_srsrc(src.raw_ptr, 0x7fffffff);

        uint64_t global_row = q_head_start_idx + row_offset;
        uint64_t global_linear_idx = global_row * LLAMA_1B_HEAD_DIM + tile_col;
        uint32_t global_byte_offset = global_linear_idx * sizeof(T);

        // Shared memory address (destination)
        // should use swizzling to avoid bank conflicts (i.e. add offset to lds_base), but not doing so rn for simplicity

        // LDS (AMD Shared mem) is in address space 3
        using as3_ptr = __attribute__((address_space(3))) void*;
        uintptr_t lds_base = reinterpret_cast<uintptr_t>(&dst.data[0]);
        as3_ptr lds_ptr = reinterpret_cast<as3_ptr>(lds_base);

        // CHECK: i dont think the ptrs are quite the same as cuda original
        // Load 16 bytes from Global[voffset] -> Shared[lds_ptr]. synchronous.
        llvm_amdgcn_raw_buffer_load_lds(
            srsrc,              // rsrc: Buffer resource descriptor
            lds_ptr,            // lds:  Destination address in shared memory
            16,     // size: Bytes to load (must be constant 16/4 etc)
            global_byte_offset, // voffset: Variable offset in global memory
            0,                  // soffset: Scalar offset (0 here)
            0,                  // offset:  Immediate offset
            static_cast<int>(coherency::cache_all) // aux: Cache policy
        );
    }

    struct controller {
        static __device__ int
        release_lid(const globals &g,
                    typename config::instruction_t &instruction, int &query) {
            int ret_order[13] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 1};
            return ret_order[query];
        }

        // q is query latr
        // l is normalization layer
        static __device__ int init_semaphores(const globals &g,
                                              megakernel::state<config> &s) {
            // init_semaphore(Q_arrived(s), 0, 1);
            // init_semaphore(O_arrived(s), 0, 1);
            // init_semaphore(L_arrived(s), 0, 1);
            // for (int i = 0; i < NUM_STAGES; i++) {
            //     init_semaphore(K_arrived(s, i), 0, 1);
            //     init_semaphore(V_arrived(s, i), 0, 1);
            //     init_semaphore(K_finished(s, i), 0, 1);
            //     init_semaphore(V_finished(s, i), 0, 1);
            // }
            // return 3 + 4 * NUM_STAGES;

            Q_arrived(s).init(0);
            O_arrived(s).init(0);
            L_arrived(s).init(0);

            for (int i = 0; i < NUM_STAGES; i++) {
                K_arrived(s, i).init(0);
                V_arrived(s, i).init(0);
                K_finished(s, i).init(0);
                V_finished(s, i).init(0);
            }

            return 3 + 4 * NUM_STAGES;
        }
    };
    struct loader {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            auto laneid = kittens::laneid();
            if (laneid >= 2 && laneid < config::NUM_PAGES) {
                int unused_page = s.pid(laneid);
                s.wait_page_ready(unused_page);
                s.finish_page(unused_page, config::NUM_CONSUMER_WARPS);
            }
        }
    };
   struct launcher {
        static __device__ void wait_for_kv(const globals &g, megakernel::state<config> &s,
                                           parsed_instruction &inst) {
            s.record(megakernel::TEVENT_AT_GMEM_WAIT);

            // Wait for the previous ops to finish (16 dims each, so 4 ops on
            // the same head)
            while (*(volatile int *)&g.Bar[{
                       inst.layer_idx, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                       LLAMA_1B_NUM_ATTENTION_HEADS + inst.kv_head_idx}] < 4) {
                // AMD-specific sleep. Arg is roughly # of 64-cycle periods.
                __builtin_amdgcn_s_sleep(1); 
            }

            while (
                *(volatile int *)&g
                     .Bar[{inst.layer_idx, OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                           LLAMA_1B_NUM_ATTENTION_HEADS +
                               LLAMA_1B_NUM_KV_HEADS + inst.kv_head_idx}] < 4) {
                __builtin_amdgcn_s_sleep(1);
            }

            s.record(megakernel::TEVENT_DONE_GMEM_WAIT);
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            // REMOVED: if (kittens::laneid() == 0) 
            // On AMD, global->shared loads are cooperative (vector instructions). 
            // The entire warp must participate in the 'kittens::load' calls below.

#ifdef KITTENS_BLACKWELL
            // Keep existing Blackwell/NVIDIA logic guard if needed, 
            // but for AMD this block is likely irrelevant or handled differently.
            // if (kittens::laneid() == 0) {
            //     s.wait_tensor_ready();
            //     arrive(s.tensor_finished, config::NUM_CONSUMER_WARPS);
            // }
#endif

            // Setup
            parsed_instruction inst{s};
            int seq_len = g.pos_id + 1;
            int total_attn_blocks = (seq_len + LLAMA_1B_KV_BLOCK_SIZE - 1) /
                                    LLAMA_1B_KV_BLOCK_SIZE;
            int blocks_per_partial =
                (total_attn_blocks + inst.num_partials - 1) /
                inst.num_partials;
            int start_blk_idx = inst.partial_idx * blocks_per_partial;
            int end_blk_idx =
                min(start_blk_idx + blocks_per_partial, total_attn_blocks);

            // Wait for the KV page
            wait_KV_page(s);

            if (start_blk_idx >= end_blk_idx)
                finish_KV_page(s);

            // Run the pipeline!
            for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                auto cur_blk_idx = start_blk_idx + i;
                int stage = cur_blk_idx % NUM_STAGES;
                kv_st &K_smem = get_K_smem(s, stage);
                kv_st &V_smem = get_V_smem(s, stage);

                if (i >= NUM_STAGES) {
                    // Assuming 'kittens::wait' maps to a compatible barrier wait 
                    // or is defined in your 'megakernel::state' utils.
                    // kittens::wait(K_finished(s, stage), (i / NUM_STAGES - 1) % 2);
                    // kittens::wait(V_finished(s, stage), (i / NUM_STAGES - 1) % 2);

                    K_finished(s, stage).wait(i / NUM_STAGES);
                    V_finished(s, stage).wait(i / NUM_STAGES);
                }

                if (cur_blk_idx == end_blk_idx - 1 &&
                    inst.partial_idx == inst.num_partials - 1) {
                    wait_for_kv(g, s, inst);
                }

                // REPLACED: kittens::tma::expect -> Not needed (hardware counts)
                // REPLACED: kittens::tma::load_async -> kittens::load (Cooperative Warp Load)
                
                // Load K tile asynchronously
                kittens::load(
                    K_smem, g.k_cache, 
                    {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0}
                );
                // If you are using mbarriers to track arrival manually, signal here:
                // K_arrived(s, stage).arrive(); 
                K_arrived(s, stage).arrive();

                // Load V tile asynchronously
                kittens::load(
                    V_smem, g.v_cache,
                    {inst.layer_idx, cur_blk_idx, inst.kv_head_idx, 0}
                );
                V_arrived(s, stage).arrive();
            }
        }
    };
     struct consumer {
        static __device__ void run(const globals &g, megakernel::state<config> &s) {

            if (kittens::warpid() == 0) {
                // Wait for the previous ops to finish
                parsed_instruction inst{s};
                int q_head_start_idx = inst.kv_head_idx * GQA_RATIO;

                if (kittens::laneid() == 0) {
                    for (int head_offset = 0; head_offset < GQA_RATIO;
                         head_offset++) {
                        // AMD: Use s_sleep for low-power waiting. 
                        // s_sleep(1) is ~64 clock cycles on CDNA.
                        while (*(volatile int *)&g
                                    .Bar[{inst.layer_idx,
                                          OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                                          q_head_start_idx + head_offset}] <
                                4) {
                            __builtin_amdgcn_s_sleep(1); 
                        }
                    }
                }
                // AMD: Ensure the warp stays converged after scalar loop
                __builtin_amdgcn_wave_barrier();

                // Setup
                int q_head_local_idx =
                    (q_head_start_idx % q_rt::tile_size_row) / 4;
                int seq_len = g.pos_id + 1;
                int total_attn_blocks = (seq_len + LLAMA_1B_KV_BLOCK_SIZE - 1) /
                                        LLAMA_1B_KV_BLOCK_SIZE;
                int blocks_per_partial =
                    (total_attn_blocks + inst.num_partials - 1) /
                    inst.num_partials;
                int start_blk_idx = inst.partial_idx * blocks_per_partial;
                int end_blk_idx =
                    min(start_blk_idx + blocks_per_partial, total_attn_blocks);
                float softmax_temp =
                    g.attn_scale * 1.44269504089f; // 1 / (sqrt(D_h) * ln(2))
                q_rt Q_reg;
                k_rt K_reg;
                v_rt V_reg;
                l_rv L_reg;
                o_rt O_reg;
                attn_fl_rt attn_fl_reg;
                attn_bf_rt attn_bf_reg;
                max_vec_rv max_vec_reg;
                max_vec_rv scaled_max_vec_reg;
                max_vec_rv last_scaled_max_vec_reg;
                max_vec_rv diff_scaled_max_vec_reg;
                norm_vec_rv norm_vec_reg;
                kittens::neg_infty(max_vec_reg);
                kittens::zero(last_scaled_max_vec_reg); // just not +-inf
                kittens::zero(norm_vec_reg);
                kittens::zero(O_reg);
                o_sv(&O_smem)[4] = get_O_smem(s);
                l_sv &L_smem = get_L_smem(s);

                // Initiate the load on Q
                wait_QOL_page(s);

                q_st &Q_smem = get_Q_smem(s);

                load_Q_async(Q_smem, g.q_post_rope, q_head_start_idx);

                // Wait for Q to arrive
                // AMD: Explicitly wait for global (vmcnt) and shared (lgkmcnt) loads to complete
                asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)");

                kittens::load(Q_reg, Q_smem);

                // Run the pipeline!
                for (int i = 0; i + start_blk_idx < end_blk_idx; ++i) {
                    int stage = i % NUM_STAGES;
                    kv_st &K_smem = get_K_smem(s, stage);
                    kv_st &V_smem = get_V_smem(s, stage);
                    // Calculate expected semaphore count (monotonic)
                    // Assuming producer increments once per tile usage.
                    int wait_target = (i / NUM_STAGES) + 1;

                    // Perform Q @ K.T
                    kittens::zero(attn_fl_reg);
                    
                    // Wait for K producer
                    K_arrived(s, stage).wait(wait_target);

                    kittens::load(K_reg, K_smem);
                    // enforces column-major layout. attn_fl_rt
                    kittens::mma_ABt(attn_fl_reg, Q_reg, K_reg, attn_fl_reg);
                    
                    // AMD: Wave barrier to ensure instruction issue order/visibility before signaling
                    __builtin_amdgcn_wave_barrier();
                    
                    // Signal K finished to producer
                    K_finished(s, stage).arrive();

                    // Mask out invalid positions at the end
                    if ((i + start_blk_idx + 1) * LLAMA_1B_KV_BLOCK_SIZE >
                        seq_len)
                        right_fill(attn_fl_reg, attn_fl_reg,
                                    seq_len % LLAMA_1B_KV_BLOCK_SIZE,
                                    -999999999999.f);

                    // Obtain maximums per row (which is per head)
                    kittens::row_max(max_vec_reg, attn_fl_reg,
                                    max_vec_reg); // includes previous max

                    // Scale attention block and maximums by sqrt(D_h)
                    kittens::mul(attn_fl_reg, attn_fl_reg, softmax_temp);
                    kittens::mul(scaled_max_vec_reg, max_vec_reg, softmax_temp);

                    // Calculate softmax numerator
                    kittens::sub_row(attn_fl_reg, attn_fl_reg, scaled_max_vec_reg);
                    kittens::exp2(attn_fl_reg, attn_fl_reg);

                    // Calculate softmax denominator
                    kittens::sub(diff_scaled_max_vec_reg, last_scaled_max_vec_reg,
                                scaled_max_vec_reg);
                    kittens::exp2(diff_scaled_max_vec_reg,
                                diff_scaled_max_vec_reg);

                    // Normalize and accumulate numerator (A @ V)
                    kittens::mul_row(O_reg, O_reg, diff_scaled_max_vec_reg);
                    
                    // Wait for V producer
                    V_arrived(s, stage).wait(wait_target);

                    kittens::load(V_reg, V_smem);
                    kittens::copy(attn_bf_reg,
                                attn_fl_reg); // Convert to bf16 to do matmul
                    kittens::mma_AB(O_reg, attn_bf_reg, V_reg, O_reg);
                    
                    __builtin_amdgcn_wave_barrier();
                    
                    // Signal V finished to producer
                    V_finished(s, stage).arrive();

                    // Normalize and accumulate demoniator
                    kittens::mul(norm_vec_reg, norm_vec_reg,
                                diff_scaled_max_vec_reg);
                    kittens::row_sum(norm_vec_reg, attn_fl_reg, norm_vec_reg);

                    // Save for next iteration
                    kittens::copy(last_scaled_max_vec_reg, scaled_max_vec_reg);
                }

                // Finish
                __builtin_amdgcn_wave_barrier();

                if (start_blk_idx < end_blk_idx) {
                    finish_KV_page(s);
                    kittens::div_row(O_reg, O_reg, norm_vec_reg);
                    kittens::log2(L_reg, norm_vec_reg);
                    kittens::add(
                        L_reg, L_reg,
                        last_scaled_max_vec_reg); // now L_reg contains the LSE
                } else {
                    // Very edgy case where no blocks are processed.
                    // Make the math work out during attention reduction!
                    kittens::neg_infty(L_reg);
                }

                // Store the results
                store_4_rows(O_smem, O_reg, q_head_local_idx);
                __builtin_amdgcn_wave_barrier();

                // Signal Output
                O_arrived(s).arrive();
                
                kittens::store(L_smem, L_reg);
                __builtin_amdgcn_wave_barrier();
                
                L_arrived(s).arrive();
            }
        }
    };
    struct storer {

        static inline __device__ void
        store_o_skip(const globals &g, megakernel::state<config> &s, int q_head_start_idx) {
            auto O_smem = get_O_smem(s);

            // Wait for Consumer to finish producing O
            if (kittens::laneid() == 0) {
                // HipKittens semaphore wait (target = 1)
                O_arrived(s).wait(1);
                s.record(megakernel::TEVENT_OUTPUT_READY);
            }
            // AMD: Ensure warp converges after scalar wait
            __builtin_amdgcn_wave_barrier();

            // Register tile for data movement (Shared -> Reg -> Global)
            kittens::rv_bf<globals::head_dim> O_bf;

            for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                auto &smem_fl = O_smem[head_offset];
                
                // 1. Load from Shared Memory (Float) into Register (BF16)
                // Note: kittens::load typically handles layout/type conversion
                kittens::load(O_bf, smem_fl);
                
                // AMD: Wave barrier to ensure registers are ready before store
                __builtin_amdgcn_wave_barrier();

                // 2. Store from Register (BF16) directly to Global Memory
                // Replacing TMA store with manual register-based store
                kittens::store(g.attn_out, O_bf, {q_head_start_idx + head_offset});
            }

            // AMD: Explicitly wait for global memory stores (vmcnt) to complete
            __builtin_amdgcn_s_waitcnt(0);
        }

        static inline __device__ void
        store_o_no_skip(const globals &g, megakernel::state<config> &s,
                        int q_head_start_idx, parsed_instruction &inst) {
            
            auto O_smem = get_O_smem(s);

            if (kittens::laneid() == 0) {
                O_arrived(s).wait(1);
                s.record(megakernel::TEVENT_OUTPUT_READY);
            }
            __builtin_amdgcn_wave_barrier();

            // Temporary register tile for intermediate storage
            // Assuming head_dim matches the partial output layout
            kittens::rv<globals::head_dim> O_reg; 

            for (int head_offset = 0; head_offset < GQA_RATIO; head_offset++) {
                // Load Shared -> Register
                kittens::load(O_reg, O_smem[head_offset]);

                // Store Register -> Global
                kittens::store(g.attn_out_intermediates, O_reg,
                    {0, q_head_start_idx + head_offset, inst.partial_idx, 0});
            }
            
            // Wait for stores
            __builtin_amdgcn_s_waitcnt(0);
        }

        static __device__ void run(const globals &g, megakernel::state<config> &s) {
            parsed_instruction inst{s};
            int laneid = kittens::laneid(); // Use standard kittens laneid
            int q_head_start_idx = inst.kv_head_idx * GQA_RATIO; 
            int q_head_vec_start_idx = q_head_start_idx % 16;

            auto skip_attn_reduction = g.skip_attn_reduction;

            if (skip_attn_reduction) {
                store_o_skip(g, s, q_head_start_idx);
            } else {
                store_o_no_skip(g, s, q_head_start_idx, inst);
            }

            // Store LSE to global memory
            if (laneid < GQA_RATIO && !skip_attn_reduction) {
                l_sv &L_smem = get_L_smem(s);
                
                // Wait for LSE production
                L_arrived(s).wait(1);

                // Manual copy Shared -> Global
                // AMD: No inline PTX needed. Standard C++ pointer access compiles to 
                // efficient ds_read_b32 and global_store_dword instructions.
                float tmp = L_smem.data[q_head_vec_start_idx + laneid];
                
                float *dst_ptr = (float *)&g.attn_lse_intermediates
                                    .raw_ptr[(q_head_start_idx + laneid) *
                                                g.attn_lse_intermediates.cols() +
                                            inst.partial_idx];
                
                *dst_ptr = tmp;
            }

            // Ensure all global writes (including LSE) are visible
            __threadfence_block();
            __builtin_amdgcn_s_waitcnt(0); 

            if (laneid == 0) {
                // 123 is arbitrary debug ID from original code
                s.record(123 + laneid);
                finish_QOL_page(s);
            }

            // Signal completion to global barrier (for reduction kernel)
            if (laneid < GQA_RATIO) {

                if (laneid == 0) {
                    s.record(megakernel::TEVENT_AT_GMEM_STORE);
                }

                if (skip_attn_reduction) {
                    atomicAdd(&g.Bar[{inst.layer_idx,
                                    OPCODE_AttentionReduction - 1, 0}],
                            1);
                } else {
                    // Assuming OPCODE_RMS_QKV_MatVecRopeAppend based on consumer context
                    atomicAdd(&g.Bar[{inst.layer_idx, 
                                    OPCODE_RMS_QKV_MatVecRopeAppend - 1,
                                    q_head_start_idx + laneid}],
                            1);
                }

                if (laneid == 0) {
                    s.record(megakernel::TEVENT_DONE_GMEM_STORE);
                }
            }
        }
    };
};