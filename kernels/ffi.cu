
#include "selective_scan_fwd_kernel.cuh"
using index_t = uint32_t;
void selective_scan_fwd_cuda_ffi(
    int batch, int dim, int seqlen, int dstate, int n_groups, int n_chunks,
    int dim_ngroups_ratio,
    bool is_variable_B,
    bool is_variable_C,

    bool delta_softplus,

    index_t A_d_stride,
    index_t A_dstate_stride,
    index_t B_batch_stride,
    index_t B_d_stride,
    index_t B_dstate_stride,
    index_t B_group_stride,
    index_t C_batch_stride,
    index_t C_d_stride,
    index_t C_dstate_stride,
    index_t C_group_stride,
    index_t u_batch_stride,
    index_t u_d_stride,
    index_t delta_batch_stride,
    index_t delta_d_stride,
    index_t z_batch_stride,
    index_t z_d_stride,
    index_t out_batch_stride,
    index_t out_d_stride,
    index_t out_z_batch_stride,
    index_t out_z_d_stride,

    // Common data pointers.
    void *__restrict__ A_ptr,
    void *__restrict__ B_ptr,
    void *__restrict__ C_ptr,
    void *__restrict__ D_ptr,
    void *__restrict__ u_ptr,
    void *__restrict__ delta_ptr,
    void *__restrict__ delta_bias_ptr,
    void *__restrict__ out_ptr,
    void *__restrict__ x_ptr,
    void *__restrict__ z_ptr,
    void *__restrict__ out_z_ptr,
    uint32_t input_dtype,
    uint32_t weight_dtype,
    cudaStream_t stream
    ){
        SSMParamsBase params {
            .batch = batch,
            .dim = dim,
            .seqlen = seqlen,
            .dstate = dstate,
            .n_groups = n_groups,
            .n_chunks = n_chunks,
            .dim_ngroups_ratio = dim_ngroups_ratio,
            .is_variable_B = is_variable_B,
            .is_variable_C = is_variable_C,
            .delta_softplus = delta_softplus,
            .A_d_stride = A_d_stride,
            .B_batch_stride = B_batch_stride,
            .B_d_stride = B_d_stride,
            .B_dstate_stride = B_dstate_stride,
            .B_group_stride = B_group_stride,
            .C_batch_stride = C_batch_stride,
            .C_d_stride = C_d_stride,
            .C_dstate_stride = C_dstate_stride,
            .C_group_stride = C_group_stride,
            .u_batch_stride = u_batch_stride,
            .u_d_stride = u_d_stride,
            .delta_batch_stride = delta_batch_stride,
            .delta_d_stride = delta_d_stride,
            .z_batch_stride = z_batch_stride,
            .z_d_stride = z_d_stride,
            .out_batch_stride = out_batch_stride,
            .out_d_stride = out_d_stride,
            .out_z_batch_stride = out_z_batch_stride,
            .out_z_d_stride = out_z_d_stride,
        };
        if (input_dtype == 2 && weight_dtype == 2){
            selective_scan_fwd_cuda<float, float>(params, stream);
        }else{
            std::cerr<<"Invalid dtypes";
            exit(1);
        }
}
