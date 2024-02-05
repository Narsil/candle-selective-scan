#include "selective_scan_fwd_kernel.cuh"

extern "C" void selective_scan_fwd_cuda_ffi(
    SSMParamsBase &params,
    uint32_t input_dtype,
    uint32_t weight_dtype,
    cudaStream_t stream
    ){
        if (input_dtype == 2 && weight_dtype == 2){
            selective_scan_fwd_cuda<float, float>(params, stream);
        }else{
            exit(1);
        }
}
