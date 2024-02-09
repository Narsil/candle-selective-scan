#include "causal_conv1d_fwd.cu"
#include "causal_conv1d_update.cu"

extern "C" void causal_conv1d_ffi(
    ConvParamsBase &params,
    uint32_t input_dtype,
    uint32_t weight_dtype,
    bool channels_last,
    cudaStream_t stream
    ){
        printf("Not update\n");
        if (input_dtype == 2 && weight_dtype == 2 && !channels_last){
        printf("Correct launch\n");
            causal_conv1d_fwd_cuda<float, float>(params, stream);
        }else if (input_dtype == 2 && weight_dtype == 2 && channels_last){
            causal_conv1d_channellast_fwd_cuda<float, float>(params, stream);
        }else{
            exit(1);
        }
}

extern "C" void causal_conv1d_update_ffi(
    ConvParamsBase &params,
    uint32_t input_dtype,
    uint32_t weight_dtype,
    cudaStream_t stream
    ){
        printf("UPdate");
        if (input_dtype == 2 && weight_dtype == 2){
            causal_conv1d_update_cuda<float, float>(params, stream);
        }else{
            exit(1);
        }
}
