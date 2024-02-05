#include "causal_conv1d_fwd.cu"

extern "C" void causal_conv1d_ffi(
    ConvParamsBase &params,
    uint32_t input_dtype,
    uint32_t weight_dtype,
    bool channels_last,
    cudaStream_t stream
    ){
        if (input_dtype == 2 && weight_dtype == 2 && !channels_last){
            causal_conv1d_fwd_cuda<float, float>(params, stream);
        }else if (input_dtype == 2 && weight_dtype == 2 && channels_last){
            causal_conv1d_channellast_fwd_cuda<float, float>(params, stream);
        }else{
            exit(1);
        }
}
