use core::ffi::{c_int, c_void};

#[repr(C)]
#[derive(Debug)]
pub struct SSMParamsBase {
    pub batch: c_int,
    pub dim: c_int,
    pub seqlen: c_int,
    pub dstate: c_int,
    pub n_groups: c_int,
    pub n_chunks: c_int,
    pub dim_ngroups_ratio: c_int,
    pub is_variable_b: bool,
    pub is_variable_c: bool,
    pub delta_softplus: bool,

    pub a_d_stride: u32,
    pub a_dstate_stride: u32,
    pub b_batch_stride: u32,
    pub b_d_stride: u32,
    pub b_dstate_stride: u32,
    pub b_group_stride: u32,
    pub c_batch_stride: u32,
    pub c_d_stride: u32,
    pub c_dstate_stride: u32,
    pub c_group_stride: u32,
    pub u_batch_stride: u32,
    pub u_d_stride: u32,
    pub delta_batch_stride: u32,
    pub delta_d_stride: u32,
    pub z_batch_stride: u32,
    pub z_d_stride: u32,
    pub out_batch_stride: u32,
    pub out_d_stride: u32,
    pub out_z_batch_stride: u32,
    pub out_z_d_stride: u32,

    // Common data pointers.
    pub a_ptr: *const c_void,
    pub b_ptr: *const c_void,
    pub c_ptr: *const c_void,
    pub d_ptr: *const c_void,
    pub u_ptr: *const c_void,
    pub delta_ptr: *const c_void,
    pub delta_bias_ptr: *const c_void,
    pub out_ptr: *const c_void,
    pub x_ptr: *const c_void,
    pub z_ptr: *const c_void,
    pub out_z_ptr: *const c_void,
}

#[repr(C)]
#[derive(Debug)]
pub struct ConvParamsBase {
    pub batch: c_int,
    pub dim: c_int,
    pub seqlen: c_int,
    pub width: c_int,
    pub silu_activation: bool,

    pub x_batch_stride: u32,
    pub x_c_stride: u32,
    pub x_l_stride: u32,
    pub weight_c_stride: u32,
    pub weight_width_stride: u32,
    pub out_batch_stride: u32,
    pub out_c_stride: u32,
    pub out_l_stride: u32,

    pub conv_state_batch_stride: u32,
    pub conv_state_c_stride: u32,
    pub conv_state_l_stride: u32,

    pub x_ptr: *const c_void,
    pub weight_ptr: *const c_void,
    pub bias_ptr: *const c_void,
    pub out_ptr: *const c_void,
    pub conv_state_ptr: *const c_void,
    pub seq_idx_ptr: *const c_void,
}

extern "C" {
    pub(crate) fn selective_scan_fwd_cuda_ffi(
        params: &SSMParamsBase,
        input_dtype: u32,
        weight_dtype: u32,
        stream: *const c_void,
    );

    pub(crate) fn selective_scan_fwd_update_cuda_ffi(
        params: &SSMParamsBase,
        input_dtype: u32,
        weight_dtype: u32,
        stream: *const c_void,
    );

    pub(crate) fn causal_conv1d_ffi(
        params: &ConvParamsBase,
        input_dtype: u32,
        weight_dtype: u32,
        channel_last: bool,
        stream: *const c_void,
    );

    pub(crate) fn causal_conv1d_update_ffi(
        params: &ConvParamsBase,
        input_dtype: u32,
        weight_dtype: u32,
        channel_last: bool,
        stream: *const c_void,
    );
}
