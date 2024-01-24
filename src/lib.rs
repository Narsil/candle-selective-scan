mod ffi;
use candle::{Tensor, Result};


pub fn apply_selective_scan(
    u: &Tensor, delta: &Tensor, a: &Tensor, b: &Tensor, c: &Tensor, d: Option<&Tensor>, z: Option<&Tensor>, delta_bias: Option<&Tensor>, delta_softplus: bool) -> Result<()>{

    let (batch, dim, seqlen) = u.dims3()?;
    let dstate = a.dims()[1];
    let is_variable_B = B.rank() >= 3;
    let is_variable_C = C.rank() >= 3;
    let n_groups = if is_variable_B{b.dims()[1]} else {1};

    let n_chunks = (seqlen + 2048 - 1) / 2048;

    let out = delta.zeros_like();
    let x = Tensor::zeros((batch, dim, n_chunks, dstate * 2), a.dtype(), a.device())?;


    ffi::selective_scan_fwd_cuda_ffi(
        batch as i32,
        dim as i32,
        seqlen as i32,
        dstate as i32,
        n_groups as i32,
        n_chunks as i32,
        dim_ngroups_ratio as i32,
        is_variable_B,
        is_variable_C,
        delta_softplus,
        A_d_stride,
        A_dstate_stride,
        B_batch_stride,
        B_d_stride,
        B_dstate_stride,
        B_group_stride,
        C_batch_stride,
        C_d_stride,
        C_dstate_stride,
        C_group_stride,
        u_batch_stride,
        u_d_stride,
        delta_batch_stride,
        delta_d_stride,
        z_batch_stride,
        z_d_stride,
        out_batch_stride,
        out_d_stride,
        out_z_batch_stride,
        out_z_d_stride,

        // Common data pointers.
        A_ptr,
        B_ptr,
        C_ptr,
        D_ptr,
        u_ptr,
        delta_ptr,
        delta_bias_ptr,
        out_ptr,
        x_ptr,
        z_ptr,
        out_z_ptr,
        input_dtype,
        weight_dtype,
        stream,

    );
    Ok(())
}


