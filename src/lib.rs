mod ffi;
use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{bail, DType, Device, Result, Storage, Tensor};
use core::ffi::c_void;

trait IDevicePtr {
    unsafe fn device_ptr(&self) -> Result<*const c_void>;
}

impl IDevicePtr for Option<Tensor> {
    unsafe fn device_ptr(&self) -> Result<*const c_void> {
        if let Some(t) = self {
            t.device_ptr()
        } else {
            Ok(std::ptr::null())
        }
    }
}

impl IDevicePtr for Option<&Tensor> {
    unsafe fn device_ptr(&self) -> Result<*const c_void> {
        if let Some(t) = self {
            t.device_ptr()
        } else {
            Ok(std::ptr::null())
        }
    }
}

impl IDevicePtr for Tensor {
    unsafe fn device_ptr(&self) -> Result<*const c_void> {
        let (storage, layout) = self.storage_and_layout();
        let slice = match &*storage {
            Storage::Cuda(q) => q,
            _ => bail!("must be a cuda tensor"),
        };
        let slice = slice.as_cuda_slice::<f32>().unwrap();
        let slice = slice.slice(layout.start_offset()..);
        Ok(*slice.device_ptr() as *const c_void)
    }
}

fn dtype_as_u32(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F16 => Ok(0),
        DType::BF16 => Ok(1),
        DType::F32 => Ok(2),
        dtype => candle::bail!("Unsupported dtype {dtype:?}"),
    }
}

pub fn apply_selective_scan(
    u: &Tensor,
    delta: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    delta_bias: Option<&Tensor>,
    delta_softplus: bool,
) -> Result<(Tensor, Option<Tensor>)> {
    unsafe {
        let (batch, dim, seqlen) = u.dims3()?;
        let dstate = a.dims()[1];
        let is_variable_b = b.rank() >= 3;
        let is_variable_c = c.rank() >= 3;
        let n_groups = if is_variable_b { b.dims()[1] } else { 1 };

        let n_chunks = (seqlen + 2048 - 1) / 2048;

        let out = delta.zeros_like()?;
        let x = Tensor::zeros((batch, dim, n_chunks, dstate * 2), a.dtype(), a.device())?;
        let (out_z, z_batch_stride, z_d_stride, out_z_batch_stride, out_z_d_stride) =
            if let Some(z) = z {
                let out_z = z.zeros_like()?;
                let z_strides = z.stride();
                let out_z_strides = out_z.stride();
                let (a, b) = (out_z_strides[0] as u32, out_z_strides[1] as u32);
                (Some(out_z), z_strides[0] as u32, z_strides[1] as u32, a, b)
            } else {
                (None, 0, 0, 0, 0)
            };

        let a_ptr = a.device_ptr()?;
        let b_ptr = b.device_ptr()?;
        let c_ptr = c.device_ptr()?;
        let d_ptr = d.device_ptr()?;
        let u_ptr = u.device_ptr()?;
        let delta_ptr = delta.device_ptr()?;
        let delta_bias_ptr = delta_bias.device_ptr()?;
        let out_ptr = out.device_ptr()?;
        let x_ptr = x.device_ptr()?;
        let z_ptr = z.device_ptr()?;
        let out_z_ptr = out_z.device_ptr()?;


        let dim_ngroups_ratio = dim / n_groups;

        let a_strides = a.stride();
        let (a_d_stride, a_dstate_stride) = (a_strides[0] as u32, a_strides[1] as u32);

        let b_strides = b.stride();
        let mut b_batch_stride = 0u32;
        let mut b_group_stride = 0u32;
        let mut b_d_stride = 0u32;
        let b_dstate_stride;
        if is_variable_b {
            b_batch_stride = b_strides[0] as u32;
            b_group_stride = b_strides[1] as u32;
            b_dstate_stride = b_strides[2] as u32;
        } else {
            b_d_stride = b_strides[0] as u32;
            b_dstate_stride = b_strides[1] as u32;
        }

        let c_strides = c.stride();
        let mut c_batch_stride = 0u32;
        let mut c_group_stride = 0u32;
        let mut c_d_stride = 0u32;
        let c_dstate_stride;
        if is_variable_c {
            c_batch_stride = c_strides[0] as u32;
            c_group_stride = c_strides[1] as u32;
            c_dstate_stride = c_strides[2] as u32;
        } else {
            c_d_stride = c_strides[0] as u32;
            c_dstate_stride = c_strides[1] as u32;
        }
        let u_strides = u.stride();
        let u_batch_stride = u_strides[0] as u32;
        let u_d_stride = u_strides[1] as u32;

        let delta_strides = delta.stride();
        let delta_batch_stride = delta_strides[0] as u32;
        let delta_d_stride = delta_strides[1] as u32;

        let out_strides = out.stride();
        let out_batch_stride = out_strides[0] as u32;
        let out_d_stride = out_strides[1] as u32;

        let input_dtype = dtype_as_u32(u.dtype())?;
        let weight_dtype = dtype_as_u32(a.dtype())?;

        let device = u.device();
        let device = match device {
            Device::Cuda(cuda_device) => cuda_device,
            dev => candle::bail!("Invalid device {dev:?}"),
        };
        let stream = *device.cu_stream() as *const c_void;

        println!("{a_ptr:?} {b_ptr:?} {c_ptr:?} {u_ptr:?} {delta_ptr:?} {delta_bias_ptr:?} {out_ptr:?} {out_z_ptr:?}");

        ffi::selective_scan_fwd_cuda_ffi(
            batch as i32,
            dim as i32,
            seqlen as i32,
            dstate as i32,
            n_groups as i32,
            n_chunks as i32,
            dim_ngroups_ratio as i32,
            is_variable_b,
            is_variable_c,
            delta_softplus,
            a_d_stride,
            a_dstate_stride,
            b_batch_stride,
            b_d_stride,
            b_dstate_stride,
            b_group_stride,
            c_batch_stride,
            c_d_stride,
            c_dstate_stride,
            c_group_stride,
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
            a_ptr,
            b_ptr,
            c_ptr,
            d_ptr,
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
        Ok((out, out_z))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::{Device, IndexOp};
    fn selective_scan(
        u: &Tensor,
        delta: &Tensor,
        a: &Tensor,
        b: &Tensor,
        c: &Tensor,
        d: &Tensor,
    ) -> Result<Tensor> {
        let (b_sz, l, d_in) = u.dims3()?;
        let n = a.dim(1)?;
        let delta = delta.t()?.reshape((b_sz, d_in, l, 1))?; // b d_in l 1
        let delta_a = delta.broadcast_mul(&a.reshape((1, d_in, 1, n))?)?.exp()?;
        let delta_b_u = delta
            .broadcast_mul(&b.reshape((b_sz, 1, l, n))?)?
            .broadcast_mul(&u.t()?.reshape((b_sz, d_in, l, 1))?)?;
        let mut xs = Tensor::zeros((b_sz, d_in, n), delta_a.dtype(), delta_a.device())?;
        let mut ys = Vec::with_capacity(l);
        for i in 0..l {
            xs = ((delta_a.i((.., .., i))? * xs)? + delta_b_u.i((.., .., i))?)?;
            let y = xs.matmul(&c.i((.., i, ..))?.unsqueeze(2)?)?.squeeze(2)?;
            ys.push(y)
        }
        let ys = Tensor::stack(ys.as_slice(), 1)?;
        ys + u.broadcast_mul(d)
    }
    #[test]
    fn test_selective_scan() {
        let seqlen = 5;
        let batch = 1;
        let hidden_dim = 1536;
        let x = 16;
        let device = Device::new_cuda(0).unwrap();
        // Tensor[dims 1, 5, 1536; f32] Tensor[dims 1, 5, 1536; f32] Tensor[dims 1536, 16; f32] Tensor[dims 1, 5, 16; f32] Tensor[dims 1, 5, 16; f32] Tensor[dims 1536; f32]
        let u = Tensor::randn(0.0f32, 1.0f32, (batch, seqlen, hidden_dim), &device).unwrap();
        let delta = Tensor::randn(0.0f32, 1.0f32, (batch, seqlen, hidden_dim), &device).unwrap();
        let a = Tensor::randn(0.0f32, 1.0f32, (hidden_dim, x), &device).unwrap();
        let b = Tensor::randn(0.0f32, 1.0f32, (batch, seqlen, x), &device).unwrap();
        let c = Tensor::randn(0.0f32, 1.0f32, (batch, seqlen, x), &device).unwrap();
        let d = Tensor::randn(0.0f32, 1.0f32, (hidden_dim,), &device).unwrap();

        let z = selective_scan(&u, &delta, &a, &b, &c, &d).unwrap();
        let (z2, _) = apply_selective_scan(&u, &delta, &a, &b, &c, Some(&d), None, None, false).unwrap();

        assert_eq!(z.dims(), &[batch, seqlen, hidden_dim]);
        assert_eq!(z2.dims(), &[batch, seqlen, hidden_dim]);
        assert_eq!(z.flatten_all().unwrap().to_vec1::<f32>().unwrap(), z2.flatten_all().unwrap().to_vec1::<f32>().unwrap());
    }
}
