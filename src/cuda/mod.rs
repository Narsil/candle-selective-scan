use candle::cuda_backend::cudarc::driver::DevicePtr;
use candle::{bail, DType, Device, Result, Storage, Tensor};
mod ffi;
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
    let b = if b.rank() == 3 {
        b.unsqueeze(1)?
    } else {
        b.clone()
    };
    let c = if c.rank() == 3 {
        c.unsqueeze(1)?
    } else {
        c.clone()
    };
    unsafe {
        let (batch, dim, seqlen) = u.dims3()?;
        let dstate = a.dim(1)?;
        let is_variable_b = b.rank() >= 3;
        let is_variable_c = c.rank() >= 3;
        let n_groups = if is_variable_b { b.dim(1)? } else { 1 };

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

        let params = ffi::SSMParamsBase {
            batch: batch as i32,
            dim: dim as i32,
            seqlen: seqlen as i32,
            dstate: dstate as i32,
            n_groups: n_groups as i32,
            n_chunks: n_chunks as i32,
            dim_ngroups_ratio: dim_ngroups_ratio as i32,
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
        };

        ffi::selective_scan_fwd_cuda_ffi(&params, input_dtype, weight_dtype, stream);
        if let Some(out_z) = out_z {
            Ok((out_z.clone(), Some(out_z.clone())))
        } else {
            Ok((out, out_z))
        }
    }
}

pub fn apply_causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
)-> Result<Tensor>{
    apply_causal_conv1d_(x, None, weight, bias, seq_idx, activation)
}

pub fn apply_causal_conv1d_update(
    x: &Tensor,
    conv_state: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
)-> Result<Tensor>{
    apply_causal_conv1d_(x, Some(conv_state), weight, bias, seq_idx, activation)
}

fn apply_causal_conv1d_(
    x: &Tensor,
    conv_state: Option<&Tensor>,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    let batch = x.dim(0)?;
    let dim = x.dim(1)?;
    let seqlen = x.dim(2)?;
    let width = weight.dim(1)?;
    assert_eq!(x.dims(), [batch, dim, seqlen]);
    assert_eq!(weight.dims(), [dim, width]);
    assert_eq!(weight.dims(), [dim, width]);

    let is_channel_last = if x.stride()[1] == 1{
        assert_eq!(x.stride()[1], 1);
        true
    }else{
        assert_eq!(x.stride()[2], 1);
        false
    };
    if is_channel_last {
       assert_eq!(dim % 8,  0, "causal_conv1d only supports channel dimension divisible by 8 for now");
    }

  assert!(width >= 2 && width <= 4, "causal_conv1d only supports width between 2 and 4");

  if let Some(seq_idx) = &seq_idx{
      assert!(is_channel_last, "seq_idx only supported for channel last layout");
      assert_eq!(seq_idx.dtype(), DType::U32);
      assert!(seq_idx.is_contiguous());
      assert_eq!(seq_idx.dims(), [batch, seqlen]);
  }

  if let Some(bias) = &bias{
      assert_eq!(bias.dtype(), x.dtype());
      assert!(bias.is_contiguous());
      assert_eq!(bias.dims(), [dim]);
  }

  let out = x.zeros_like()?;


  let xs = x.stride();
  let (x_batch_stride, x_c_stride, x_l_stride) = (xs[0] as u32, xs[1] as u32, xs[2] as u32);

  let ws = x.stride();
  let (weight_c_stride, weight_width_stride) = (ws[0] as u32, ws[1] as u32);

  let outs = out.stride();
  let (out_batch_stride, out_c_stride, out_l_stride) = (outs[0] as u32, outs[1] as u32, outs[2] as u32);

  let (conv_state_batch_stride, conv_state_c_stride, conv_state_l_stride) = if let Some(conv_state) = &conv_state{
      let conv_states = conv_state.stride();
      (conv_states[0] as u32, conv_states[1] as u32, conv_states[2] as u32)
  }else{
      (0, 0, 0)
  };

  unsafe{
  let conv_state_ptr = conv_state.device_ptr()?;
  let x_ptr = x.device_ptr()?;
  let weight_ptr = weight.device_ptr()?;
  let bias_ptr = bias.device_ptr()?;
  let out_ptr = out.device_ptr()?;
  let seq_idx_ptr = seq_idx.device_ptr()?;
  let params = ffi::ConvParamsBase{
      batch: batch as i32,
      dim: dim as i32,
      seqlen: seqlen as i32,
      width: width as i32,
      silu_activation: activation,

      x_batch_stride,
      x_c_stride,
      x_l_stride,

      weight_c_stride,
      weight_width_stride,

      out_batch_stride,
      out_c_stride,
      out_l_stride,

      conv_state_batch_stride,
      conv_state_c_stride,
      conv_state_l_stride,

      x_ptr,
      weight_ptr,
      bias_ptr,
      out_ptr,
      conv_state_ptr,
      seq_idx_ptr,



  };
    let input_dtype = dtype_as_u32(x.dtype())?;
    let weight_dtype = dtype_as_u32(weight.dtype())?;

    let device = x.device();
    let device = match device {
        Device::Cuda(cuda_device) => cuda_device,
        dev => candle::bail!("Invalid device {dev:?}"),
    };
    let stream = *device.cu_stream() as *const c_void;
    ffi::causal_conv1d_ffi(&params, input_dtype, weight_dtype, is_channel_last, stream);
  }
    Ok(out)
}
