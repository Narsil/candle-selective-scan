use candle::{bail, Device, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

mod cpu;
mod utils;

use utils::{check_same_device, SameDevice};

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
    check_same_device!(u, delta, a, b, c, d, z, delta_bias);

    match u.device() {
        Device::Cpu => {
            cpu::apply_selective_scan(u, delta, a, b, c, d, z, delta_bias, delta_softplus)
        }
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::apply_selective_scan(u, delta, a, b, c, d, z, delta_bias, delta_softplus)
        }
        _ => cpu::apply_selective_scan(u, delta, a, b, c, d, z, delta_bias, delta_softplus),
    }
}

pub fn apply_selective_scan_update(
    state: &mut Tensor,
    u: &Tensor,
    delta: &Tensor,
    a: &Tensor,
    b: &Tensor,
    c: &Tensor,
    d: Option<&Tensor>,
    z: Option<&Tensor>,
    delta_bias: Option<&Tensor>,
    delta_softplus: bool,
) -> Result<(Tensor, Tensor)> {
    check_same_device!(state, u, delta, a, b, c, d, z, delta_bias);

    match u.device() {
        _ => cpu::apply_selective_scan_update(
            state,
            u,
            delta,
            a,
            b,
            c,
            d,
            z,
            delta_bias,
            delta_softplus,
        ),
    }
}

pub fn apply_causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    check_same_device!(x, weight, bias, seq_idx);

    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => cuda::apply_causal_conv1d(x, weight, bias, seq_idx, activation),
        _ => cpu::apply_causal_conv1d(x, weight, bias, seq_idx, activation),
    }
}

pub fn apply_causal_conv1d_update(
    x: &Tensor,
    conv_state: &mut Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    check_same_device!(x, conv_state, weight, bias, seq_idx);

    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::apply_causal_conv1d_update(x, conv_state, weight, bias, seq_idx, activation)
        }
        _ => cpu::apply_causal_conv1d_update(x, conv_state, weight, bias, seq_idx, activation),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle::DType;

    macro_rules! assert_close {
        ($left:ident, $right:ident, rtol=$rtol:expr, $($arg:tt)*) => {{
            assert_eq!($left.shape(), $right.shape(), $($arg),*);
            let diff = (($left - &$right)?.abs().unwrap() / $right.abs().unwrap()).unwrap();
            let tol = (diff.ge($rtol))
                .unwrap()
                .to_dtype(DType::F32)?
                .sum_all()
                .unwrap();
            let total = tol.to_scalar::<f32>()?;
            assert_eq!(total, 0.0, $($arg),*);
        }};
    }

    #[test]
    fn test_selective_scan_original() -> Result<()> {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0)?;

        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;

        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests");
        for filename in std::fs::read_dir(dir).unwrap() {
            let path = filename?.path();
            if path.display().to_string().ends_with(".safetensors") {
                println!("Test {}", path.display());
                let weights = candle::safetensors::load(path.clone(), &device)?;
                let file = std::fs::File::open(path.clone()).unwrap();
                let buffer = unsafe { memmap2::MmapOptions::new().map(&file).unwrap() };
                let (_, metadata) = safetensors::SafeTensors::read_metadata(&buffer).unwrap();
                let metadata = metadata.metadata();

                let u = weights.get("u").unwrap();
                let delta = weights.get("delta").unwrap();
                let a = weights.get("A").unwrap();
                let b = weights.get("B").unwrap();
                let c = weights.get("C").unwrap();
                let d = weights.get("D");
                let z = weights.get("z");
                let delta_bias = weights.get("delta_bias");
                let delta_softplus = metadata.as_ref().map(|m| m.get("delta_softplus"))
                    == Some(Some(&"True".to_string()));
                let out = weights.get("out").unwrap();
                let (out2, _) =
                    apply_selective_scan(&u, &delta, &a, &b, &c, d, z, delta_bias, delta_softplus)?;

                assert_close!(out, out2, rtol = 1e-2,);
            }
        }
        Ok(())
    }

    #[test]
    fn test_causal_conv1d_small() -> Result<()> {
        #[cfg(feature = "cuda")]
        let device = Device::new_cuda(0)?;

        #[cfg(not(feature = "cuda"))]
        let device = Device::Cpu;

        let dim = 3;
        let seqlen = 8;
        let width = 2;
        let silu_activation = false;
        let _dtype = DType::F32;
        let _channel_last = false;
        let batch = 2;

        let total = batch * dim * seqlen;
        let x = Tensor::arange(0f32, total as f32, &device)?.reshape((batch, dim, seqlen))?;

        let total = dim * width;
        let weight = Tensor::arange(0f32, total as f32, &device)?.reshape((dim, width))?;
        let out = apply_causal_conv1d(&x, &weight, None, None, silu_activation)?;

        assert_eq!(out.dims(), [batch, dim, seqlen]);

        assert_eq!(
            out.to_vec3::<f32>()?,
            [
                [
                    [0., 1., 2., 3., 4., 5., 6., 7.],
                    [24., 43., 48., 53., 58., 63., 68., 73.],
                    [80., 149., 158., 167., 176., 185., 194., 203.]
                ],
                [
                    [24., 25., 26., 27., 28., 29., 30., 31.],
                    [96., 163., 168., 173., 178., 183., 188., 193.],
                    [200., 365., 374., 383., 392., 401., 410., 419.]
                ]
            ]
        );

        Ok(())
    }
}
