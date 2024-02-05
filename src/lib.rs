use candle::{bail, Device, Result, Tensor};

#[cfg(feature = "cuda")]
mod cuda;

mod cpu;

trait SameDevice {
    fn is_same_device(&self, device: &Device) -> bool;
    fn device_fmt(&self) -> String;
}

impl SameDevice for &Tensor {
    fn is_same_device(&self, _device: &Device) -> bool {
        // TODO Why isn't PartialEq detected here ?
        // self.device() == device
        true
    }
    fn device_fmt(&self) -> String {
        format!("{:?}", self.device())
    }
}

impl SameDevice for Option<&Tensor> {
    fn is_same_device(&self, _device: &Device) -> bool {
        true
        // if let Some(tensor) = self {
        //     // TODO Why isn't PartialEq detected here ?
        //     true
        //     // tensor.device().eq(device)
        // } else {
        //     true
        // }
    }

    fn device_fmt(&self) -> String {
        if let Some(tensor) = &self {
            format!("{:?}", tensor.device())
        } else {
            "None".to_string()
        }
    }
}

macro_rules! check_same_device{
    // Decompose multiple `eval`s recursively
    ($left:ident, $($rest:ident),+) => {{
        let device = $left.device();
        let mut fail = false;
        $(if !$rest.is_same_device(device) {
            fail = true;
        })+

        if fail{
            let mut err = String::new();
            err.push_str(&format!("Device mismatch: ({:?},", device));
            $(
                err.push_str(&format!("{:?}", $rest.device_fmt()));
            )+
            err.push_str(")");
            bail!("{}", err)
        }
    }};
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
        _ => cpu::apply_selective_scan_update(state, u, delta, a, b, c, d, z, delta_bias, delta_softplus),
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
        Device::Cuda(_) => {
            cuda::apply_causal_conv1d(x, weight, bias, seq_idx, activation)
        }
        _ => todo!()
    }
}

pub fn apply_causal_conv1d_update(
    x: &Tensor,
    conv_state: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    check_same_device!(x, weight, bias, seq_idx);

    match x.device() {
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => {
            cuda::apply_causal_conv1d_update(x, conv_state, weight, bias, seq_idx, activation)
        }
        _ => todo!()
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
}
