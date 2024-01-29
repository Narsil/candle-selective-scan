    use candle_selective_scan::*;
    use candle::{IndexOp, Device, DType, Result, Tensor};
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
fn main(){
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
        let d = Tensor::randn(0.0f32, 1.0f32, (hidden_dim, ), &device).unwrap();

        let z = selective_scan(&u, &delta, &a, &b, &c, &d).unwrap();
        let z2 = Tensor::zeros((batch, seqlen, hidden_dim), DType::F32, &device).unwrap();
        apply_selective_scan(&u, &delta, &a, &b, &c, Some(&d), Some(&z), None, false).unwrap();
        assert_eq!(z.dims(), &[batch, seqlen, hidden_dim]);
}
