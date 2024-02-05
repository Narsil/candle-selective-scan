use candle::{DType, IndexOp, Result, Tensor};

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
    let device = u.device();
    let dtype_in = u.dtype();
    let u = u.to_dtype(DType::F32)?;
    let mut delta: Tensor = delta.to_dtype(DType::F32)?;
    if let Some(delta_bias) = delta_bias {
        let delta_bias = delta_bias
            .to_dtype(DType::F32)?
            .unsqueeze(0)?
            .unsqueeze(2)?;
        delta = (delta.broadcast_add(&delta_bias))?;
    }
    if delta_softplus {
        delta = ((delta.exp()? + 1.)?).log()?;
    }

    let batch = u.dim(0)?;
    let seqlen = u.dim(2)?;
    let dim = a.dim(0)?;
    let dstate = a.dim(1)?;
    let is_variable_b = b.rank() >= 3;
    let is_variable_c = c.rank() >= 3;

    let b = b.to_dtype(DType::F32)?;
    let c = c.to_dtype(DType::F32)?;

    let mut x = Tensor::zeros((batch, dim, dstate), DType::F32, device)?;

    let delta = delta.reshape((batch, dim, seqlen, 1))?;
    let delta_a = delta
        .broadcast_mul(&a.reshape((1, dim, 1, dstate))?)?
        .exp()?;
    let delta_b_u = if !is_variable_b {
        let tmp = (delta * u.clone())?;
        tmp.broadcast_mul(&b.reshape((1, dim, 1, dstate))?)?
    } else {
        let tmp = (delta.mul(&u.unsqueeze(3)?))?;
        let b = if b.rank() == 3 {
            // B is batch, dim, seqlen
            let b = b.t()?;
            let b = b.reshape((batch, 1, seqlen, dstate))?;
            assert_eq!(b.dims(), [batch, 1, seqlen, dstate]);
            b
        } else {
            let h = b.dim(1)?;
            let r = dim / h;
            // let b = b.repeat((1, r, 1, 1))?;
            // Candle's repeat is repeat interleave
            // This makes it like a regular repeat
            let b = b.unsqueeze(2)?;
            let b = b.repeat((1, 1, r, 1, 1))?;
            let b = b.reshape((batch, dim, dstate, seqlen))?;
            let b = b.t()?;
            assert_eq!(b.dims(), [batch, dim, seqlen, dstate]);
            b
        };
        tmp.broadcast_mul(&b)?
    };
    let c = if is_variable_c && c.rank() == 4 {
        let h = c.dim(1)?;
        let r = dim / h;
        // Candle's repeat is repeat interleave
        // This makes it like a regular repeat
        let c = c.unsqueeze(2)?;
        let c = c.repeat((1, 1, r, 1, 1))?;
        let c = c.reshape((batch, dim, dstate, seqlen))?;
        assert_eq!(c.dims(), [batch, dim, dstate, seqlen]);
        c
    } else {
        c
    };
    let mut ys = Vec::with_capacity(seqlen);
    for i in 0..seqlen {
        x = ((delta_a.i((.., .., i))? * x)? + delta_b_u.i((.., .., i))?)?;
        let y = if !is_variable_c {
            let right = &c.unsqueeze(0)?.t()?;
            x.matmul(&right)?.squeeze(2)?
        } else if c.rank() == 3 {
            let right = &c.i((.., .., i))?.unsqueeze(2)?;
            assert_eq!(x.dims(), [batch, dim, dstate]);
            assert_eq!(right.dims(), [batch, dstate, 1]);
            x.matmul(&right)?.squeeze(2)?
        } else {
            let right = &c.i((.., .., .., i))?.unsqueeze(3)?;
            assert_eq!(right.dims(), [batch, dim, dstate, 1]);
            assert_eq!(x.dims(), [batch, dim, dstate]);
            x.unsqueeze(2)?.matmul(&right)?.squeeze(3)?.squeeze(2)?
        };
        ys.push(y)
    }
    let y = Tensor::stack(ys.as_slice(), 2)?;
    let out = if let Some(d) = d {
        let d = d.reshape((1, dim, 1))?;
        (y + u.broadcast_mul(&d))?
    } else {
        y
    };
    let last_state = x;
    let out = if let Some(z) = z {
        (out * silu(z)?)?
    } else {
        out
    };
    let out = out.to_dtype(dtype_in)?;

    Ok((out, Some(last_state)))
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x / (x.neg()?.exp()? + 1.0)?
}
