use candle::{bail, DType, Device, IndexOp, Result, Storage, Tensor};

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
        if b.rank() == 3 {
            // B is batch, dim, seqlen
            let tmp = (delta.mul(&u.unsqueeze(3)?))?;
            let b = b.t()?;
            let b = b.reshape((batch, 1, seqlen, dstate))?;
            tmp.broadcast_mul(&b)?
        } else {
            todo!("GRouped");
        }
    };
    let mut ys = Vec::with_capacity(seqlen);
    for i in 0..seqlen {
        x = ((delta_a.i((.., .., i))? * x)? + delta_b_u.i((.., .., i))?)?;
        let y = if c.rank() == 3 {
            let right = &c.i((.., .., i))?.unsqueeze(1)?.t()?;
            x.matmul(&right)?.squeeze(2)?
        } else {
            x.matmul(&c.i((.., i, ..))?.unsqueeze(2)?)?.squeeze(2)?
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
