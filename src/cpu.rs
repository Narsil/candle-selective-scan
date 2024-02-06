use candle::{DType, IndexOp, Result, Tensor, D};

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
        delta = softplus(&delta)?
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
    let (batch, dim, dstate) = state.dims3()?;
    assert_eq!(u.dims(), [batch, dim]);
    assert_eq!(delta.dims(), [batch, dim]);
    assert_eq!(a.dims(), [dim, dstate]);
    assert_eq!(b.dims(), [batch, dstate]);
    assert_eq!(c.dims(), [batch, dstate]);

    let delta = if let Some(delta_bias) = delta_bias {
        delta.broadcast_add(&delta_bias)?
    } else {
        delta.clone()
    };
    let delta = if delta_softplus {
        softplus(&delta)?
    } else {
        delta.clone()
    };
    // delta: (batch, dim)
    // a: (dim, dstate)
    // delta_a = (batch, dim, dstate)
    let delta = delta.unsqueeze(2)?;
    let delta_a = delta.broadcast_mul(&a.unsqueeze(0)?)?;
    let delta_a = delta_a.exp()?;
    assert_eq!(delta_a.dims(), [batch, dim, dstate]);

    let delta_b = delta.broadcast_mul(&b.unsqueeze(1)?)?;

    let new_state = ((&*state * &delta_a)? + delta_b.broadcast_mul(&u.unsqueeze(2)?)?)?;

    let cstate = state.to_dtype(c.dtype())?;
    let mut out = cstate.matmul(&c.unsqueeze(2)?)?.squeeze(1)?;
    if let Some(d) = &d {
        out = (out + u.broadcast_mul(&d.unsqueeze(0)?)?)?;
    }
    if let Some(z) = &z {
        out = (out * silu(z)?)?;
    }

    Ok((out, new_state))
}

pub fn apply_causal_conv1d(
    x: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    _seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x = x.to_dtype(weight.dtype())?;
    let seqlen = x.dim(2)?;
    let (dim, width) = weight.dims2()?;
    let padding = width - 1;
    let stride = 1;
    let dilation = 1;
    let groups = dim;
    let mut out = x.conv1d(&weight.unsqueeze(1)?, padding, stride, dilation, groups)?;
    if let Some(bias) = bias {
        out = (out + bias)?;
    }
    let mut out = out.i((.., .., ..seqlen))?;
    if activation {
        out = silu(&out)?;
    }
    let out = out.to_dtype(dtype_in)?;
    Ok(out)
}

pub fn apply_causal_conv1d_update(
    x: &Tensor,
    conv_state: &mut Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    _seq_idx: Option<&Tensor>,
    activation: bool,
) -> Result<Tensor> {
    let dtype_in = x.dtype();
    let x = x.to_dtype(weight.dtype())?;

    let new_state = Tensor::cat(&[conv_state.i((.., .., 1..))?, x], D::Minus1)?;
    let mut out = (new_state * weight)?.sum(D::Minus1)?;
    if let Some(bias) = bias {
        out = (out + bias)?;
    }
    if activation {
        out = silu(&out)?;
    }
    let out = out.to_dtype(dtype_in)?;
    Ok(out)
}

fn silu(x: &Tensor) -> Result<Tensor> {
    x / (x.neg()?.exp()? + 1.0)?
}

fn softplus(x: &Tensor) -> Result<Tensor> {
    ((x.exp()? + 1.)?).log()
}
