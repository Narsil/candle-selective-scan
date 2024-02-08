use candle::{Device, Tensor};

pub(crate) trait SameDevice {
    fn is_same_device(&self, device: &Device) -> bool;
    fn device_fmt(&self) -> String;
}

impl SameDevice for &mut Tensor {
    fn is_same_device(&self, _device: &Device) -> bool {
        // TODO Why isn't PartialEq detected here ?
        // self.device() == device
        true
    }
    fn device_fmt(&self) -> String {
        format!("{:?}", self.device())
    }
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
pub(crate) use check_same_device;
