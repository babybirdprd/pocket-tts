//! Dynamic int8 quantization, port of upstream `pocket_tts/quantization.py`.
//!
//! Upstream quantizes the FlowLM transformer's attention (Q/K/V/output
//! projections) and FFN (linear1/linear2) to dynamic int8 via torchao or
//! torch.ao; the flow matching network and the Mimi decoder stay float32.
//! The candle equivalent used here is the q8_0 quantized matmul: weights are
//! stored in 8-bit blocks and activations are quantized on the fly inside
//! the matmul kernel. Works on CPU and Metal.

use anyhow::Result;
use candle_core::Tensor;
use candle_core::quantized::{GgmlDType, QMatMul, QTensor};
use candle_nn::{Linear, Module};

/// Layer groups that can be quantized, mirroring upstream's group keys.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantizeGroup {
    /// Attention Q/K/V and output projections.
    Attention,
    /// Feed-forward linear1/linear2.
    Ffn,
    /// The flow matching MLP. Upstream supports it but does not recommend it;
    /// not ported (requests for it fail loudly).
    FlowNet,
}

/// Upstream `RECOMMENDED_CONFIG`: what `quantize=True` applies.
pub const RECOMMENDED_CONFIG: &[QuantizeGroup] = &[QuantizeGroup::Attention, QuantizeGroup::Ffn];

/// A linear projection that is either full-precision or int8-quantized.
///
/// Starts life as `Full` (a plain candle `Linear`); `quantize_int8` swaps the
/// weight for a q8_0 tensor. The projections used by the FlowLM transformer
/// have no bias.
#[derive(Clone, Debug)]
pub enum MaybeQuantLinear {
    Full(Linear),
    Int8(QMatMul),
}

impl MaybeQuantLinear {
    pub fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        match self {
            MaybeQuantLinear::Full(linear) => linear.forward(x),
            MaybeQuantLinear::Int8(qmatmul) => qmatmul.forward(x),
        }
    }

    /// Quantize the weight to q8_0 in place. No-op when already quantized.
    pub fn quantize_int8(&mut self) -> Result<()> {
        if let MaybeQuantLinear::Full(linear) = self {
            if linear.bias().is_some() {
                anyhow::bail!("int8 quantization only supports bias-free projections");
            }
            let qtensor = QTensor::quantize(linear.weight(), GgmlDType::Q8_0)?;
            *self = MaybeQuantLinear::Int8(QMatMul::from_qtensor(qtensor)?);
        }
        Ok(())
    }

    pub fn is_quantized(&self) -> bool {
        matches!(self, MaybeQuantLinear::Int8(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    #[test]
    fn quantized_matmul_stays_close_to_full_precision() -> Result<()> {
        let device = Device::Cpu;
        // Q8_0 wants the inner dimension in blocks of 32.
        let weight = Tensor::randn(0f32, 0.02, (64, 128), &device)?;
        let x = Tensor::randn(0f32, 1.0, (1, 4, 128), &device)?;

        let mut proj = MaybeQuantLinear::Full(Linear::new(weight, None));
        let full = proj.forward(&x)?;
        proj.quantize_int8()?;
        assert!(proj.is_quantized());
        let quant = proj.forward(&x)?;

        let diff = (full - quant)?
            .abs()?
            .max_all()?
            .to_dtype(DType::F32)?
            .to_scalar::<f32>()?;
        assert!(diff < 0.05, "int8 error too large: {diff}");
        Ok(())
    }
}
