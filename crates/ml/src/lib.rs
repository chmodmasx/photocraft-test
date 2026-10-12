//! Optional CPU inpainting backend with no document, UI or network dependencies.
//! Public selection convention: 255=remove, 0=keep.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
mod native;
#[cfg(all(feature = "onnx", not(target_arch = "wasm32")))]
pub use native::NativeMigan;

pub const MAX_SIDE: u32 = 2048;
pub const MAX_PIXELS: usize = 4_194_304;

#[derive(Debug, thiserror::Error)]
pub enum MlError {
    #[error("invalid MI-GAN input: {0}")]
    Invalid(String),
    #[error("MI-GAN inference error: {0}")]
    Inference(String),
    #[error("MI-GAN model error: {0}")]
    Model(String),
    #[error("cancelled")]
    Cancelled,
}
pub type Result<T> = std::result::Result<T, MlError>;

pub struct InpaintRequest {
    pub width: u32,
    pub height: u32,
    pub rgb: Vec<u8>,
    pub selection: Vec<u8>,
}
impl InpaintRequest {
    pub fn validate(&self) -> Result<()> {
        let n = (self.width as usize).checked_mul(self.height as usize)
            .ok_or_else(|| MlError::Invalid("image size overflow".into()))?;
        if self.width == 0 || self.height == 0 || self.width > MAX_SIDE || self.height > MAX_SIDE || n > MAX_PIXELS {
            return Err(MlError::Invalid("maximum 2048x2048 and 4MP".into()));
        }
        if self.rgb.len() != n * 3 || self.selection.len() != n {
            return Err(MlError::Invalid("unexpected RGB/selection buffer lengths".into()));
        }
        if !self.selection.iter().any(|&a| a > 0) || !self.selection.iter().any(|&a| a == 0) {
            return Err(MlError::Invalid("selection must leave part of the image unchanged".into()));
        }
        Ok(())
    }
    pub fn image_nchw(&self) -> Result<Vec<u8>> {
        self.validate()?;
        let n = self.selection.len();
        let mut out = vec![0; n * 3];
        for (i, rgb) in self.rgb.chunks_exact(3).enumerate() {
            for ch in 0..3 { out[ch * n + i] = rgb[ch]; }
        }
        Ok(out)
    }
    pub fn known_mask(&self) -> Result<Vec<u8>> {
        self.validate()?;
        Ok(self.selection.iter().map(|&a| if a == 0 { 255 } else { 0 }).collect())
    }
}
/// Strictly preserve RGB bytes outside the selection, blending soft edges inside.
pub fn compose(request: &InpaintRequest, nchw: &[u8]) -> Result<Vec<u8>> {
    request.validate()?;
    let n = request.selection.len();
    if nchw.len() != n * 3 { return Err(MlError::Invalid("unexpected output length".into())); }
    let mut output = request.rgb.clone();
    for (i, &alpha) in request.selection.iter().enumerate() {
        if alpha == 0 { continue; }
        let a = f32::from(alpha) / 255.0;
        for channel in 0..3 {
            let j = i * 3 + channel;
            let original = f32::from(output[j]);
            let new = f32::from(nchw[channel * n + i]);
            output[j] = (original * (1.0 - a) + new * a).round().clamp(0.0, 255.0) as u8;
        }
    }
    Ok(output)
}
#[cfg(test)]
mod tests {
    use super::*;
    fn sample() -> InpaintRequest {
        InpaintRequest { width: 2, height: 1, rgb: vec![10,20,30,40,50,60], selection: vec![0,255] }
    }
    #[test]
    fn tensor_layout_and_mask_polarity() {
        assert_eq!(sample().image_nchw().unwrap(), [10,40,20,50,30,60]);
        assert_eq!(sample().known_mask().unwrap(), [255,0]);
    }
    #[test]
    fn outside_selection_unchanged_and_soft_edges() {
        let raw = [200,210,200,210,200,210];
        assert_eq!(&compose(&sample(), &raw).unwrap()[..3], &[10,20,30]);
        let mut req = sample();
        req.selection[1] = 128;
        let pixels = compose(&req, &raw).unwrap();
        assert!(pixels[3] > 40 && pixels[3] < 210);
    }
    #[test]
    fn rejects_malformed_inputs() {
        let mut r = sample();
        r.width = u32::MAX;
        assert!(r.validate().is_err());
        let mut r = sample();
        r.rgb.truncate(2);
        assert!(r.validate().is_err());
        let mut r = sample();
        r.selection = vec![255,255];
        assert!(r.validate().is_err());
        assert!(compose(&sample(), &[]).is_err());
    }
}
