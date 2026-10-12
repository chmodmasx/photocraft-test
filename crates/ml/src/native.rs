//! ONNX Runtime CPU adapter; verified model file supplied explicitly by the user.
use crate::{InpaintRequest, MlError, Result};
use ort::session::{Session as OnnxSession, builder::GraphOptimizationLevel};
use ort::value::Tensor;
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::Read;
use std::path::Path;
use std::sync::{Mutex, PoisonError};

const MODEL_BYTES: u64 = 28_079_181;
const MODEL_SHA256: &str = "6f1f3530a1a2324b19752018ce756088b07973cda8d7d890034ace5c8a48c40b";

pub struct NativeMigan {
    session: Mutex<OnnxSession>,
}
fn verify(path: &Path) -> Result<()> {
    let mut file = File::open(path).map_err(|e| MlError::Model(e.to_string()))?;
    if file.metadata().map_err(|e| MlError::Model(e.to_string()))?.len() != MODEL_BYTES {
        return Err(MlError::Model("model file size does not match the pinned export".into()));
    }
    let mut hash = Sha256::new();
    let mut chunk = [0; 65536];
    loop {
        let n = file.read(&mut chunk).map_err(|e| MlError::Model(e.to_string()))?;
        if n == 0 {
            break;
        }
        hash.update(&chunk[..n]);
    }
    if format!("{:x}", hash.finalize()) != MODEL_SHA256 {
        return Err(MlError::Model("model SHA-256 mismatch".into()));
    }
    Ok(())
}
impl NativeMigan {
    pub fn load(path: &Path, threads: usize) -> Result<Self> {
        if !(1..=64).contains(&threads) {
            return Err(MlError::Invalid("threads must be 1..=64".into()));
        }
        verify(path)?;
        let session = OnnxSession::builder()
            .map_err(|e| MlError::Inference(e.to_string()))?
            .with_intra_threads(threads)
            .map_err(|e| MlError::Inference(e.to_string()))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| MlError::Inference(e.to_string()))?
            .commit_from_file(path)
            .map_err(|e| MlError::Inference(e.to_string()))?;
        Ok(Self { session: Mutex::new(session) })
    }
    /// Cancellation is checked before and after ONNX; the native call itself is not interrupted.
    pub fn inpaint(&self, req: &InpaintRequest, cancel: impl Fn() -> bool) -> Result<Vec<u8>> {
        if cancel() {
            return Err(MlError::Cancelled);
        }
        let image =
            Tensor::from_array(([1usize, 3, req.height as usize, req.width as usize], req.image_nchw()?)).map_err(|e| MlError::Inference(e.to_string()))?;
        let mask =
            Tensor::from_array(([1usize, 1, req.height as usize, req.width as usize], req.known_mask()?)).map_err(|e| MlError::Inference(e.to_string()))?;
        if cancel() {
            return Err(MlError::Cancelled);
        }
        let mut session = self.session.lock().unwrap_or_else(PoisonError::into_inner);
        let outputs = session.run(ort::inputs!["image" => image, "mask" => mask]).map_err(|e| MlError::Inference(e.to_string()))?;
        if cancel() {
            return Err(MlError::Cancelled);
        }
        let value = outputs.iter().next().ok_or_else(|| MlError::Inference("no output".into()))?.1;
        let (shape, data) = value.try_extract_tensor::<u8>().map_err(|e| MlError::Inference(e.to_string()))?;
        let expected = [1i64, 3, i64::from(req.height), i64::from(req.width)];
        if shape.iter().copied().collect::<Vec<_>>() != expected {
            return Err(MlError::Inference("unexpected NCHW tensor shape".into()));
        }
        // The engine applies the selection mask as a layer mask; blending here
        // would apply feathered coverage twice and create dark/transparent seams.
        let count = req.selection.len();
        let mut rgb = vec![0u8; count * 3];
        for (i, pixel) in rgb.chunks_exact_mut(3).enumerate() {
            for channel in 0..3 {
                pixel[channel] = data[channel * count + i];
            }
        }
        Ok(rgb)
    }
}
