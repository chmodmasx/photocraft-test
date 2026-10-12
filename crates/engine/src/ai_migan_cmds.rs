//! Experimental command: local selection -> MI-GAN CPU -> new masked raster layer.
//! Hidden from the menus until native inference, quality, and resource limits are validated.
use std::path::PathBuf;
use photocraft_cms::{Builtin, Intent, Transform};
use photocraft_color::{ColorMode, PixelFormat, SampleType};
use photocraft_doc::{Document, Layer, LayerMask};
use photocraft_geom::Rect;
use photocraft_ml::{InpaintRequest, NativeMigan};
use photocraft_raster::Surface;
use serde_json::{json, Value};
use crate::commands::CommandSpec;
use crate::{EngineError, Result, Session};

pub const ID: &str = "edit.aiRemoveSelection";
const LABEL: &str = "AI Remove Selection (MI-GAN)";

fn invalid(msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: ID.into(), msg: msg.into() }
}
fn failure(e: impl std::fmt::Display) -> EngineError {
    EngineError::Other(e.to_string())
}
fn enabled(s: &Session) -> std::result::Result<(), String> {
    let state = s.active().ok_or("no document open")?;
    if state.doc.mode != ColorMode::Rgb {
        return Err("experimental MI-GAN supports only RGB documents".into());
    }
    let selection = state.doc.selection.as_ref().ok_or("select an area before running MI-GAN")?;
    if selection.content_bounds().intersect(&state.doc.bounds()).is_empty() {
        return Err("selection is empty".into());
    }
    let bounds = state.doc.bounds();
    let n = (bounds.width() as usize).checked_mul(bounds.height() as usize)
        .ok_or("image dimensions overflow")?;
    if bounds.width() == 0 || bounds.height() == 0
        || bounds.width() > photocraft_ml::MAX_SIDE
        || bounds.height() > photocraft_ml::MAX_SIDE
        || n > photocraft_ml::MAX_PIXELS {
        return Err("experimental limit is 2048 pixels per side, 4MP total".into());
    }
    Ok(())
}

fn prepare(doc: &Document, sel: &Surface, ctx: &crate::jobs::JobCtx) -> Result<InpaintRequest> {
    ctx.check()?;
    let area = doc.bounds();
    let fmt = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let mut surface = Surface::new(fmt);
    let mut data = Vec::new();
    photocraft_compose::render_bands(doc, area, 0, |band| -> Result<()> {
        data.clear();
        data.resize(band.px.len() * 4, 0.0);
        for (p, out) in band.px.iter().zip(data.as_chunks_mut::<4>().0.iter_mut()) {
            photocraft_raster::from_rgba_into(&fmt, *p, out);
        }
        surface.write_region(band.rect, &data);
        Ok(())
    })?;
    ctx.check()?;
    let transform = Transform::new(
        &crate::color_cmds::composite_profile(doc),
        Builtin::Srgb.profile(),
        Intent::RelativeColorimetric,
        true,
    ).map_err(failure)?;
    let srgb = crate::color_cmds::convert_surface(&surface, ColorMode::Rgb, fmt, &transform);
    let rgba = srgb.to_interleaved(area);
    let count = area.width() as usize * area.height() as usize;
    if rgba.len() != count * 4 {
        return Err(failure("invalid composed pixels"));
    }
    if rgba.chunks_exact(4).any(|px| px[3] != 255) {
        return Err(invalid("the first MI-GAN prototype requires opaque RGB pixels"));
    }
    let mut rgb = Vec::with_capacity(count * 3);
    for p in rgba.chunks_exact(4) {
        rgb.extend_from_slice(&p[..3]);
    }
    let channel_count = sel.channels();
    if channel_count == 0 {
        return Err(failure("selection has no channels"));
    }
    let samples = sel.read_region(area);
    if samples.len() != count * channel_count {
        return Err(failure("selection dimensions do not match the document"));
    }
    let selection = samples.chunks_exact(channel_count)
        .map(|pixel| (pixel[0].clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect();
    let request = InpaintRequest {
        width: area.width(), height: area.height(), rgb, selection,
    };
    request.validate().map_err(failure)?;
    Ok(request)
}

fn to_document(doc: &Document, rgb: &[u8], area: Rect) -> Result<Surface> {
    let n = area.width() as usize * area.height() as usize;
    if rgb.len() != n * 3 {
        return Err(failure("invalid native inference output"));
    }
    let src_format = PixelFormat::new(ColorMode::Rgb, SampleType::U8, true);
    let mut rgba = Vec::with_capacity(n * 4);
    for rgb in rgb.chunks_exact(3) {
        rgba.extend_from_slice(&[rgb[0], rgb[1], rgb[2], 255]);
    }
    let mut source = Surface::new(src_format);
    source.write_interleaved(area, &rgba);
    let target = PixelFormat::new(doc.mode, doc.depth, true);
    let transform = Transform::new(
        Builtin::Srgb.profile(),
        &crate::color_cmds::document_profile(doc),
        Intent::RelativeColorimetric,
        true,
    ).map_err(failure)?;
    let result = crate::color_cmds::convert_surface(&source, ColorMode::Rgb, target, &transform);
    Ok(if result.format() == target { result } else { result.convert(target) })
}

fn run(s: &mut Session, p: &Value) -> Result<Value> {
    enabled(s).map_err(invalid)?;
    let obj = p.as_object().ok_or_else(|| invalid("parameters must be an object"))?;
    if obj.keys().any(|key| key != "threads") {
        return Err(invalid("only the threads parameter is supported"));
    }
    let threads = match obj.get("threads") {
        None => std::thread::available_parallelism().map_or(4, |n| n.get().min(8)),
        Some(Value::Number(n)) => n.as_u64()
            .and_then(|v| usize::try_from(v).ok())
            .filter(|v| (1..=64).contains(v))
            .ok_or_else(|| invalid("threads must be 1..=64"))?,
        _ => return Err(invalid("threads must be 1..=64")),
    };
    let model_path = std::env::var_os("PHOTOCRAFT_MIGAN_MODEL")
        .map(PathBuf::from)
        .ok_or_else(|| invalid("set PHOTOCRAFT_MIGAN_MODEL to the ONNX file"))?;
    let state = s.active().ok_or(EngineError::NoDocument)?;
    let snapshot = state.doc.clone();
    let mask = snapshot.selection.clone().ok_or_else(|| invalid("selection missing"))?;
    let area = snapshot.bounds();
    crate::jobs::run(
        s, LABEL, true,
        move |ctx| {
            ctx.progress(0.05, "Preparing image and selection");
            let input = prepare(&snapshot, &mask, ctx)?;
            ctx.progress(0.2, "Verifying MI-GAN model");
            let model = NativeMigan::load(&model_path, threads).map_err(failure)?;
            ctx.check()?;
            ctx.progress(0.4, "Running MI-GAN on CPU");
            let pixels = model.inpaint(&input, || ctx.cancelled()).map_err(|e| match e {
                photocraft_ml::MlError::Cancelled => EngineError::Cancelled,
                e => EngineError::Other(e.to_string()),
            })?;
            ctx.check()?;
            ctx.progress(0.9, "Preparing result");
            Ok((pixels, mask))
        },
        move |session, (pixels, selection)| {
            let id = session.edit(LABEL, |doc, active| {
                let converted = to_document(doc, &pixels, area)?;
                let mut layer = Layer::raster(LABEL.into(), converted.format());
                *layer.surface_mut().ok_or_else(|| failure("could not create layer"))? = converted;
                layer.mask = Some(LayerMask {
                    surface: selection,
                    ..LayerMask::reveal_all()
                });
                let id = doc.insert_above(*active, layer);
                *active = Some(id);
                Ok(id)
            })?;
            Ok(json!({
                "layer": id.0, "backend": "migan-cpu",
                "bounds": [area.x0, area.y0, area.width(), area.height()]
            }))
        },
    )
}
pub fn specs() -> Vec<CommandSpec> {
    vec![CommandSpec {
        id: ID,
        label: LABEL,
        menu: &[],
        shortcut: None,
        params: "Optional threads integer (1-64). Uses PHOTOCRAFT_MIGAN_MODEL. Adds one masked layer.",
        enabled,
        run,
        journal: true,
    }]
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn disabled_without_selection() {
        let mut session = Session::new();
        assert!(!session.is_enabled(ID));
        session.execute("file.new", json!({"width": 64, "height": 64})).unwrap();
        assert!(!session.is_enabled(ID));
        assert!(session.execute(ID, json!({})).is_err());
    }
}
