use ocrkit_image_core::{assess_quality, ImageSize, LayoutManifest};
use std::env;
use std::fs;
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("usage: ocrkit-image inspect --manifest PATH --width N --height N");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<String, String> {
    if args.first().map(String::as_str) != Some("inspect") {
        return Err("expected the inspect command".to_string());
    }

    let manifest_path = required_value(&args, "--manifest")?;
    let width = parse_dimension(&args, "--width")?;
    let height = parse_dimension(&args, "--height")?;
    let manifest: LayoutManifest = serde_json::from_str(
        &fs::read_to_string(manifest_path).map_err(|err| format!("read manifest: {err}"))?,
    )
    .map_err(|err| format!("parse manifest: {err}"))?;
    manifest.validate().map_err(|err| err.to_string())?;

    let source = ImageSize::new(width, height);
    let quality = assess_quality(source, manifest.standard_size).map_err(|err| err.to_string())?;
    let scaled_rois = manifest
        .scaled_rois(source)
        .map_err(|err| err.to_string())?;

    serde_json::to_string_pretty(&serde_json::json!({
        "schema_version": manifest.schema_version,
        "layout_version": manifest.layout_version,
        "source_size": source,
        "standard_size": manifest.standard_size,
        "quality": quality,
        "rois": scaled_rois,
    }))
    .map_err(|err| format!("encode result: {err}"))
}

fn required_value<'a>(args: &'a [String], name: &str) -> Result<&'a str, String> {
    let index = args
        .iter()
        .position(|arg| arg == name)
        .ok_or_else(|| format!("missing {name}"))?;
    args.get(index + 1)
        .map(String::as_str)
        .filter(|value| !value.is_empty() && !value.starts_with('-'))
        .ok_or_else(|| format!("missing value for {name}"))
}

fn parse_dimension(args: &[String], name: &str) -> Result<u32, String> {
    let value = required_value(args, name)?;
    let dimension = value
        .parse::<u32>()
        .map_err(|_| format!("{name} must be a positive integer"))?;
    if dimension == 0 {
        return Err(format!("{name} must be a positive integer"));
    }
    Ok(dimension)
}
