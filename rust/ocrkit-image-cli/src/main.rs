use image::{imageops::FilterType, DynamicImage, GenericImageView, ImageFormat, ImageReader};
use ocrkit_image_core::{assess_quality, ImageSize, LayoutManifest, QualityReport, RoiBox};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::env;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Component, Path, PathBuf};
use std::process::ExitCode;

fn main() -> ExitCode {
    match run(env::args().skip(1).collect()) {
        Ok(output) => {
            println!("{output}");
            ExitCode::SUCCESS
        }
        Err(message) => {
            eprintln!("error: {message}");
            eprintln!("usage:");
            eprintln!("  ocrkit-image inspect --manifest PATH --width N --height N");
            eprintln!("  ocrkit-image crop-batch --manifest PATH --cases PATH --input-root PATH --output-dir PATH");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<String>) -> Result<String, String> {
    match args.first().map(String::as_str) {
        Some("inspect") => inspect(&args),
        Some("crop-batch") => crop_batch(&args),
        _ => Err("expected the inspect or crop-batch command".to_string()),
    }
}

fn inspect(args: &[String]) -> Result<String, String> {
    let manifest_path = required_value(args, "--manifest")?;
    let width = parse_dimension(args, "--width")?;
    let height = parse_dimension(args, "--height")?;
    let manifest = read_manifest(manifest_path)?;

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

#[derive(Debug, Deserialize)]
struct BatchCase {
    id: String,
    image: String,
    #[serde(default = "default_split")]
    split: String,
}

fn default_split() -> String {
    "train".to_string()
}

#[derive(serde::Serialize)]
struct CropBatchManifest {
    schema_version: String,
    layout_version: String,
    standard_size: ImageSize,
    sources: Vec<CropSource>,
}

#[derive(serde::Serialize)]
struct CropSource {
    source_id: String,
    split: String,
    source_file: String,
    source_sha256: String,
    source_size: ImageSize,
    normalized_size: ImageSize,
    quality: QualityReport,
    rois: std::collections::BTreeMap<String, CropArtifact>,
}

#[derive(serde::Serialize)]
struct CropArtifact {
    path: String,
    #[serde(rename = "box")]
    box_: RoiBox,
    size: ImageSize,
    sha256: String,
}

fn crop_batch(args: &[String]) -> Result<String, String> {
    let manifest = read_manifest(required_value(args, "--manifest")?)?;
    let cases_path = required_value(args, "--cases")?;
    let input_root = canonical_directory(required_value(args, "--input-root")?, "input root")?;
    let output_dir = PathBuf::from(required_value(args, "--output-dir")?);
    fs::create_dir_all(&output_dir).map_err(|err| format!("create output directory: {err}"))?;
    let cases: Vec<BatchCase> = read_json(cases_path, "cases")?;

    let mut sources = Vec::with_capacity(cases.len());
    for case in cases {
        validate_component(&case.id, "case id")?;
        if case.split != "train" && case.split != "holdout" {
            return Err(format!(
                "case {} has invalid split: {}",
                case.id, case.split
            ));
        }
        let source_path = safe_input_path(&input_root, &case.image)?;
        let source_bytes =
            fs::read(&source_path).map_err(|err| format!("read source image: {err}"))?;
        let source_sha256 = sha256(&source_bytes);
        let source_image = ImageReader::new(std::io::Cursor::new(&source_bytes))
            .with_guessed_format()
            .map_err(|err| format!("detect image format for {}: {err}", case.image))?
            .decode()
            .map_err(|err| format!("decode {}: {err}", case.image))?;
        let source_size = image_size(&source_image);
        let quality =
            assess_quality(source_size, manifest.standard_size).map_err(|err| err.to_string())?;
        let normalized = source_image.resize_exact(
            manifest.standard_size.width,
            manifest.standard_size.height,
            FilterType::Triangle,
        );

        let mut rois = std::collections::BTreeMap::new();
        for (roi_name, roi) in &manifest.rois {
            validate_component(roi_name, "ROI name")?;
            let crop = normalized.crop_imm(roi.x1, roi.y1, roi.width(), roi.height());
            let relative_path = PathBuf::from("images")
                .join(&case.split)
                .join(&case.id)
                .join(format!("{roi_name}.png"));
            let crop_path = output_dir.join(&relative_path);
            if let Some(parent) = crop_path.parent() {
                fs::create_dir_all(parent)
                    .map_err(|err| format!("create crop directory: {err}"))?;
            }
            crop.save_with_format(&crop_path, ImageFormat::Png)
                .map_err(|err| format!("write crop {}: {err}", crop_path.display()))?;
            let crop_bytes = fs::read(&crop_path).map_err(|err| format!("read crop: {err}"))?;
            rois.insert(
                roi_name.clone(),
                CropArtifact {
                    path: relative_path.to_string_lossy().into_owned(),
                    box_: *roi,
                    size: ImageSize::new(roi.width(), roi.height()),
                    sha256: sha256(&crop_bytes),
                },
            );
        }

        sources.push(CropSource {
            source_id: case.id,
            split: case.split,
            source_file: case.image,
            source_sha256,
            source_size,
            normalized_size: manifest.standard_size,
            quality,
            rois,
        });
    }

    let result = CropBatchManifest {
        schema_version: "1".to_string(),
        layout_version: manifest.layout_version,
        standard_size: manifest.standard_size,
        sources,
    };
    let manifest_path = output_dir.join("crop_manifest.json");
    let encoded =
        serde_json::to_vec_pretty(&result).map_err(|err| format!("encode crop manifest: {err}"))?;
    let mut file =
        File::create(&manifest_path).map_err(|err| format!("create crop manifest: {err}"))?;
    file.write_all(&encoded)
        .and_then(|_| file.write_all(b"\n"))
        .map_err(|err| format!("write crop manifest: {err}"))?;
    serde_json::to_string_pretty(&serde_json::json!({
        "sources": result.sources.len(),
        "crop_manifest": manifest_path,
    }))
    .map_err(|err| format!("encode result: {err}"))
}

fn read_manifest(path: &str) -> Result<LayoutManifest, String> {
    let manifest: LayoutManifest = read_json(path, "manifest")?;
    manifest.validate().map_err(|err| err.to_string())?;
    Ok(manifest)
}

fn read_json<T: for<'de> Deserialize<'de>>(path: &str, label: &str) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|err| format!("read {label}: {err}"))?;
    serde_json::from_str(&text).map_err(|err| format!("parse {label}: {err}"))
}

fn image_size(image: &DynamicImage) -> ImageSize {
    let (width, height) = image.dimensions();
    ImageSize::new(width, height)
}

fn sha256(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    format!("{digest:x}")
}

fn canonical_directory(path: &str, label: &str) -> Result<PathBuf, String> {
    let path = PathBuf::from(path);
    if !path.is_dir() {
        return Err(format!("{label} is not a directory: {}", path.display()));
    }
    fs::canonicalize(&path).map_err(|err| format!("resolve {label}: {err}"))
}

fn safe_input_path(root: &Path, relative: &str) -> Result<PathBuf, String> {
    let path = Path::new(relative);
    if path.is_absolute()
        || path.components().any(|component| {
            matches!(
                component,
                Component::ParentDir | Component::RootDir | Component::Prefix(_)
            )
        })
    {
        return Err(format!(
            "source image path must stay under input root: {relative}"
        ));
    }
    let resolved = fs::canonicalize(root.join(path))
        .map_err(|err| format!("resolve source image {relative}: {err}"))?;
    if !resolved.starts_with(root) {
        return Err(format!("source image path escapes input root: {relative}"));
    }
    Ok(resolved)
}

fn validate_component(value: &str, label: &str) -> Result<(), String> {
    if value.is_empty()
        || value == "."
        || value == ".."
        || value.contains('/')
        || value.contains('\\')
    {
        return Err(format!(
            "{label} must be a single safe path component: {value}"
        ));
    }
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn crop_batch_writes_roi_dimensions_and_hash_manifest() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = env::temp_dir().join(format!("ocrkit-image-cli-test-{suffix}"));
        let input_root = root.join("input");
        let output_dir = root.join("output");
        fs::create_dir_all(&input_root).unwrap();

        let source = ImageBuffer::from_pixel(4, 2, Rgb([120u8, 80u8, 40u8]));
        source.save(input_root.join("source.png")).unwrap();
        let layout = LayoutManifest {
            schema_version: "1".to_string(),
            layout_version: "test-v1".to_string(),
            standard_size: ImageSize::new(4, 2),
            rois: std::collections::BTreeMap::from([(
                "left_panel".to_string(),
                RoiBox::new(0, 0, 2, 1),
            )]),
        };
        let manifest_path = root.join("layout.json");
        fs::write(&manifest_path, serde_json::to_vec(&layout).unwrap()).unwrap();
        let cases_path = root.join("cases.json");
        fs::write(
            &cases_path,
            br#"[{"id":"case-1","image":"source.png","split":"train"}]"#,
        )
        .unwrap();

        let args = vec![
            "crop-batch".to_string(),
            "--manifest".to_string(),
            manifest_path.to_string_lossy().into_owned(),
            "--cases".to_string(),
            cases_path.to_string_lossy().into_owned(),
            "--input-root".to_string(),
            input_root.to_string_lossy().into_owned(),
            "--output-dir".to_string(),
            output_dir.to_string_lossy().into_owned(),
        ];
        crop_batch(&args).unwrap();

        let crop_path = output_dir.join("images/train/case-1/left_panel.png");
        assert_eq!(
            ImageReader::open(&crop_path)
                .unwrap()
                .decode()
                .unwrap()
                .dimensions(),
            (2, 1)
        );
        let crop_hash = sha256(&fs::read(&crop_path).unwrap());
        let crop_manifest: serde_json::Value =
            serde_json::from_slice(&fs::read(output_dir.join("crop_manifest.json")).unwrap())
                .unwrap();
        assert_eq!(
            crop_manifest["sources"][0]["rois"]["left_panel"]["sha256"],
            crop_hash
        );
        assert_eq!(
            crop_manifest["sources"][0]["rois"]["left_panel"]["box"],
            serde_json::json!({"x1": 0, "y1": 0, "x2": 2, "y2": 1})
        );

        fs::remove_dir_all(root).unwrap();
    }
}
