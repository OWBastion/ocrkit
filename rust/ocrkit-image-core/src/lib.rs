use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct ImageSize {
    pub width: u32,
    pub height: u32,
}

impl ImageSize {
    pub const fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    pub fn aspect_ratio(self) -> Option<f64> {
        (self.height > 0).then_some(self.width as f64 / self.height as f64)
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct RoiBox {
    pub x1: u32,
    pub y1: u32,
    pub x2: u32,
    pub y2: u32,
}

impl RoiBox {
    pub const fn new(x1: u32, y1: u32, x2: u32, y2: u32) -> Self {
        Self { x1, y1, x2, y2 }
    }

    pub fn width(self) -> u32 {
        self.x2.saturating_sub(self.x1)
    }

    pub fn height(self) -> u32 {
        self.y2.saturating_sub(self.y1)
    }

    pub fn scaled_to(self, source: ImageSize, standard: ImageSize) -> Result<Self, LayoutError> {
        if source.width == 0 || source.height == 0 {
            return Err(LayoutError::InvalidImageSize(source));
        }
        if standard.width == 0 || standard.height == 0 {
            return Err(LayoutError::InvalidImageSize(standard));
        }

        Ok(Self {
            x1: scale_start(self.x1, source.width, standard.width),
            y1: scale_start(self.y1, source.height, standard.height),
            x2: scale_end(self.x2, source.width, standard.width),
            y2: scale_end(self.y2, source.height, standard.height),
        })
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub struct LayoutManifest {
    pub schema_version: String,
    pub layout_version: String,
    pub standard_size: ImageSize,
    pub rois: BTreeMap<String, RoiBox>,
}

impl LayoutManifest {
    pub fn validate(&self) -> Result<(), LayoutError> {
        if self.schema_version.trim().is_empty() {
            return Err(LayoutError::EmptyVersion("schema_version"));
        }
        if self.layout_version.trim().is_empty() {
            return Err(LayoutError::EmptyVersion("layout_version"));
        }
        if self.standard_size.width == 0 || self.standard_size.height == 0 {
            return Err(LayoutError::InvalidImageSize(self.standard_size));
        }
        if self.rois.is_empty() {
            return Err(LayoutError::NoRois);
        }

        for (name, roi) in &self.rois {
            if roi.x1 >= roi.x2 || roi.y1 >= roi.y2 {
                return Err(LayoutError::InvalidRoi(name.clone(), *roi));
            }
            if roi.x2 > self.standard_size.width || roi.y2 > self.standard_size.height {
                return Err(LayoutError::RoiOutsideCanvas(
                    name.clone(),
                    *roi,
                    self.standard_size,
                ));
            }
        }
        Ok(())
    }

    pub fn scaled_rois(&self, source: ImageSize) -> Result<BTreeMap<String, RoiBox>, LayoutError> {
        self.validate()?;
        if source.width == 0 || source.height == 0 {
            return Err(LayoutError::InvalidImageSize(source));
        }
        self.rois
            .iter()
            .map(|(name, roi)| {
                roi.scaled_to(source, self.standard_size)
                    .map(|scaled| (name.clone(), scaled))
            })
            .collect()
    }
}

#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
pub struct QualityReport {
    pub original_size: ImageSize,
    pub aspect_ratio: Option<f64>,
    pub layout_confidence: f64,
    pub cropped: bool,
    pub warnings: Vec<String>,
}

pub fn assess_quality(source: ImageSize, target: ImageSize) -> Result<QualityReport, LayoutError> {
    if source.width == 0 || source.height == 0 {
        return Err(LayoutError::InvalidImageSize(source));
    }
    if target.width == 0 || target.height == 0 {
        return Err(LayoutError::InvalidImageSize(target));
    }

    let aspect_ratio = source.width as f64 / source.height as f64;
    let target_ratio = target.width as f64 / target.height as f64;
    let aspect_difference = (aspect_ratio - target_ratio).abs() / target_ratio;
    let tolerance = 0.03;
    let cropped = aspect_difference > tolerance;
    let layout_confidence = (1.0 - aspect_difference / tolerance).clamp(0.0, 1.0);
    let warnings = if cropped {
        vec![
            "quality.aspect_ratio_mismatch".to_string(),
            "quality.possible_crop".to_string(),
        ]
    } else {
        Vec::new()
    };

    Ok(QualityReport {
        original_size: source,
        aspect_ratio: Some(round_four(aspect_ratio)),
        layout_confidence: round_four(layout_confidence),
        cropped,
        warnings,
    })
}

fn scale_start(value: u32, source: u32, standard: u32) -> u32 {
    ((value as u64 * source as u64) / standard as u64) as u32
}

fn scale_end(value: u32, source: u32, standard: u32) -> u32 {
    ((value as u64 * source as u64).div_ceil(standard as u64)) as u32
}

fn round_four(value: f64) -> f64 {
    (value * 10_000.0).round() / 10_000.0
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LayoutError {
    EmptyVersion(&'static str),
    InvalidImageSize(ImageSize),
    InvalidRoi(String, RoiBox),
    RoiOutsideCanvas(String, RoiBox, ImageSize),
    NoRois,
}

impl fmt::Display for LayoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyVersion(name) => write!(f, "{name} must not be empty"),
            Self::InvalidImageSize(size) => {
                write!(
                    f,
                    "image size must be non-zero: {}x{}",
                    size.width, size.height
                )
            }
            Self::InvalidRoi(name, roi) => write!(
                f,
                "ROI {name} has invalid bounds: ({}, {})-({}, {})",
                roi.x1, roi.y1, roi.x2, roi.y2
            ),
            Self::RoiOutsideCanvas(name, roi, canvas) => write!(
                f,
                "ROI {name} ({}, {})-({}, {}) exceeds canvas {}x{}",
                roi.x1, roi.y1, roi.x2, roi.y2, canvas.width, canvas.height
            ),
            Self::NoRois => write!(f, "layout must contain at least one ROI"),
        }
    }
}

impl std::error::Error for LayoutError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest() -> LayoutManifest {
        LayoutManifest {
            schema_version: "1".to_string(),
            layout_version: "1280x720-v3".to_string(),
            standard_size: ImageSize::new(1280, 720),
            rois: BTreeMap::from([("right_panel".to_string(), RoiBox::new(1040, 0, 1275, 105))]),
        }
    }

    #[test]
    fn validates_and_scales_roi_without_losing_edge_coverage() {
        let layout = manifest();
        assert!(layout.validate().is_ok());
        assert_eq!(
            layout.scaled_rois(ImageSize::new(2560, 1440)).unwrap()["right_panel"],
            RoiBox::new(2080, 0, 2550, 210)
        );
    }

    #[test]
    fn rejects_roi_outside_standard_canvas() {
        let mut layout = manifest();
        layout
            .rois
            .insert("bad".to_string(), RoiBox::new(0, 0, 1281, 10));
        assert!(matches!(
            layout.validate(),
            Err(LayoutError::RoiOutsideCanvas(..))
        ));
    }

    #[test]
    fn quality_matches_python_aspect_contract() {
        let quality =
            assess_quality(ImageSize::new(1500, 1000), ImageSize::new(1280, 720)).unwrap();
        assert_eq!(quality.aspect_ratio, Some(1.5));
        assert_eq!(quality.layout_confidence, 0.0);
        assert!(quality.cropped);
        assert_eq!(
            quality.warnings,
            vec![
                "quality.aspect_ratio_mismatch".to_string(),
                "quality.possible_crop".to_string()
            ]
        );
    }
}
