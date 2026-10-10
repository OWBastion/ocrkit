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

/// 3440x1440; the 3% tolerance also covers 21:9, 2560x1080 and 3840x1600.
const ULTRAWIDE_ASPECT: f64 = 43.0 / 18.0;
const ASPECT_TOLERANCE: f64 = 0.03;

fn relative_aspect_difference(aspect_ratio: f64, target_ratio: f64) -> f64 {
    (aspect_ratio - target_ratio).abs() / target_ratio
}

pub fn is_ultrawide(source: ImageSize) -> bool {
    source.height > 0
        && relative_aspect_difference(source.width as f64 / source.height as f64, ULTRAWIDE_ASPECT)
            <= ASPECT_TOLERANCE
}

/// The canvas a source is normalized onto: ultrawide sources keep the standard height scale.
pub fn canvas_size(source: ImageSize, standard: ImageSize) -> ImageSize {
    if !is_ultrawide(source) {
        return standard;
    }
    let numerator = standard.height as u64 * source.width as u64;
    let width = (numerator * 2 + source.height as u64) / (2 * source.height as u64);
    ImageSize::new(width as u32, standard.height)
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

    /// Place the ROI on a canvas wider than the standard one: left ROIs stay put, right ROIs
    /// follow the right edge and central ROIs the middle.
    pub fn anchored(self, standard: ImageSize, canvas_width: u32) -> Self {
        let extra = canvas_width.saturating_sub(standard.width);
        let doubled_centre = (self.x1 + self.x2) as u64 * 3;
        let shift = if doubled_centre < 2 * standard.width as u64 {
            0
        } else if doubled_centre > 4 * standard.width as u64 {
            extra
        } else {
            extra / 2
        };
        Self::new(self.x1 + shift, self.y1, self.x2 + shift, self.y2)
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

    pub fn anchored_rois(&self, canvas: ImageSize) -> BTreeMap<String, RoiBox> {
        self.rois
            .iter()
            .map(|(name, roi)| (name.clone(), roi.anchored(self.standard_size, canvas.width)))
            .collect()
    }

    pub fn scaled_rois(&self, source: ImageSize) -> Result<BTreeMap<String, RoiBox>, LayoutError> {
        self.validate()?;
        if source.width == 0 || source.height == 0 {
            return Err(LayoutError::InvalidImageSize(source));
        }
        let canvas = canvas_size(source, self.standard_size);
        self.anchored_rois(canvas)
            .into_iter()
            .map(|(name, roi)| roi.scaled_to(source, canvas).map(|scaled| (name, scaled)))
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
    let target_ratio = if is_ultrawide(source) {
        ULTRAWIDE_ASPECT
    } else {
        target.width as f64 / target.height as f64
    };
    let aspect_difference = relative_aspect_difference(aspect_ratio, target_ratio);
    let cropped = aspect_difference > ASPECT_TOLERANCE;
    let layout_confidence = (1.0 - aspect_difference / ASPECT_TOLERANCE).clamp(0.0, 1.0);
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
    fn ultrawide_keeps_height_scale_and_anchors_rois_to_screen_edges() {
        let mut layout = manifest();
        layout
            .rois
            .insert("left_panel".to_string(), RoiBox::new(35, 15, 220, 225));
        layout.rois.insert(
            "center_banner".to_string(),
            RoiBox::new(150, 125, 1160, 215),
        );
        let source = ImageSize::new(3440, 1440);

        assert_eq!(
            canvas_size(source, layout.standard_size),
            ImageSize::new(1720, 720)
        );
        assert_eq!(
            canvas_size(ImageSize::new(2560, 1440), layout.standard_size),
            layout.standard_size
        );
        let anchored = layout.anchored_rois(ImageSize::new(1720, 720));
        assert_eq!(anchored["left_panel"], RoiBox::new(35, 15, 220, 225));
        assert_eq!(anchored["center_banner"], RoiBox::new(370, 125, 1380, 215));
        assert_eq!(anchored["right_panel"], RoiBox::new(1480, 0, 1715, 105));
        assert_eq!(
            layout.scaled_rois(source).unwrap()["right_panel"],
            RoiBox::new(2960, 0, 3430, 210)
        );
    }

    #[test]
    fn ultrawide_is_not_flagged_as_cropped() {
        for (width, height) in [(3440, 1440), (2560, 1080), (3840, 1600)] {
            let quality =
                assess_quality(ImageSize::new(width, height), ImageSize::new(1280, 720)).unwrap();
            assert!(
                !quality.cropped && quality.warnings.is_empty(),
                "{width}x{height}"
            );
        }
        assert!(
            assess_quality(ImageSize::new(2000, 1000), ImageSize::new(1280, 720))
                .unwrap()
                .cropped
        );
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
