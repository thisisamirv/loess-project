//! Layer 6: Adapters
//!
//! This layer provides user-facing APIs that adapt the engine layer for different
//! execution modes and use cases:
//!
//! - **Batch**: Unified adapter for parallel/sequential execution
//! - **Streaming**: Chunked processing for large datasets
//! - **Online**: Incremental updates for real-time data

// Unified batch adapter for LOESS smoothing.
pub mod batch;

// Streaming LOESS for large datasets.
pub mod streaming;

// Online LOESS for real-time data streams.
pub mod online;

use loess_rs::internals::primitives::errors::LoessError;
use loess_rs::internals::primitives::policies::DistanceMetric;

pub(crate) fn apply_weighted_metric_weights<T>(
    metric: &mut DistanceMetric<T>,
    weights: Option<Vec<T>>,
) -> Result<(), LoessError> {
    if let Some(weights) = weights {
        if let DistanceMetric::Weighted(metric_weights) = metric {
            *metric_weights = weights;
        } else {
            return Err(LoessError::InvalidInput(
                "weighted_metric_weights requires distance_metric(\"weighted\")".into(),
            ));
        }
    } else if let DistanceMetric::Weighted(weights) = metric
        && weights.is_empty()
    {
        return Err(LoessError::InvalidOption {
            option: "distance_metric",
            value: "weighted".to_string(),
            valid: "use .weighted_metric_weights(vec![...]) to supply per-dimension weights",
        });
    }
    Ok(())
}
