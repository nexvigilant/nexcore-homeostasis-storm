//! # Homeostasis Machine — Storm Detection and Prevention
//!
//! Cytokine-storm-style cascade detection and coordinated prevention for the
//! Homeostasis Machine.
//!
//! ## Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`detection`] | [`StormDetector`] — evaluates risk score and storm phase per tick |
//! | [`prevention`] | [`CircuitBreaker`] + [`RateLimiter`] — per-component dampeners |
//! | [`breaker`] | [`StormBreaker`] — coordinates all dampeners under one protocol |

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![cfg_attr(
    not(test),
    deny(clippy::unwrap_used, clippy::expect_used, clippy::panic)
)]

pub mod breaker;
pub mod detection;
pub mod prevention;

pub use breaker::StormBreaker;
pub use detection::{CascadePattern, StormDetector, StormSignature, SystemEvent, TrendLabel};
pub use prevention::{CircuitBreaker, RateLimiter};
