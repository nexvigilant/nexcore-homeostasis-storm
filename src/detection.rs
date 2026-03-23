//! Storm detection engine — early warning for cascading failure.
//!
//! The cytokine storm teaches us the signature of runaway amplification:
//! 1. Response level increasing while threat level is stable or decreasing.
//! 2. Proportionality ratio >> 1 (massive over-response).
//! 3. Response rate of change itself accelerating.
//! 4. Internal damage from the system's own response.
//!
//! This module detects these signatures before the system reaches the point
//! of no return. Better a false positive than a missed storm.

use nexcore_homeostasis_primitives::enums::StormPhase;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::time::Duration;
use tokio::time::Instant;

// =============================================================================
// Public types
// =============================================================================

/// The diagnostic signature of a storm at a single point in time.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StormSignature {
    /// Current storm phase.
    pub phase: StormPhase,
    /// Integrated risk score in `[0, 1]`.
    pub risk_score: f64,
    /// Response-to-threat ratio.
    pub proportionality: f64,
    /// Trend direction of the proportionality ratio.
    pub proportionality_trend: TrendLabel,
    /// Acceleration of the response level (rate-of-change of rate-of-change).
    pub response_acceleration: f64,
    /// Raw threat level at evaluation time.
    pub threat_level: f64,
    /// Raw response level at evaluation time.
    pub response_level: f64,
    /// How long proportionality has been above the warning threshold, in seconds.
    pub duration_at_elevated_secs: f64,
    /// Time since the storm was first detected, in seconds (zero if no active storm).
    pub time_since_detection_secs: f64,
    /// Whether the system appears to be damaging itself.
    pub self_damage_detected: bool,
    /// Labels identifying the detected self-damage sources.
    pub self_damage_sources: Vec<String>,
}

/// Three-way trend label.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TrendLabel {
    /// Values rising over the sample window.
    Increasing,
    /// No statistically significant change.
    Stable,
    /// Values falling over the sample window.
    Decreasing,
}

/// A detected cascade pattern — one subsystem's failure propagating to another's.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CascadePattern {
    /// The system whose failure propagated outward.
    pub source_system: String,
    /// Systems that failed within the cascade window.
    pub affected_systems: Vec<String>,
    /// Milliseconds between the source failure and the first downstream failure.
    pub propagation_delay_ms: f64,
    /// Number of affected systems (rough amplification measure).
    pub amplification_factor: f64,
}

/// A single timestamped system event used as input to [`StormDetector::detect_cascade`].
#[derive(Clone, Debug)]
pub struct SystemEvent {
    /// Name of the originating subsystem.
    pub system: String,
    /// Event category — `"error"`, `"failure"`, or `"timeout"` trigger cascade checks.
    pub event_type: String,
    /// Monotonic timestamp of the event.
    pub timestamp: Instant,
}

impl SystemEvent {
    /// Construct a new event timestamped to now.
    pub fn new(system: impl Into<String>, event_type: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            event_type: event_type.into(),
            timestamp: Instant::now(),
        }
    }
}

// =============================================================================
// Internal types
// =============================================================================

#[derive(Clone, Debug)]
struct Sample {
    threat: f64,
    response: f64,
    damage: f64,
    proportionality: f64,
    recorded_at: Instant,
}

struct SelfDamageResult {
    detected: bool,
    sources: Vec<String>,
}

impl SelfDamageResult {
    fn none() -> Self {
        Self {
            detected: false,
            sources: vec![],
        }
    }
}

// =============================================================================
// StormDetector
// =============================================================================

/// The main storm detection engine.
///
/// Evaluates system state on each control-loop tick, returning a
/// [`StormSignature`] that captures the risk score, current phase, and all
/// contributing indicators.
///
/// The risk score is computed from five weighted factors:
///
/// | Factor | Max weight |
/// |--------|-----------|
/// | Proportionality level | 0.25 |
/// | Proportionality trend | 0.25 |
/// | Response acceleration | 0.20 |
/// | Duration at elevated | 0.15 |
/// | Self-damage detected | 0.15 |
#[derive(Debug)]
pub struct StormDetector {
    /// Proportionality ratio that starts building risk (default 3.0).
    pub proportionality_warning: f64,
    /// Proportionality ratio for high risk (default 5.0).
    pub proportionality_critical: f64,
    /// Proportionality ratio at which maximum proportionality weight applies (default 10.0).
    pub proportionality_storm: f64,
    /// Acceleration above this contributes moderate risk (default 0.1).
    pub acceleration_warning: f64,
    /// Acceleration above this contributes high risk (default 0.3).
    pub acceleration_critical: f64,
    /// Duration above warning before partial duration risk applies (default 5 min).
    pub elevated_duration_warning: Duration,
    /// Duration above warning before full duration risk applies (default 15 min).
    pub elevated_duration_critical: Duration,
    /// Maximum age of retained history samples (default 30 min).
    pub history_window: Duration,

    history: VecDeque<Sample>,
    max_history_samples: usize,
    current_phase: StormPhase,
    storm_start: Option<Instant>,
    elevated_start: Option<Instant>,
    detected_cascades: usize,
}

impl Default for StormDetector {
    fn default() -> Self {
        Self::new(
            3.0,
            5.0,
            10.0,
            0.1,
            0.3,
            Duration::from_secs(300),
            Duration::from_secs(900),
            Duration::from_secs(1800),
            500,
        )
    }
}

impl StormDetector {
    /// Create a `StormDetector` with explicit configuration.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        proportionality_warning: f64,
        proportionality_critical: f64,
        proportionality_storm: f64,
        acceleration_warning: f64,
        acceleration_critical: f64,
        elevated_duration_warning: Duration,
        elevated_duration_critical: Duration,
        history_window: Duration,
        max_history_samples: usize,
    ) -> Self {
        Self {
            proportionality_warning,
            proportionality_critical,
            proportionality_storm,
            acceleration_warning,
            acceleration_critical,
            elevated_duration_warning,
            elevated_duration_critical,
            history_window,
            history: VecDeque::with_capacity(max_history_samples.min(512)),
            max_history_samples,
            current_phase: StormPhase::Clear,
            storm_start: None,
            elevated_start: None,
            detected_cascades: 0,
        }
    }

    /// Evaluate the current state for storm signatures.
    ///
    /// Call on every control-loop iteration. Returns a [`StormSignature`] with
    /// the current risk assessment.
    pub fn evaluate(
        &mut self,
        threat_level: f64,
        response_level: f64,
        damage_level: f64,
        _extra_metrics: Option<&std::collections::HashMap<String, f64>>,
    ) -> StormSignature {
        let proportionality = Self::compute_proportionality(threat_level, response_level);
        self.record_sample(threat_level, response_level, damage_level, proportionality);

        let proportionality_trend = self.calc_trend_proportionality(10);
        let response_acceleration = self.calc_acceleration_response(10);
        let dmg = self.check_self_damage(response_level);

        // Duration at elevated proportionality.
        let duration_elevated = if proportionality > self.proportionality_warning {
            if self.elevated_start.is_none() {
                self.elevated_start = Some(Instant::now());
            }
            self.elevated_start
                .map(|t| t.elapsed())
                .unwrap_or(Duration::ZERO)
        } else {
            self.elevated_start = None;
            Duration::ZERO
        };

        let base_risk = self.calc_risk_score(
            proportionality,
            proportionality_trend,
            response_acceleration,
            duration_elevated,
            dmg.detected,
        );

        // Sustained storm bonus: a steady-state massive over-response is itself
        // dangerous even when the signal is constant (no trend / no acceleration).
        // If many recent samples are at storm-level proportionality, the
        // persistence of over-response contributes additional risk.
        let sustained_bonus = {
            let recent = self.last_n_values(15, |s| s.proportionality);
            let at_storm = recent
                .iter()
                .filter(|p| **p >= self.proportionality_storm)
                .count();
            let raw = if at_storm >= 10 {
                0.15
            } else if at_storm >= 5 {
                0.08
            } else {
                0.0
            };
            // Dampen when current proportionality shows resolution —
            // the storm memory fades as the system returns to normal.
            if proportionality < self.proportionality_warning {
                raw * 0.3
            } else {
                raw
            }
        };
        let risk_score = (base_risk + sustained_bonus).min(1.0);

        let phase = self.determine_phase(risk_score);

        // Storm timing.
        let time_since_detection = match phase {
            StormPhase::Active | StormPhase::Peak => {
                if self.storm_start.is_none() {
                    self.storm_start = Some(Instant::now());
                    tracing::error!("STORM DETECTED");
                }
                self.storm_start
                    .map(|t| t.elapsed())
                    .unwrap_or(Duration::ZERO)
            }
            StormPhase::Clear => {
                self.storm_start = None;
                Duration::ZERO
            }
            _ => {
                if matches!(
                    self.current_phase,
                    StormPhase::Active | StormPhase::Peak | StormPhase::Imminent
                ) && phase == StormPhase::Resolving
                {
                    tracing::warn!("storm beginning to resolve");
                }
                Duration::ZERO
            }
        };

        self.current_phase = phase;

        StormSignature {
            phase,
            risk_score,
            proportionality,
            proportionality_trend,
            response_acceleration,
            threat_level,
            response_level,
            duration_at_elevated_secs: duration_elevated.as_secs_f64(),
            time_since_detection_secs: time_since_detection.as_secs_f64(),
            self_damage_detected: dmg.detected,
            self_damage_sources: dmg.sources,
        }
    }

    /// Detect cascade patterns from a list of system events.
    ///
    /// Searches for failure events (`"error"`, `"failure"`, `"timeout"`) that
    /// occur within 5 seconds across multiple subsystems.
    pub fn detect_cascade(&mut self, events: &[SystemEvent]) -> Vec<CascadePattern> {
        let mut sorted = events.to_vec();
        sorted.sort_by(|a, b| a.timestamp.cmp(&b.timestamp));

        let failure_types = ["error", "failure", "timeout"];
        let failures: Vec<&SystemEvent> = sorted
            .iter()
            .filter(|e| failure_types.contains(&e.event_type.as_str()))
            .collect();

        let mut cascades = Vec::new();
        let window = Duration::from_secs(5);

        let n = failures.len();
        for i in 0..n {
            let event = failures[i];
            let window_end = event.timestamp + window;

            let mut affected: Vec<String> = Vec::new();
            for j in (i + 1)..n {
                let next = failures[j];
                if next.timestamp > window_end {
                    break;
                }
                if next.system != event.system {
                    affected.push(next.system.clone());
                }
            }

            if !affected.is_empty() {
                let delay_ms = if i + 1 < n {
                    failures[i + 1]
                        .timestamp
                        .duration_since(event.timestamp)
                        .as_secs_f64()
                        * 1000.0
                } else {
                    0.0
                };

                let amp = affected.len() as f64;
                cascades.push(CascadePattern {
                    source_system: event.system.clone(),
                    affected_systems: affected,
                    propagation_delay_ms: delay_ms,
                    amplification_factor: amp,
                });
                self.detected_cascades += 1;
            }
        }

        cascades
    }

    /// Snapshot of detector statistics.
    pub fn get_statistics(&self) -> serde_json::Value {
        serde_json::json!({
            "current_phase": self.current_phase,
            "history_samples": self.history.len(),
            "detected_cascades": self.detected_cascades,
        })
    }

    // =========================================================================
    // Private helpers
    // =========================================================================

    fn compute_proportionality(threat: f64, response: f64) -> f64 {
        if threat < 0.01 {
            if response > 0.0 { response } else { 1.0 }
        } else {
            response / threat
        }
    }

    fn record_sample(&mut self, threat: f64, response: f64, damage: f64, proportionality: f64) {
        let now = Instant::now();

        // Evict samples older than the history window.
        // Use checked_sub because tokio::time::Instant cannot be subtracted by
        // a Duration that exceeds elapsed time since process start.
        if let Some(cutoff) = now.checked_sub(self.history_window) {
            while matches!(self.history.front(), Some(s) if s.recorded_at < cutoff) {
                self.history.pop_front();
            }
        }

        // Hard capacity cap.
        if self.history.len() >= self.max_history_samples {
            self.history.pop_front();
        }

        self.history.push_back(Sample {
            threat,
            response,
            damage,
            proportionality,
            recorded_at: now,
        });
    }

    /// Collect the last `n` values of a named field into a Vec.
    fn last_n_values<F>(&self, n: usize, extractor: F) -> Vec<f64>
    where
        F: Fn(&Sample) -> f64,
    {
        let len = self.history.len();
        if len == 0 {
            return vec![];
        }
        let start = len.saturating_sub(n);
        self.history
            .iter()
            .skip(start)
            .map(|s| extractor(s))
            .collect()
    }

    fn calc_trend_proportionality(&self, window: usize) -> TrendLabel {
        let values = self.last_n_values(window, |s| s.proportionality);
        if values.len() < 2 {
            return TrendLabel::Stable;
        }
        let slope = linear_slope(&values);
        let y_mean = sample_mean(&values);
        let threshold = if y_mean.abs() > f64::EPSILON {
            y_mean.abs() * 0.05
        } else {
            0.01
        };

        if slope > threshold {
            TrendLabel::Increasing
        } else if slope < -threshold {
            TrendLabel::Decreasing
        } else {
            TrendLabel::Stable
        }
    }

    fn calc_acceleration_response(&self, half_window: usize) -> f64 {
        let values = self.last_n_values(half_window * 2, |s| s.response);
        let total = values.len();
        if total < 4 {
            return 0.0;
        }
        let mid = total / 2;
        let first = &values[..mid];
        let second = &values[mid..];
        if first.len() < 2 || second.len() < 2 {
            return 0.0;
        }
        let rate_first = (first[first.len() - 1] - first[0]) / first.len() as f64;
        let rate_second = (second[second.len() - 1] - second[0]) / second.len() as f64;
        rate_second - rate_first
    }

    fn check_self_damage(&self, response_level: f64) -> SelfDamageResult {
        let damages = self.last_n_values(5, |s| s.damage);
        if damages.len() < 5 {
            return SelfDamageResult::none();
        }

        // Damage trend: increasing while response is high?
        let dmg_slope = linear_slope(&damages);
        let dmg_mean = sample_mean(&damages);
        let threshold = if dmg_mean.abs() > f64::EPSILON {
            dmg_mean.abs() * 0.05
        } else {
            0.01
        };

        if dmg_slope > threshold && response_level > 50.0 {
            return SelfDamageResult {
                detected: true,
                sources: vec!["response_induced_damage".into()],
            };
        }

        // Pearson correlation between response and damage over last 10 samples.
        let responses = self.last_n_values(10, |s| s.response);
        let dmgs10 = self.last_n_values(10, |s| s.damage);
        if responses.len() >= 10 && pearson_correlation(&responses, &dmgs10) > 0.7 {
            return SelfDamageResult {
                detected: true,
                sources: vec!["correlated_response_damage".into()],
            };
        }

        SelfDamageResult::none()
    }

    fn calc_risk_score(
        &self,
        proportionality: f64,
        trend: TrendLabel,
        acceleration: f64,
        duration_elevated: Duration,
        self_damage: bool,
    ) -> f64 {
        let mut risk = 0.0_f64;

        // Factor 1: proportionality level (max 0.25)
        risk += if proportionality >= self.proportionality_storm {
            0.25
        } else if proportionality >= self.proportionality_critical {
            0.18
        } else if proportionality >= self.proportionality_warning {
            0.10
        } else {
            0.0
        };

        // Factor 2: proportionality trend (max 0.25)
        if trend == TrendLabel::Increasing {
            risk += 0.25;
        }

        // Factor 3: response acceleration (max 0.20)
        risk += if acceleration > self.acceleration_critical {
            0.20
        } else if acceleration > self.acceleration_warning {
            0.10
        } else {
            0.0
        };

        // Factor 4: duration at elevated (max 0.15)
        risk += if duration_elevated >= self.elevated_duration_critical {
            0.15
        } else if duration_elevated >= self.elevated_duration_warning {
            0.08
        } else {
            0.0
        };

        // Factor 5: self-damage (max 0.15)
        if self_damage {
            risk += 0.15;
        }

        risk.min(1.0)
    }

    fn determine_phase(&self, risk_score: f64) -> StormPhase {
        if risk_score >= 0.9 {
            StormPhase::Peak
        } else if risk_score >= 0.7 {
            StormPhase::Active
        } else if risk_score >= 0.5 {
            StormPhase::Imminent
        } else if risk_score >= 0.3 {
            StormPhase::Warning
        } else if risk_score >= 0.15 {
            StormPhase::Watching
        } else if matches!(
            self.current_phase,
            StormPhase::Active | StormPhase::Peak | StormPhase::Imminent
        ) {
            StormPhase::Resolving
        } else {
            StormPhase::Clear
        }
    }
}

// =============================================================================
// Math helpers (pub(crate) for use in prevention.rs tests)
// =============================================================================

/// Arithmetic mean of a slice; returns 0.0 for empty input.
fn sample_mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Linear regression slope via ordinary least squares.
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let x_mean = (n - 1) as f64 / 2.0;
    let y_mean = sample_mean(values);

    let numerator: f64 = values
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f64 - x_mean) * (y - y_mean))
        .sum();
    let denominator: f64 = (0..n).map(|i| (i as f64 - x_mean).powi(2)).sum();

    if denominator == 0.0 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Pearson product-moment correlation coefficient.
pub(crate) fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }
    let n = x.len();
    let mx = sample_mean(x);
    let my = sample_mean(y);
    let num: f64 = (0..n).map(|i| (x[i] - mx) * (y[i] - my)).sum();
    let dx: f64 = (0..n).map(|i| (x[i] - mx).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = (0..n).map(|i| (y[i] - my).powi(2)).sum::<f64>().sqrt();
    if dx == 0.0 || dy == 0.0 {
        0.0
    } else {
        num / (dx * dy)
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time;

    #[test]
    fn normal_conditions_returns_clear() {
        let mut d = StormDetector::default();
        let sig = d.evaluate(10.0, 12.0, 0.0, None);
        assert_eq!(sig.phase, StormPhase::Clear);
        assert!(sig.risk_score < 0.15);
    }

    #[test]
    fn zero_threat_proportionality_equals_response() {
        let mut d = StormDetector::default();
        let sig = d.evaluate(0.0, 5.0, 0.0, None);
        assert!((sig.proportionality - 5.0).abs() < 0.001);
    }

    #[test]
    fn high_proportionality_adds_risk() {
        let mut d = StormDetector::default();
        // 50/2 = 25 — exceeds storm threshold (10)
        let sig = d.evaluate(2.0, 50.0, 0.0, None);
        assert!(
            sig.risk_score >= 0.25,
            "expected >= 0.25, got {}",
            sig.risk_score
        );
    }

    #[test]
    fn risk_score_capped_at_one() {
        let mut d = StormDetector::default();
        for _ in 0..30 {
            d.evaluate(1.0, 100.0, 30.0, None);
        }
        let sig = d.evaluate(1.0, 100.0, 30.0, None);
        assert!(
            sig.risk_score <= 1.0,
            "risk exceeded 1.0: {}",
            sig.risk_score
        );
    }

    #[test]
    fn sustained_storm_produces_elevated_phase() {
        let mut d = StormDetector::default();
        for _ in 0..30 {
            d.evaluate(1.0, 100.0, 10.0, None);
        }
        let sig = d.evaluate(1.0, 100.0, 10.0, None);
        assert!(
            matches!(
                sig.phase,
                StormPhase::Active | StormPhase::Peak | StormPhase::Imminent | StormPhase::Warning
            ),
            "expected elevated phase, got {:?}",
            sig.phase
        );
    }

    #[test]
    fn phase_after_storm_resolves_gracefully() {
        let mut d = StormDetector::default();
        for _ in 0..30 {
            d.evaluate(1.0, 100.0, 10.0, None);
        }
        let active = d.evaluate(1.0, 100.0, 10.0, None);
        assert!(matches!(
            active.phase,
            StormPhase::Active | StormPhase::Peak | StormPhase::Imminent | StormPhase::Warning
        ));
        let after = d.evaluate(10.0, 11.0, 0.0, None);
        assert!(
            matches!(
                after.phase,
                StormPhase::Resolving | StormPhase::Clear | StormPhase::Watching
            ),
            "expected post-storm phase, got {:?}",
            after.phase
        );
    }

    #[test]
    fn signature_round_trips_through_json() {
        let mut d = StormDetector::default();
        let sig = d.evaluate(5.0, 10.0, 0.0, None);
        let json = serde_json::to_string(&sig).unwrap();
        let back: StormSignature = serde_json::from_str(&json).unwrap();
        assert!((back.risk_score - sig.risk_score).abs() < 1e-9);
        assert_eq!(back.phase, sig.phase);
    }

    #[tokio::test]
    async fn elevated_duration_accumulates_over_time() {
        time::pause();
        let mut d = StormDetector::default();
        // Proportionality = 30 > 3.0 warning threshold.
        d.evaluate(1.0, 30.0, 0.0, None);
        time::advance(Duration::from_secs(400)).await;
        d.evaluate(1.0, 30.0, 0.0, None);
        time::advance(Duration::from_secs(400)).await;
        let sig = d.evaluate(1.0, 30.0, 0.0, None);
        assert!(
            sig.duration_at_elevated_secs > 0.0,
            "expected elevated duration > 0, got {}",
            sig.duration_at_elevated_secs
        );
    }

    #[test]
    fn detect_cascade_finds_cross_system_failures() {
        let mut d = StormDetector::default();
        let now = Instant::now();
        let events = vec![
            SystemEvent {
                system: "db".into(),
                event_type: "failure".into(),
                timestamp: now,
            },
            SystemEvent {
                system: "api".into(),
                event_type: "error".into(),
                timestamp: now + Duration::from_secs(1),
            },
            SystemEvent {
                system: "cache".into(),
                event_type: "timeout".into(),
                timestamp: now + Duration::from_secs(2),
            },
        ];
        let cascades = d.detect_cascade(&events);
        assert!(!cascades.is_empty(), "expected at least one cascade");
        assert_eq!(cascades[0].source_system, "db");
    }

    #[test]
    fn detect_cascade_ignores_same_system() {
        let mut d = StormDetector::default();
        let now = Instant::now();
        let events = vec![
            SystemEvent {
                system: "db".into(),
                event_type: "failure".into(),
                timestamp: now,
            },
            SystemEvent {
                system: "db".into(),
                event_type: "failure".into(),
                timestamp: now + Duration::from_secs(1),
            },
        ];
        let cascades = d.detect_cascade(&events);
        assert!(
            cascades.is_empty(),
            "same-system failures should not cascade"
        );
    }

    #[test]
    fn pearson_correlation_perfect_positive() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        assert!((pearson_correlation(&x, &y) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn pearson_correlation_flat_y_is_zero() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![3.0, 3.0, 3.0, 3.0, 3.0];
        assert_eq!(pearson_correlation(&x, &y), 0.0);
    }

    #[test]
    fn linear_slope_positive_series() {
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(linear_slope(&v) > 0.0);
    }

    #[test]
    fn linear_slope_flat_series_is_zero() {
        let v = vec![5.0, 5.0, 5.0, 5.0, 5.0];
        assert!(linear_slope(&v).abs() < 1e-9);
    }
}
