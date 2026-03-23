//! `StormBreaker` — integrated storm-breaking protocol.
//!
//! Combines circuit breakers and rate limiters to interrupt cascading
//! failure amplification when a storm is detected.
//!
//! ## T1 Grounding
//!
//! - `∂` (Boundary) — enforces loop-gain ceiling across all registered components
//! - `→` (Causality) — breaks runaway causal chains before they saturate
//! - `ς` (State) — tracks pre-storm rates for gradual recovery

use crate::prevention::{CircuitBreaker, RateLimiter};
use nexcore_homeostasis_primitives::enums::CircuitState;
use serde_json::Value;
use std::collections::HashMap;
use tokio::time::Instant;
use tracing::{error, warn};

/// Integrated storm-breaking system.
///
/// Registers [`CircuitBreaker`] and [`RateLimiter`] components, then applies a
/// coordinated dampening protocol when a cascade storm is detected.
///
/// Biological analog: Corticosteroids — broad systemic immunosuppression during
/// a cytokine storm.
///
/// # Protocol
///
/// On [`activate_storm_protocol`](StormBreaker::activate_storm_protocol):
/// 1. Force-opens every registered circuit breaker.
/// 2. Saves original rate-limiter rates and reduces them to
///    `emergency_dampening_factor` (default 10%).
///
/// On [`deactivate_storm_protocol`](StormBreaker::deactivate_storm_protocol):
/// 1. Restores rate limiters to 50% of their pre-storm rates.
/// 2. Closes all circuit breakers (they re-open automatically on further failures).
///
/// # Example
///
/// ```rust,no_run
/// # tokio_test::block_on(async {
/// use nexcore_homeostasis_storm::breaker::StormBreaker;
/// use nexcore_homeostasis_storm::prevention::{CircuitBreaker, RateLimiter};
///
/// let mut sb = StormBreaker::default();
/// sb.add_circuit("db", CircuitBreaker::with_defaults("db"));
/// sb.add_limiter("writes", RateLimiter::new("writes", 100.0, 1000.0));
///
/// sb.activate_storm_protocol().await;
/// assert!(sb.is_active());
///
/// sb.deactivate_storm_protocol().await;
/// assert!(!sb.is_active());
/// # });
/// ```
pub struct StormBreaker {
    /// Fraction of original rate applied during storm protocol (default `0.1` = 10%).
    pub emergency_dampening_factor: f64,
    circuit_breakers: HashMap<String, CircuitBreaker>,
    rate_limiters: HashMap<String, RateLimiter>,
    storm_protocol_active: bool,
    storm_start: Option<Instant>,
    pre_storm_rates: HashMap<String, f64>,
}

impl Default for StormBreaker {
    fn default() -> Self {
        Self::new(0.1)
    }
}

impl StormBreaker {
    /// Create a new `StormBreaker`.
    ///
    /// `emergency_dampening_factor` — fraction of original rate applied when the
    /// storm protocol activates (e.g. `0.1` = 10%).
    #[must_use]
    pub fn new(emergency_dampening_factor: f64) -> Self {
        Self {
            emergency_dampening_factor,
            circuit_breakers: HashMap::new(),
            rate_limiters: HashMap::new(),
            storm_protocol_active: false,
            storm_start: None,
            pre_storm_rates: HashMap::new(),
        }
    }

    /// Register a circuit breaker under `name`.
    pub fn add_circuit(&mut self, name: impl Into<String>, circuit: CircuitBreaker) {
        let n = name.into();
        tracing::info!(circuit = %n, "circuit registered");
        self.circuit_breakers.insert(n, circuit);
    }

    /// Register a rate limiter under `name`.
    pub fn add_limiter(&mut self, name: impl Into<String>, limiter: RateLimiter) {
        let n = name.into();
        tracing::info!(limiter = %n, "rate limiter registered");
        self.rate_limiters.insert(n, limiter);
    }

    /// Activate the storm-breaking protocol.
    ///
    /// Idempotent: calling this while already active is a no-op.
    pub async fn activate_storm_protocol(&mut self) {
        if self.storm_protocol_active {
            warn!("storm protocol already active — ignoring");
            return;
        }
        error!("STORM PROTOCOL ACTIVATED");
        self.storm_protocol_active = true;
        self.storm_start = Some(Instant::now());

        // Step 1 — force-open all circuits.
        for circuit in self.circuit_breakers.values() {
            circuit.force_open().await;
        }

        // Step 2 — snapshot current rates then reduce to emergency level.
        // Collect first to avoid simultaneous mutable + immutable borrows.
        let snapshots: Vec<(String, f64)> = {
            let mut v = Vec::with_capacity(self.rate_limiters.len());
            for (name, limiter) in &self.rate_limiters {
                v.push((name.clone(), limiter.tokens_per_second().await));
            }
            v
        };
        let factor = self.emergency_dampening_factor;
        for (name, original_rate) in snapshots {
            self.pre_storm_rates.insert(name.clone(), original_rate);
            if let Some(limiter) = self.rate_limiters.get(&name) {
                limiter.set_rate(original_rate * factor).await;
            }
        }
    }

    /// Deactivate the storm protocol and begin gradual recovery.
    ///
    /// Calling this when inactive is a no-op.
    pub async fn deactivate_storm_protocol(&mut self) {
        if !self.storm_protocol_active {
            return;
        }
        warn!("storm protocol deactivating — beginning recovery");

        // Restore rate limiters to 50% of pre-storm rates.
        for (name, limiter) in &self.rate_limiters {
            if let Some(&original) = self.pre_storm_rates.get(name) {
                limiter.set_rate(original * 0.5).await;
            }
        }

        // Close circuits (they re-open automatically on subsequent failures).
        for circuit in self.circuit_breakers.values() {
            circuit.force_close().await;
        }

        let duration_secs = self
            .storm_start
            .map(|t| t.elapsed().as_secs_f64())
            .unwrap_or(0.0);
        warn!(duration_secs, "storm protocol deactivated");

        self.storm_protocol_active = false;
        self.pre_storm_rates.clear();
    }

    /// Force-open a single named circuit (surgical intervention).
    ///
    /// No-op if the circuit is not registered.
    pub async fn quick_break(&self, name: &str) {
        if let Some(circuit) = self.circuit_breakers.get(name) {
            warn!(circuit = name, "quick break activated");
            circuit.force_open().await;
        }
    }

    /// Reduce a specific rate limiter by `factor` (e.g. `0.5` halves the rate).
    ///
    /// No-op if the limiter is not registered.
    pub async fn throttle(&self, name: &str, factor: f64) {
        if let Some(limiter) = self.rate_limiters.get(name) {
            let current = limiter.tokens_per_second().await;
            limiter.set_rate(current * factor).await;
            warn!(limiter = name, factor, "throttled");
        }
    }

    /// Whether the storm protocol is currently active.
    pub fn is_active(&self) -> bool {
        self.storm_protocol_active
    }

    /// Status snapshot as a JSON value.
    pub async fn get_status(&self) -> Value {
        let mut circuits = serde_json::Map::new();
        for (name, circuit) in &self.circuit_breakers {
            circuits.insert(name.clone(), circuit.get_statistics().await);
        }
        let mut limiters = serde_json::Map::new();
        for (name, limiter) in &self.rate_limiters {
            limiters.insert(name.clone(), limiter.get_statistics().await);
        }
        serde_json::json!({
            "storm_protocol_active": self.storm_protocol_active,
            "storm_duration_secs": self.storm_start
                .map(|t| t.elapsed().as_secs_f64())
                .unwrap_or(0.0),
            "circuit_breakers": Value::Object(circuits),
            "rate_limiters": Value::Object(limiters),
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prevention::{CircuitBreaker, RateLimiter};
    use nexcore_homeostasis_primitives::enums::CircuitState;
    use std::time::Duration;

    fn make_breaker() -> StormBreaker {
        let mut sb = StormBreaker::default();
        sb.add_circuit(
            "db",
            CircuitBreaker::new("db", 5, 2, Duration::from_secs(30), 3),
        );
        sb.add_circuit(
            "api",
            CircuitBreaker::new("api", 5, 2, Duration::from_secs(30), 3),
        );
        sb.add_limiter("writes", RateLimiter::new("writes", 100.0, 1_000.0));
        sb.add_limiter("reads", RateLimiter::new("reads", 200.0, 2_000.0));
        sb
    }

    #[tokio::test]
    async fn breaker_starts_inactive() {
        let sb = StormBreaker::default();
        assert!(!sb.is_active());
    }

    #[tokio::test]
    async fn registered_components_appear_in_status() {
        let mut sb = StormBreaker::default();
        sb.add_circuit("db", CircuitBreaker::with_defaults("db"));
        sb.add_limiter("api", RateLimiter::new("api", 100.0, 500.0));
        let status = sb.get_status().await;
        assert!(status["circuit_breakers"]["db"].is_object());
        assert!(status["rate_limiters"]["api"].is_object());
    }

    #[tokio::test]
    async fn activate_opens_all_circuits() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        assert!(sb.is_active());
        assert_eq!(sb.circuit_breakers["db"].state().await, CircuitState::Open);
        assert_eq!(sb.circuit_breakers["api"].state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn activate_reduces_rate_limits_to_factor() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        // 100 tokens/s * 0.1 = 10 tokens/s
        let rate = sb.rate_limiters["writes"].tokens_per_second().await;
        assert!((rate - 10.0).abs() < 0.01, "expected 10, got {rate}");
    }

    #[tokio::test]
    async fn activate_is_idempotent() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        sb.activate_storm_protocol().await; // second call ignored
        assert!(sb.is_active());
    }

    #[tokio::test]
    async fn deactivate_closes_all_circuits() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        sb.deactivate_storm_protocol().await;
        assert!(!sb.is_active());
        assert_eq!(
            sb.circuit_breakers["db"].state().await,
            CircuitState::Closed
        );
        assert_eq!(
            sb.circuit_breakers["api"].state().await,
            CircuitState::Closed
        );
    }

    #[tokio::test]
    async fn deactivate_restores_rates_to_fifty_percent() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        sb.deactivate_storm_protocol().await;
        // Original 100 → reduced to 10 → restored to 50 (100 * 0.5)
        let rate = sb.rate_limiters["writes"].tokens_per_second().await;
        assert!((rate - 50.0).abs() < 0.01, "expected 50, got {rate}");
    }

    #[tokio::test]
    async fn deactivate_when_inactive_is_noop() {
        let mut sb = make_breaker();
        sb.deactivate_storm_protocol().await; // should not panic
        assert!(!sb.is_active());
    }

    #[tokio::test]
    async fn quick_break_opens_specific_circuit_only() {
        let mut sb = make_breaker();
        sb.quick_break("db").await;
        assert_eq!(sb.circuit_breakers["db"].state().await, CircuitState::Open);
        assert_eq!(
            sb.circuit_breakers["api"].state().await,
            CircuitState::Closed
        );
    }

    #[tokio::test]
    async fn quick_break_unknown_name_is_noop() {
        let mut sb = make_breaker();
        sb.quick_break("nonexistent").await; // must not panic
    }

    #[tokio::test]
    async fn throttle_reduces_specific_limiter() {
        let mut sb = make_breaker();
        sb.throttle("writes", 0.5).await;
        let rate = sb.rate_limiters["writes"].tokens_per_second().await;
        assert!((rate - 50.0).abs() < 0.01, "expected 50, got {rate}");
        // reads unaffected
        let reads_rate = sb.rate_limiters["reads"].tokens_per_second().await;
        assert!((reads_rate - 200.0).abs() < 0.01);
    }

    #[tokio::test]
    async fn get_status_contains_all_components_when_active() {
        let mut sb = make_breaker();
        sb.activate_storm_protocol().await;
        let status = sb.get_status().await;
        assert!(status["storm_protocol_active"].as_bool().unwrap());
        assert!(status["circuit_breakers"]["db"].is_object());
        assert!(status["circuit_breakers"]["api"].is_object());
        assert!(status["rate_limiters"]["writes"].is_object());
        assert!(status["rate_limiters"]["reads"].is_object());
    }

    #[tokio::test]
    async fn status_is_inactive_before_activation() {
        let sb = make_breaker();
        let status = sb.get_status().await;
        assert!(!status["storm_protocol_active"].as_bool().unwrap());
    }
}
