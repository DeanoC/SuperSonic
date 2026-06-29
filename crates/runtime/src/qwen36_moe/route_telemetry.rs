//! Pure Qwen3.6-MoE route telemetry and transition prediction contracts.

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExpertRoute {
    pub rank: usize,
    pub expert_idx: usize,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct MoeTransitionPredictor {
    top_k: usize,
    min_observations: u32,
    observations_by_previous_rank: Vec<u32>,
    repeated_current_by_previous_rank: Vec<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeTransitionCandidate {
    pub expert_idx: usize,
    pub previous_rank: usize,
    pub repeats: u32,
    pub observations: u32,
}

impl MoeTransitionCandidate {
    pub fn reuse_probability(self) -> f64 {
        if self.observations == 0 {
            0.0
        } else {
            self.repeats as f64 / self.observations as f64
        }
    }
}

impl MoeTransitionPredictor {
    pub fn new(top_k: usize, min_observations: u32) -> Self {
        Self {
            top_k,
            min_observations,
            observations_by_previous_rank: vec![0; top_k],
            repeated_current_by_previous_rank: vec![0; top_k],
        }
    }

    pub fn update(&mut self, routes: &[ExpertRoute], previous_routes: &[usize]) {
        for (previous_rank, &expert_idx) in previous_routes.iter().take(self.top_k).enumerate() {
            self.observations_by_previous_rank[previous_rank] =
                self.observations_by_previous_rank[previous_rank].saturating_add(1);
            if routes.iter().any(|route| route.expert_idx == expert_idx) {
                self.repeated_current_by_previous_rank[previous_rank] =
                    self.repeated_current_by_previous_rank[previous_rank].saturating_add(1);
            }
        }
    }

    pub fn scored_candidates(
        &self,
        previous_routes: &[usize],
        limit: usize,
    ) -> Vec<MoeTransitionCandidate> {
        let mut scored = Vec::new();
        for (previous_rank, &expert_idx) in previous_routes.iter().take(self.top_k).enumerate() {
            let observations = self.observations_by_previous_rank[previous_rank];
            if observations < self.min_observations {
                continue;
            }
            let repeats = self.repeated_current_by_previous_rank[previous_rank];
            if repeats == 0 {
                continue;
            }
            scored.push(MoeTransitionCandidate {
                expert_idx,
                previous_rank,
                repeats,
                observations,
            });
        }
        scored.sort_by(|a, b| {
            let lhs = (a.repeats as u64) * (b.observations as u64);
            let rhs = (b.repeats as u64) * (a.observations as u64);
            rhs.cmp(&lhs)
                .then_with(|| a.previous_rank.cmp(&b.previous_rank))
        });
        scored.into_iter().take(limit).collect()
    }

    pub fn candidates(&self, previous_routes: &[usize], limit: usize) -> Vec<usize> {
        self.scored_candidates(previous_routes, limit)
            .into_iter()
            .map(|candidate| candidate.expert_idx)
            .collect()
    }
}

#[derive(Debug, Clone)]
pub struct MoeRouteTelemetry {
    pub observations_by_rank: Vec<u64>,
    pub resident_before_by_rank: Vec<u64>,
    pub repeated_previous_by_rank: Vec<u64>,
    pub repeated_previous_rank_by_current_rank: Vec<Vec<u64>>,
    pub weight_sum_by_rank: Vec<f64>,
}

impl MoeRouteTelemetry {
    pub fn new(top_k: usize) -> Self {
        Self {
            observations_by_rank: vec![0; top_k],
            resident_before_by_rank: vec![0; top_k],
            repeated_previous_by_rank: vec![0; top_k],
            repeated_previous_rank_by_current_rank: vec![vec![0; top_k]; top_k],
            weight_sum_by_rank: vec![0.0; top_k],
        }
    }

    pub fn record_route_observation(&mut self, route: &ExpertRoute, previous_routes: &[usize]) {
        if route.rank >= self.observations_by_rank.len() {
            return;
        }
        self.observations_by_rank[route.rank] += 1;
        self.weight_sum_by_rank[route.rank] += route.weight as f64;
        if let Some(previous_rank) = previous_routes
            .iter()
            .position(|&expert_idx| expert_idx == route.expert_idx)
        {
            self.repeated_previous_by_rank[route.rank] += 1;
            if let Some(row) = self
                .repeated_previous_rank_by_current_rank
                .get_mut(route.rank)
            {
                if let Some(cell) = row.get_mut(previous_rank) {
                    *cell += 1;
                }
            }
        }
    }

    pub fn record_resident_before(&mut self, route_rank: usize) {
        if let Some(count) = self.resident_before_by_rank.get_mut(route_rank) {
            *count += 1;
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        fn probability(count: u64, observations: u64) -> f64 {
            if observations == 0 {
                0.0
            } else {
                count as f64 / observations as f64
            }
        }

        let avg_weight_by_rank: Vec<f64> = self
            .weight_sum_by_rank
            .iter()
            .zip(&self.observations_by_rank)
            .map(|(sum, count)| {
                if *count == 0 {
                    0.0
                } else {
                    sum / *count as f64
                }
            })
            .collect();
        let repeated_previous_probability_by_current_rank: Vec<f64> = self
            .repeated_previous_by_rank
            .iter()
            .zip(&self.observations_by_rank)
            .map(|(count, observations)| probability(*count, *observations))
            .collect();
        let same_rank_repeat_probability_by_rank: Vec<f64> = self
            .repeated_previous_rank_by_current_rank
            .iter()
            .enumerate()
            .map(|(rank, row)| {
                probability(
                    row.get(rank).copied().unwrap_or(0),
                    self.observations_by_rank.get(rank).copied().unwrap_or(0),
                )
            })
            .collect();
        let top_k = self.observations_by_rank.len();
        let mut repeated_current_by_previous_rank = vec![0u64; top_k];
        let mut best_previous_rank_by_current_rank = vec![None; top_k];
        let mut best_current_rank_by_previous_rank = vec![None; top_k];
        let mut best_transition: Option<(usize, usize, u64)> = None;
        for (current_rank, row) in self
            .repeated_previous_rank_by_current_rank
            .iter()
            .enumerate()
        {
            let mut best_previous: Option<(usize, u64)> = None;
            for (previous_rank, &count) in row.iter().enumerate() {
                if previous_rank < repeated_current_by_previous_rank.len() {
                    repeated_current_by_previous_rank[previous_rank] += count;
                }
                if count > 0
                    && best_previous
                        .map(|(_, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_previous = Some((previous_rank, count));
                }
                if count > 0
                    && best_transition
                        .map(|(_, _, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_transition = Some((current_rank, previous_rank, count));
                }
            }
            best_previous_rank_by_current_rank[current_rank] =
                best_previous.map(|(previous_rank, _)| previous_rank);
        }
        for previous_rank in 0..top_k {
            let mut best_current: Option<(usize, u64)> = None;
            for (current_rank, row) in self
                .repeated_previous_rank_by_current_rank
                .iter()
                .enumerate()
            {
                let count = row.get(previous_rank).copied().unwrap_or(0);
                if count > 0
                    && best_current
                        .map(|(_, best_count)| count > best_count)
                        .unwrap_or(true)
                {
                    best_current = Some((current_rank, count));
                }
            }
            best_current_rank_by_previous_rank[previous_rank] =
                best_current.map(|(current_rank, _)| current_rank);
        }
        let repeated_current_probability_by_previous_rank: Vec<f64> =
            repeated_current_by_previous_rank
                .iter()
                .zip(&self.observations_by_rank)
                .map(|(count, observations)| probability(*count, *observations))
                .collect();
        let best_transition_json = best_transition.map(|(current_rank, previous_rank, count)| {
            serde_json::json!({
                "current_rank": current_rank,
                "previous_rank": previous_rank,
                "count": count,
                "probability_by_current_rank": probability(
                    count,
                    self.observations_by_rank
                        .get(current_rank)
                        .copied()
                        .unwrap_or(0),
                ),
            })
        });
        serde_json::json!({
            "observations_by_rank": &self.observations_by_rank,
            "resident_before_by_rank": &self.resident_before_by_rank,
            "repeated_previous_by_rank": &self.repeated_previous_by_rank,
            "repeated_previous_probability_by_current_rank": repeated_previous_probability_by_current_rank,
            "repeated_previous_rank_by_current_rank": &self.repeated_previous_rank_by_current_rank,
            "same_rank_repeat_probability_by_rank": same_rank_repeat_probability_by_rank,
            "repeated_current_by_previous_rank": repeated_current_by_previous_rank,
            "repeated_current_probability_by_previous_rank": repeated_current_probability_by_previous_rank,
            "best_previous_rank_by_current_rank": best_previous_rank_by_current_rank,
            "best_current_rank_by_previous_rank": best_current_rank_by_previous_rank,
            "best_transition": best_transition_json,
            "avg_weight_by_rank": avg_weight_by_rank,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{ExpertRoute, MoeRouteTelemetry, MoeTransitionPredictor};

    #[test]
    fn transition_predictor_waits_for_warmup_and_scores_repeats() {
        let mut predictor = MoeTransitionPredictor::new(3, 2);
        let previous_routes = [10, 20, 30];
        let routes = [
            ExpertRoute {
                rank: 0,
                expert_idx: 20,
                weight: 0.5,
            },
            ExpertRoute {
                rank: 1,
                expert_idx: 99,
                weight: 0.25,
            },
        ];

        predictor.update(&routes, &previous_routes);
        assert!(predictor.candidates(&previous_routes, 2).is_empty());

        predictor.update(&routes, &previous_routes);
        assert_eq!(predictor.candidates(&previous_routes, 2), vec![20]);
        let scored = predictor.scored_candidates(&previous_routes, 2);
        assert_eq!(scored.len(), 1);
        assert_eq!(scored[0].expert_idx, 20);
        assert_eq!(scored[0].observations, 2);
        assert_eq!(scored[0].repeats, 2);
        assert_eq!(scored[0].reuse_probability(), 1.0);
    }

    #[test]
    fn route_telemetry_records_previous_rank_transition_matrix() {
        let mut telemetry = MoeRouteTelemetry::new(3);
        let previous_routes = [7, 11, 13];
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 0,
                expert_idx: 11,
                weight: 0.5,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 1,
                expert_idx: 7,
                weight: 0.25,
            },
            &previous_routes,
        );
        telemetry.record_route_observation(
            &ExpertRoute {
                rank: 2,
                expert_idx: 99,
                weight: 0.125,
            },
            &previous_routes,
        );

        assert_eq!(telemetry.observations_by_rank, vec![1, 1, 1]);
        assert_eq!(telemetry.repeated_previous_by_rank, vec![1, 1, 0]);
        assert_eq!(
            telemetry.repeated_previous_rank_by_current_rank,
            vec![vec![0, 1, 0], vec![1, 0, 0], vec![0, 0, 0]]
        );
        assert_eq!(
            telemetry
                .to_json()
                .get("best_previous_rank_by_current_rank")
                .unwrap(),
            &serde_json::json!([1, 0, null])
        );
    }
}
