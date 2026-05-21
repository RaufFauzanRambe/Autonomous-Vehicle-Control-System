#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <deque>
#include <unordered_map>
#include <unordered_set>

namespace avcs {

struct RadarDetection {
    double range;        // meters
    double azimuth;      // radians
    double elevation;    // radians
    double range_rate;   // m/s (Doppler)
    double rcs;          // radar cross section
    double snr;          // signal-to-noise ratio
    int cluster_id;
    double timestamp;
};

struct RadarTrack {
    int track_id;
    Eigen::Vector3d position;   // [x, y, z] in vehicle frame
    Eigen::Vector3d velocity;   // [vx, vy, vz]
    Eigen::Matrix3d covariance;
    int age;
    int hits;
    int miss_streak;
    double last_timestamp;
    enum State { TENTATIVE, CONFIRMED, COASTING, LOST } state;
};

class RadarProcessor {
public:
    RadarProcessor(double max_range = 200.0,
                   double max_azimuth = M_PI/3,
                   double range_resolution = 0.5,
                   double velocity_resolution = 0.2,
                   int min_hits_to_confirm = 3,
                   int max_coasting_frames = 5,
                   double association_threshold = 3.0)
        : max_range_(max_range),
          max_azimuth_(max_azimuth),
          range_resolution_(range_resolution),
          velocity_resolution_(velocity_resolution),
          min_hits_to_confirm_(min_hits_to_confirm),
          max_coasting_frames_(max_coasting_frames),
          association_threshold_(association_threshold),
          next_track_id_(0)
    {}

    // Main pipeline: chain all processing steps
    std::vector<RadarTrack> process(const std::vector<RadarDetection>& detections, double timestamp) {
        // Step 1: Filter invalid / low-quality detections
        std::vector<RadarDetection> valid_dets = filterDetections(detections);
        std::cout << "[RadarProcessor] Valid detections: " << valid_dets.size()
                  << " / " << detections.size() << std::endl;

        // Step 2: Convert to Cartesian coordinates for association
        std::vector<Eigen::Vector3d> cart_positions = convertToCartesian(valid_dets);

        // Step 3: Predict existing tracks forward to current timestamp
        double dt = 0.0;
        if (!tracks_.empty()) {
            // Use the oldest track's timestamp as reference, or just compute from first track
            auto it = tracks_.begin();
            dt = timestamp - it->second.last_timestamp;
            if (dt < 0) dt = 0.0; // guard against clock skew
            if (dt > 1.0) dt = 1.0; // cap large time gaps
        }
        predictTracks(dt);

        // Step 4: Associate detections with predicted track positions
        std::vector<std::pair<int,int>> associations = associateDetections(valid_dets);

        // Step 5: Update matched tracks with associated detections
        updateTracks(valid_dets, associations);

        // Step 6: Manage track lifecycle (confirm, coast, delete)
        manageTracks();

        // Step 7: Initialize new tracks for unassociated detections
        // Collect unassociated detection indices
        std::unordered_set<int> associated_det_indices;
        for (const auto& assoc : associations) {
            associated_det_indices.insert(assoc.second);
        }
        for (size_t i = 0; i < valid_dets.size(); ++i) {
            if (associated_det_indices.find(static_cast<int>(i)) == associated_det_indices.end()) {
                initializeNewTrack(valid_dets[i], timestamp);
            }
        }

        // Step 8: Return confirmed tracks
        return getConfirmedTracks();
    }

    // Filter out detections that are out of range, low SNR, or invalid
    std::vector<RadarDetection> filterDetections(const std::vector<RadarDetection>& detections) {
        std::vector<RadarDetection> result;
        result.reserve(detections.size());

        for (const auto& det : detections) {
            // Check range bounds
            if (det.range < 0.5 || det.range > max_range_) {
                continue;
            }
            // Check azimuth bounds
            if (std::abs(det.azimuth) > max_azimuth_) {
                continue;
            }
            // Check elevation bounds (reasonable limits)
            if (std::abs(det.elevation) > M_PI / 4.0) {
                continue;
            }
            // Check for valid range rate (max reasonable vehicle speed ~100 m/s)
            if (std::abs(det.range_rate) > 100.0) {
                continue;
            }
            // Check SNR threshold (minimum 3 dB to be considered valid)
            if (det.snr < 3.0) {
                continue;
            }
            // Check for NaN / Inf values
            if (!std::isfinite(det.range) || !std::isfinite(det.azimuth) ||
                !std::isfinite(det.elevation) || !std::isfinite(det.range_rate) ||
                !std::isfinite(det.rcs) || !std::isfinite(det.snr)) {
                continue;
            }
            result.push_back(det);
        }
        return result;
    }

    // Convert detections from polar (range, azimuth, elevation) to Cartesian (x, y, z)
    std::vector<Eigen::Vector3d> convertToCartesian(const std::vector<RadarDetection>& detections) {
        std::vector<Eigen::Vector3d> result;
        result.reserve(detections.size());

        for (const auto& det : detections) {
            Eigen::Vector3d cart = polarToCartesian(det.range, det.azimuth, det.elevation);
            result.push_back(cart);
        }
        return result;
    }

    // Predict all tracks forward using constant-velocity Kalman prediction
    void predictTracks(double dt) {
        if (dt <= 0.0) return;

        // Constant-velocity state transition matrix for [x, y, z] with velocity
        // State: [px, py, pz, vx, vy, vz] but we store position/velocity separately
        // Prediction: p_new = p_old + v * dt
        // Covariance prediction: P_new = F * P * F^T + Q
        // For simplicity with 3x3 position covariance:
        //   F = I + dt * 0 (since velocity is separate, position prediction is direct)
        // We use a process noise model that grows with time

        double q_pos = 0.5;    // position process noise variance (m^2)
        double q_vel = 1.0;    // velocity process noise variance (m^2/s^2)

        for (auto& kv : tracks_) {
            RadarTrack& track = kv.second;

            // Predict position using constant velocity model
            track.position += track.velocity * dt;

            // Grow covariance to account for prediction uncertainty
            // P_pred = P + Q_process
            // Process noise increases with dt
            Eigen::Matrix3d Q_process = Eigen::Matrix3d::Identity() * (q_pos + q_vel * dt * dt);
            track.covariance += Q_process;

            // Ensure covariance remains symmetric
            track.covariance = 0.5 * (track.covariance + track.covariance.transpose());

            // Ensure covariance is positive definite by clamping eigenvalues
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(track.covariance);
            Eigen::Vector3d eigenvalues = solver.eigenvalues();
            bool needs_fix = false;
            for (int i = 0; i < 3; ++i) {
                if (eigenvalues(i) < 0.01) {
                    eigenvalues(i) = 0.01;
                    needs_fix = true;
                }
            }
            if (needs_fix) {
                track.covariance = solver.eigenvectors() * eigenvalues.asDiagonal() * solver.eigenvectors().transpose();
            }

            track.age++;
            track.last_timestamp += dt;
        }
    }

    // Associate detections to existing tracks using nearest-neighbor with Mahalanobis gating
    // Returns pairs of (track_id, detection_index)
    std::vector<std::pair<int,int>> associateDetections(const std::vector<RadarDetection>& detections) {
        std::vector<std::pair<int,int>> associations;

        if (tracks_.empty() || detections.empty()) {
            return associations;
        }

        // Convert all detections to Cartesian
        std::vector<Eigen::Vector3d> det_positions = convertToCartesian(detections);

        // Compute cost matrix: Mahalanobis distance between each track and each detection
        int n_tracks = static_cast<int>(tracks_.size());
        int n_dets = static_cast<int>(detections.size());

        // Build ordered list of track ids
        std::vector<int> track_ids;
        track_ids.reserve(tracks_.size());
        for (const auto& kv : tracks_) {
            track_ids.push_back(kv.first);
        }

        // Compute all distances
        // Using a simple greedy nearest-neighbor approach
        std::vector<bool> track_matched(n_tracks, false);
        std::vector<bool> det_matched(n_dets, false);

        // Build list of (distance, track_idx, det_idx) for all valid pairs
        struct AssociationCandidate {
            double distance;
            int track_idx;
            int det_idx;
        };
        std::vector<AssociationCandidate> candidates;
        candidates.reserve(n_tracks * n_dets);

        for (int ti = 0; ti < n_tracks; ++ti) {
            const RadarTrack& track = tracks_.at(track_ids[ti]);
            // Only try to associate with tentative or confirmed tracks
            if (track.state == RadarTrack::LOST) continue;

            Eigen::Matrix3d cov_inv;
            bool invertible;
            track.covariance.computeInverseWithCheck(cov_inv, invertible);
            if (!invertible) {
                cov_inv = Eigen::Matrix3d::Identity(); // fallback
            }

            for (int di = 0; di < n_dets; ++di) {
                Eigen::Vector3d innovation = det_positions[di] - track.position;
                double mahal_dist = std::sqrt(innovation.transpose() * cov_inv * innovation);

                if (mahal_dist < association_threshold_) {
                    candidates.push_back({mahal_dist, ti, di});
                }
            }
        }

        // Sort candidates by distance (greedy best-first)
        std::sort(candidates.begin(), candidates.end(),
                  [](const AssociationCandidate& a, const AssociationCandidate& b) {
                      return a.distance < b.distance;
                  });

        // Greedy assignment: pick closest first, skip already matched
        for (const auto& cand : candidates) {
            if (track_matched[cand.track_idx] || det_matched[cand.det_idx]) {
                continue;
            }
            associations.push_back({track_ids[cand.track_idx], cand.det_idx});
            track_matched[cand.track_idx] = true;
            det_matched[cand.det_idx] = true;
        }

        return associations;
    }

    // Update matched tracks using Kalman update equations
    void updateTracks(const std::vector<RadarDetection>& detections,
                      const std::vector<std::pair<int,int>>& associations) {
        // Measurement model: z = H * x, where H maps state to measurement space
        // Since we store position and velocity separately and measurement is position:
        // H = I (3x3), innovation = z - position_predicted

        Eigen::Matrix3d R = Eigen::Matrix3d::Identity(); // measurement noise covariance
        R(0, 0) = range_resolution_ * range_resolution_;
        R(1, 1) = range_resolution_ * range_resolution_;
        R(2, 2) = range_resolution_ * range_resolution_;

        for (const auto& assoc : associations) {
            int track_id = assoc.first;
            int det_idx = assoc.second;

            auto it = tracks_.find(track_id);
            if (it == tracks_.end()) continue;

            RadarTrack& track = it->second;
            const RadarDetection& det = detections[det_idx];

            // Convert detection to Cartesian measurement
            Eigen::Vector3d z = polarToCartesian(det.range, det.azimuth, det.elevation);

            // Innovation (measurement residual)
            Eigen::Vector3d y = z - track.position;

            // Innovation covariance: S = H * P * H^T + R = P + R (since H = I)
            Eigen::Matrix3d S = track.covariance + R;

            // Kalman gain: K = P * H^T * S^{-1} = P * S^{-1}
            Eigen::Matrix3d S_inv;
            bool invertible;
            S.computeInverseWithCheck(S_inv, invertible);
            if (!invertible) continue;

            Eigen::Matrix3d K = track.covariance * S_inv;

            // State update: position
            track.position = track.position + K * y;

            // Velocity update: estimate velocity from Doppler and innovation
            // Use range rate to improve velocity estimate
            // v_radial = range_rate, project onto position direction for 3D velocity update
            Eigen::Vector3d pos_norm = track.position.normalized();
            Eigen::Vector3d doppler_velocity = pos_norm * det.range_rate;

            // Blend Doppler velocity with innovation-derived velocity
            double alpha = 0.7; // weight for Doppler measurement
            if (track.age > 2) {
                Eigen::Vector3d innovation_velocity = K * y / 0.1; // approximate dt
                track.velocity = (1.0 - alpha) * track.velocity + alpha * doppler_velocity +
                                 (1.0 - alpha) * innovation_velocity * 0.1;
            } else {
                // For new tracks, initialize velocity from Doppler
                track.velocity = doppler_velocity;
            }

            // Covariance update: P = (I - K * H) * P = (I - K) * P
            Eigen::Matrix3d I_K = Eigen::Matrix3d::Identity() - K;
            track.covariance = I_K * track.covariance;

            // Ensure symmetry
            track.covariance = 0.5 * (track.covariance + track.covariance.transpose());

            // Update track bookkeeping
            track.hits++;
            track.miss_streak = 0;
            track.last_timestamp = det.timestamp;

            // Update track state based on hit count
            if (track.state == RadarTrack::TENTATIVE && track.hits >= min_hits_to_confirm_) {
                track.state = RadarTrack::CONFIRMED;
                std::cout << "[RadarProcessor] Track " << track_id << " confirmed (hits="
                          << track.hits << ")" << std::endl;
            } else if (track.state == RadarTrack::COASTING) {
                track.state = RadarTrack::CONFIRMED;
            }
        }
    }

    // Manage track lifecycle: handle coasting, deletion, and state transitions
    void manageTracks() {
        std::vector<int> tracks_to_delete;

        for (auto& kv : tracks_) {
            RadarTrack& track = kv.second;

            switch (track.state) {
                case RadarTrack::TENTATIVE:
                    // Tentative tracks: delete if missed too many times
                    if (track.miss_streak > 2) {
                        tracks_to_delete.push_back(kv.first);
                        std::cout << "[RadarProcessor] Deleting tentative track "
                                  << kv.first << " (missed " << track.miss_streak << " times)" << std::endl;
                    }
                    break;

                case RadarTrack::CONFIRMED:
                    // Confirmed tracks: transition to coasting if missed
                    if (track.miss_streak > 0) {
                        track.state = RadarTrack::COASTING;
                        std::cout << "[RadarProcessor] Track " << kv.first
                                  << " entering coast mode (missed " << track.miss_streak << ")" << std::endl;
                    }
                    break;

                case RadarTrack::COASTING:
                    // Coasting tracks: delete if missed for too many frames
                    if (track.miss_streak > max_coasting_frames_) {
                        track.state = RadarTrack::LOST;
                        tracks_to_delete.push_back(kv.first);
                        std::cout << "[RadarProcessor] Deleting coasting track "
                                  << kv.first << " (missed " << track.miss_streak << " frames)" << std::endl;
                    } else {
                        // Grow covariance during coasting (increased uncertainty)
                        Eigen::Matrix3d coast_noise = Eigen::Matrix3d::Identity() * 2.0;
                        track.covariance += coast_noise;
                        track.covariance = 0.5 * (track.covariance + track.covariance.transpose());
                    }
                    break;

                case RadarTrack::LOST:
                    // Lost tracks should already be deleted; safety net
                    tracks_to_delete.push_back(kv.first);
                    break;
            }
        }

        // Delete tracks marked for removal
        for (int id : tracks_to_delete) {
            tracks_.erase(id);
        }
    }

    // Return all confirmed tracks
    std::vector<RadarTrack> getConfirmedTracks() const {
        std::vector<RadarTrack> confirmed;
        for (const auto& kv : tracks_) {
            if (kv.second.state == RadarTrack::CONFIRMED) {
                confirmed.push_back(kv.second);
            }
        }

        // Sort by track ID for deterministic output
        std::sort(confirmed.begin(), confirmed.end(),
                  [](const RadarTrack& a, const RadarTrack& b) {
                      return a.track_id < b.track_id;
                  });

        return confirmed;
    }

private:
    double max_range_, max_azimuth_;
    double range_resolution_, velocity_resolution_;
    int min_hits_to_confirm_, max_coasting_frames_;
    double association_threshold_;

    std::unordered_map<int, RadarTrack> tracks_;
    int next_track_id_;

    // Convert polar coordinates to Cartesian
    Eigen::Vector3d polarToCartesian(double range, double azimuth, double elevation) {
        double x = range * std::cos(elevation) * std::cos(azimuth);
        double y = range * std::cos(elevation) * std::sin(azimuth);
        double z = range * std::sin(elevation);
        return Eigen::Vector3d(x, y, z);
    }

    // Compute Mahalanobis distance between a measurement and a track
    double computeMahalanobisDistance(const Eigen::Vector3d& measurement, const RadarTrack& track) {
        Eigen::Vector3d innovation = measurement - track.position;
        Eigen::Matrix3d cov_inv;
        bool invertible;
        track.covariance.computeInverseWithCheck(cov_inv, invertible);
        if (!invertible) {
            // Fallback: use Euclidean distance if covariance is singular
            return innovation.norm();
        }
        double mahal_sq = innovation.transpose() * cov_inv * innovation;
        return std::sqrt(std::max(0.0, mahal_sq));
    }

    // Initialize a new track from an unassociated detection
    void initializeNewTrack(const RadarDetection& det, double timestamp) {
        RadarTrack track;
        track.track_id = next_track_id_++;
        track.position = polarToCartesian(det.range, det.azimuth, det.elevation);

        // Initialize velocity from Doppler range rate
        Eigen::Vector3d pos_norm = track.position.normalized();
        track.velocity = pos_norm * det.range_rate;

        // Initialize covariance with uncertainty based on measurement resolution
        track.covariance = Eigen::Matrix3d::Identity();
        track.covariance(0, 0) = range_resolution_ * range_resolution_ * 4.0;
        track.covariance(1, 1) = range_resolution_ * range_resolution_ * 4.0;
        track.covariance(2, 2) = range_resolution_ * range_resolution_ * 4.0;

        track.age = 0;
        track.hits = 1;
        track.miss_streak = 0;
        track.last_timestamp = timestamp;
        track.state = RadarTrack::TENTATIVE;

        tracks_[track.track_id] = track;
    }
};

} // namespace avcs
