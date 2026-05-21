#include <Eigen/Dense>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <unordered_map>
#include <queue>

namespace avcs {

struct Point3D {
    double x, y, z, intensity;
};

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey& o) const { return x==o.x && y==o.y && z==o.z; }
};

struct VoxelKeyHash {
    size_t operator()(const VoxelKey& k) const {
        return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1) ^ (std::hash<int>()(k.z) << 2);
    }
};

struct Cluster {
    std::vector<Point3D> points;
    Eigen::Vector3d centroid;
    Eigen::Vector3d dimensions;
    double mean_intensity;
    int num_points;
};

class LidarProcessor {
public:
    LidarProcessor(double voxel_size = 0.1,
                   double roi_x_min = 0, double roi_x_max = 70,
                   double roi_y_min = -40, double roi_y_max = 40,
                   double roi_z_min = -3, double roi_z_max = 1,
                   double ground_threshold = 0.2,
                   double cluster_tolerance = 0.5,
                   int min_cluster_size = 10,
                   int max_cluster_size = 5000,
                   double ego_radius = 1.5)
        : voxel_size_(voxel_size),
          roi_x_min_(roi_x_min), roi_x_max_(roi_x_max),
          roi_y_min_(roi_y_min), roi_y_max_(roi_y_max),
          roi_z_min_(roi_z_min), roi_z_max_(roi_z_max),
          ground_threshold_(ground_threshold),
          cluster_tolerance_(cluster_tolerance),
          min_cluster_size_(min_cluster_size),
          max_cluster_size_(max_cluster_size),
          ego_radius_(ego_radius)
    {}

    // Main pipeline: chain all processing steps
    std::vector<Cluster> process(const std::vector<Point3D>& raw_cloud) {
        std::vector<Point3D> cloud = removeInvalidPoints(raw_cloud);
        std::cout << "[LidarProcessor] After invalid removal: " << cloud.size() << " points" << std::endl;

        cloud = cropROI(cloud);
        std::cout << "[LidarProcessor] After ROI crop: " << cloud.size() << " points" << std::endl;

        cloud = removeEgoPoints(cloud);
        std::cout << "[LidarProcessor] After ego removal: " << cloud.size() << " points" << std::endl;

        cloud = removeGroundPlane(cloud);
        std::cout << "[LidarProcessor] After ground removal: " << cloud.size() << " points" << std::endl;

        cloud = voxelize(cloud);
        std::cout << "[LidarProcessor] After voxelization: " << cloud.size() << " points" << std::endl;

        std::vector<Cluster> clusters = clusterPoints(cloud);
        std::cout << "[LidarProcessor] Found " << clusters.size() << " clusters" << std::endl;

        return clusters;
    }

    // Remove NaN, infinite, and zero-distance points
    std::vector<Point3D> removeInvalidPoints(const std::vector<Point3D>& cloud) {
        std::vector<Point3D> result;
        result.reserve(cloud.size());

        for (const auto& pt : cloud) {
            if (!std::isfinite(pt.x) || !std::isfinite(pt.y) ||
                !std::isfinite(pt.z) || !std::isfinite(pt.intensity)) {
                continue;
            }
            double dist_sq = pt.x * pt.x + pt.y * pt.y + pt.z * pt.z;
            if (dist_sq < 1e-6) {
                continue; // skip origin-duplicate points
            }
            result.push_back(pt);
        }
        return result;
    }

    // Crop points to rectangular region of interest
    std::vector<Point3D> cropROI(const std::vector<Point3D>& cloud) {
        std::vector<Point3D> result;
        result.reserve(cloud.size());

        for (const auto& pt : cloud) {
            if (pt.x >= roi_x_min_ && pt.x <= roi_x_max_ &&
                pt.y >= roi_y_min_ && pt.y <= roi_y_max_ &&
                pt.z >= roi_z_min_ && pt.z <= roi_z_max_) {
                result.push_back(pt);
            }
        }
        return result;
    }

    // Remove points within the ego-vehicle radius (self-hit returns)
    std::vector<Point3D> removeEgoPoints(const std::vector<Point3D>& cloud) {
        std::vector<Point3D> result;
        result.reserve(cloud.size());
        double ego_r_sq = ego_radius_ * ego_radius_;

        for (const auto& pt : cloud) {
            double dist_sq = pt.x * pt.x + pt.y * pt.y;
            if (dist_sq > ego_r_sq) {
                result.push_back(pt);
            }
        }
        return result;
    }

    // RANSAC-like ground plane removal
    // Fits a plane z = a*x + b*y + c and removes inlier points below threshold
    std::vector<Point3D> removeGroundPlane(const std::vector<Point3D>& cloud) {
        if (cloud.size() < 3) {
            return cloud;
        }

        const int max_iterations = 200;
        const double distance_threshold = ground_threshold_;
        const int n_samples = 3; // minimum samples for a plane

        std::vector<int> best_inliers;
        std::uniform_int_distribution<int> dist(0, static_cast<int>(cloud.size()) - 1);

        // Simple deterministic seed for reproducibility in this context
        unsigned int seed = 42;
        auto rand_idx = [&]() -> int {
            seed = seed * 1103515245 + 12345;
            return static_cast<int>((seed / 65536) % static_cast<unsigned int>(cloud.size()));
        };

        for (int iter = 0; iter < max_iterations; ++iter) {
            // Pick 3 random distinct points
            int i1 = rand_idx();
            int i2 = rand_idx();
            int i3 = rand_idx();
            if (i1 == i2 || i2 == i3 || i1 == i3) continue;

            Eigen::Vector3d p1(cloud[i1].x, cloud[i1].y, cloud[i1].z);
            Eigen::Vector3d p2(cloud[i2].x, cloud[i2].y, cloud[i2].z);
            Eigen::Vector3d p3(cloud[i3].x, cloud[i3].y, cloud[i3].z);

            // Compute plane normal from cross product of two edge vectors
            Eigen::Vector3d v1 = p2 - p1;
            Eigen::Vector3d v2 = p3 - p1;
            Eigen::Vector3d normal = v1.cross(v2);

            double norm_len = normal.norm();
            if (norm_len < 1e-8) continue; // degenerate
            normal /= norm_len;

            // We want the normal pointing upward (positive z component)
            if (normal.z() < 0) {
                normal = -normal;
            }

            // Skip planes that are too steep (not ground-like)
            // Ground normal should be roughly vertical: normal.z() > cos(30deg) ~ 0.866
            if (normal.z() < 0.8) continue;

            double d = -normal.dot(p1); // plane equation: normal.x*x + normal.y*y + normal.z*z + d = 0

            // Count inliers
            std::vector<int> inliers;
            for (size_t i = 0; i < cloud.size(); ++i) {
                double distance = std::abs(normal.x() * cloud[i].x +
                                           normal.y() * cloud[i].y +
                                           normal.z() * cloud[i].z + d);
                if (distance < distance_threshold) {
                    inliers.push_back(static_cast<int>(i));
                }
            }

            if (inliers.size() > best_inliers.size()) {
                best_inliers = std::move(inliers);
            }
        }

        // Mark ground inliers for removal
        std::vector<bool> is_ground(cloud.size(), false);
        for (int idx : best_inliers) {
            is_ground[idx] = true;
        }

        std::vector<Point3D> result;
        result.reserve(cloud.size());
        for (size_t i = 0; i < cloud.size(); ++i) {
            if (!is_ground[i]) {
                result.push_back(cloud[i]);
            }
        }
        return result;
    }

    // Voxel grid downsampling using unordered_map with VoxelKey
    std::vector<Point3D> voxelize(const std::vector<Point3D>& cloud) {
        if (cloud.empty()) return {};

        std::unordered_map<VoxelKey, std::vector<Point3D>, VoxelKeyHash> voxel_map;

        // Assign each point to a voxel
        for (const auto& pt : cloud) {
            VoxelKey key;
            key.x = static_cast<int>(std::floor(pt.x / voxel_size_));
            key.y = static_cast<int>(std::floor(pt.y / voxel_size_));
            key.z = static_cast<int>(std::floor(pt.z / voxel_size_));
            voxel_map[key].push_back(pt);
        }

        // Compute centroid of each voxel
        std::vector<Point3D> result;
        result.reserve(voxel_map.size());

        for (const auto& kv : voxel_map) {
            const std::vector<Point3D>& voxel_pts = kv.second;
            Point3D centroid_pt = {0, 0, 0, 0};
            for (const auto& vp : voxel_pts) {
                centroid_pt.x += vp.x;
                centroid_pt.y += vp.y;
                centroid_pt.z += vp.z;
                centroid_pt.intensity += vp.intensity;
            }
            double n = static_cast<double>(voxel_pts.size());
            centroid_pt.x /= n;
            centroid_pt.y /= n;
            centroid_pt.z /= n;
            centroid_pt.intensity /= n;
            result.push_back(centroid_pt);
        }
        return result;
    }

    // DBSCAN-based clustering
    std::vector<Cluster> clusterPoints(const std::vector<Point3D>& cloud) {
        if (cloud.empty()) return {};

        std::vector<int> labels = dbscan(cloud);

        // Collect points by cluster label
        std::unordered_map<int, std::vector<Point3D>> cluster_map;
        for (size_t i = 0; i < cloud.size(); ++i) {
            int label = labels[i];
            if (label > 0) { // skip noise (label == -1) and unvisited (label == 0)
                cluster_map[label].push_back(cloud[i]);
            }
        }

        // Build cluster objects with computed properties
        std::vector<Cluster> clusters;
        for (auto& kv : cluster_map) {
            std::vector<Point3D>& pts = kv.second;
            int n_pts = static_cast<int>(pts.size());

            // Filter by cluster size
            if (n_pts < min_cluster_size_ || n_pts > max_cluster_size_) {
                continue;
            }

            Cluster cluster;
            cluster.points = std::move(pts);
            cluster.num_points = n_pts;

            // Compute centroid
            Eigen::Vector3d sum = Eigen::Vector3d::Zero();
            double intensity_sum = 0.0;
            for (const auto& pt : cluster.points) {
                sum += Eigen::Vector3d(pt.x, pt.y, pt.z);
                intensity_sum += pt.intensity;
            }
            cluster.centroid = sum / n_pts;
            cluster.mean_intensity = intensity_sum / n_pts;

            // Compute axis-aligned bounding box dimensions
            double min_x = std::numeric_limits<double>::max();
            double max_x = std::numeric_limits<double>::lowest();
            double min_y = std::numeric_limits<double>::max();
            double max_y = std::numeric_limits<double>::lowest();
            double min_z = std::numeric_limits<double>::max();
            double max_z = std::numeric_limits<double>::lowest();

            for (const auto& pt : cluster.points) {
                min_x = std::min(min_x, pt.x); max_x = std::max(max_x, pt.x);
                min_y = std::min(min_y, pt.y); max_y = std::max(max_y, pt.y);
                min_z = std::min(min_z, pt.z); max_z = std::max(max_z, pt.z);
            }
            cluster.dimensions = Eigen::Vector3d(max_x - min_x, max_y - min_y, max_z - min_z);

            clusters.push_back(std::move(cluster));
        }

        // Sort clusters by number of points descending
        std::sort(clusters.begin(), clusters.end(),
                  [](const Cluster& a, const Cluster& b) { return a.num_points > b.num_points; });

        return clusters;
    }

private:
    double voxel_size_;
    double roi_x_min_, roi_x_max_, roi_y_min_, roi_y_max_, roi_z_min_, roi_z_max_;
    double ground_threshold_;
    double cluster_tolerance_;
    int min_cluster_size_, max_cluster_size_;
    double ego_radius_;

    // DBSCAN implementation
    // Returns label per point: -1 = noise, 0 = unvisited, >0 = cluster id
    std::vector<int> dbscan(const std::vector<Point3D>& cloud) {
        const int UNVISITED = 0;
        const int NOISE = -1;
        const int min_pts = min_cluster_size_;
        const double eps = cluster_tolerance_;
        const double eps_sq = eps * eps;

        int n = static_cast<int>(cloud.size());
        std::vector<int> labels(n, UNVISITED);
        int cluster_id = 0;

        for (int i = 0; i < n; ++i) {
            if (labels[i] != UNVISITED) continue;

            std::vector<int> neighbors = regionQuery(cloud, i, eps);

            if (static_cast<int>(neighbors.size()) < min_pts) {
                labels[i] = NOISE;
                continue;
            }

            cluster_id++;
            labels[i] = cluster_id;

            // BFS expansion of the cluster
            std::queue<int> seed_queue;
            for (int nb : neighbors) {
                if (nb != i) seed_queue.push(nb);
            }

            while (!seed_queue.empty()) {
                int curr = seed_queue.front();
                seed_queue.pop();

                if (labels[curr] == NOISE) {
                    labels[curr] = cluster_id; // reassign noise to cluster
                }
                if (labels[curr] != UNVISITED) {
                    continue;
                }

                labels[curr] = cluster_id;
                std::vector<int> curr_neighbors = regionQuery(cloud, curr, eps);

                if (static_cast<int>(curr_neighbors.size()) >= min_pts) {
                    for (int nb : curr_neighbors) {
                        if (labels[nb] == UNVISITED || labels[nb] == NOISE) {
                            seed_queue.push(nb);
                        }
                    }
                }
            }
        }
        return labels;
    }

    // Find all points within eps distance of point at index idx
    std::vector<int> regionQuery(const std::vector<Point3D>& cloud, int idx, double eps) {
        std::vector<int> neighbors;
        double eps_sq = eps * eps;
        double cx = cloud[idx].x;
        double cy = cloud[idx].y;
        double cz = cloud[idx].z;

        for (int i = 0; i < static_cast<int>(cloud.size()); ++i) {
            double dx = cloud[i].x - cx;
            double dy = cloud[i].y - cy;
            double dz = cloud[i].z - cz;
            double dist_sq = dx * dx + dy * dy + dz * dz;
            if (dist_sq <= eps_sq) {
                neighbors.push_back(i);
            }
        }
        return neighbors;
    }
};

} // namespace avcs
