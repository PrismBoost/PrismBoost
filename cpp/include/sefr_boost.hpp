#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace sefrboost {

constexpr double kEps = 1e-12;
constexpr double kMaxNewtonLeaf = 20.0;

enum class SplitMode : int {
    SefrOnly = 0,
    AxisFallback = 1,
    HybridSampled = 2,
    Hybrid = 3,
};

SplitMode parse_split_mode(const std::string& mode);

struct TreeNode {
    bool is_leaf = true;
    double value = 0.0;
    std::vector<double> coef;
    double intercept = 0.0;
    int left = -1;
    int right = -1;
};

struct Tree {
    std::vector<TreeNode> nodes;
    int root = -1;
};

struct ClassifierModel {
    int n_features = 0;
    int n_classes = 2;
    double learning_rate = 0.1;
    double mc_leaf_scale = 1.0;
    std::vector<double> init_score;  // scalar for binary, K for multiclass
    std::vector<int64_t> classes;
    // binary/regression: trees_[t]; multiclass: trees_[t][k]
    std::vector<Tree> trees_flat;
    bool multiclass = false;
    int trees_per_stage = 1;
};

struct RegressorModel {
    int n_features = 0;
    double learning_rate = 0.1;
    double init_score = 0.0;
    std::vector<Tree> trees;
};

double newton_leaf_value(
    const double* residuals,
    const double* p,
    const double* weights,
    int n
);

double mse_leaf_value(
    const double* residuals,
    const double* weights,
    int n
);

void fit_sefr_linear(
    const double* X,
    int n,
    int p,
    const int* y_bin,
    const double* sample_weight,
    std::vector<double>& coef_out,
    double& intercept_out
);

std::pair<double, double> best_split_threshold(
    const double* proj,
    const double* residuals,
    const double* sample_weight,
    const double* hessian,
    int n,
    int min_samples_leaf
);

std::tuple<int, double, double> best_axis_split(
    const double* X,
    int n,
    int p,
    const double* residuals,
    const double* sample_weight,
    const double* hessian,
    int min_samples_leaf,
    const int* feature_indices,
    int n_features_subset
);

Tree grow_tree(
    const double* X,
    int n_samples,
    int n_features,
    const double* residuals,
    const double* p,
    const double* sample_weight,
    const std::vector<int>& idx,
    int depth,
    int max_depth,
    int min_samples_leaf,
    int min_samples_split,
    bool regression,
    SplitMode split_mode,
    std::mt19937& rng
);

std::vector<double> predict_tree(
    const Tree& tree,
    const double* X,
    int n_samples,
    int n_features
);

ClassifierModel fit_classifier(
    const double* X,
    int n_samples,
    int n_features,
    const int64_t* y_idx,
    const double* sample_weight,
    int n_classes,
    int n_estimators,
    double learning_rate,
    int max_depth,
    int min_samples_leaf,
    int min_samples_split,
    double subsample,
    SplitMode split_mode,
    uint32_t random_state
);

RegressorModel fit_regressor(
    const double* X,
    int n_samples,
    int n_features,
    const double* y,
    const double* sample_weight,
    int n_estimators,
    double learning_rate,
    int max_depth,
    int min_samples_leaf,
    int min_samples_split,
    double subsample,
    SplitMode split_mode,
    uint32_t random_state
);

std::vector<double> predict_classifier_proba_pos(
    const ClassifierModel& model,
    const double* X,
    int n_samples
);

std::vector<std::vector<double>> predict_classifier_proba_multiclass(
    const ClassifierModel& model,
    const double* X,
    int n_samples
);

std::vector<double> predict_classifier_decision_binary(
    const ClassifierModel& model,
    const double* X,
    int n_samples
);

std::vector<std::vector<double>> predict_classifier_decision_multiclass(
    const ClassifierModel& model,
    const double* X,
    int n_samples
);

std::vector<double> predict_regressor(
    const RegressorModel& model,
    const double* X,
    int n_samples
);

std::size_t tree_size_bytes(const Tree& tree);

std::size_t classifier_model_size_bytes(const ClassifierModel& model);

std::size_t regressor_model_size_bytes(const RegressorModel& model);

}  // namespace sefrboost
