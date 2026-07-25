#include "sefr_boost.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <tuple>

namespace sefrboost {

namespace {

constexpr double kSplitNone = std::numeric_limits<double>::quiet_NaN();

double clip(double x, double lo, double hi) {
    return std::max(lo, std::min(hi, x));
}

void sanitize_coef(std::vector<double>& coef) {
    for (double& c : coef) {
        if (!std::isfinite(c)) {
            c = 0.0;
        }
        c = clip(c, -1e4, 1e4);
    }
}

double dot_row(const double* x, const std::vector<double>& coef, int p) {
    double s = 0.0;
    for (int j = 0; j < p; ++j) {
        s += x[j] * coef[static_cast<size_t>(j)];
    }
    return s;
}

std::vector<int> axis_feature_subset(int n_features, std::mt19937& rng) {
    int k = std::max(1, static_cast<int>(std::sqrt(static_cast<double>(n_features))));
    k = std::min(k, n_features);
    std::vector<int> all(static_cast<size_t>(n_features));
    std::iota(all.begin(), all.end(), 0);
    std::shuffle(all.begin(), all.end(), rng);
    all.resize(static_cast<size_t>(k));
    return all;
}

}  // namespace

SplitMode parse_split_mode(const std::string& mode) {
    if (mode == "sefr_only") {
        return SplitMode::SefrOnly;
    }
    if (mode == "axis_fallback") {
        return SplitMode::AxisFallback;
    }
    if (mode == "hybrid_sampled") {
        return SplitMode::HybridSampled;
    }
    if (mode == "hybrid") {
        return SplitMode::Hybrid;
    }
    throw std::invalid_argument("split_mode must be one of sefr_only, axis_fallback, hybrid_sampled, hybrid");
}

double newton_leaf_value(
    const double* residuals,
    const double* p,
    const double* weights,
    int n
) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; ++i) {
        double h = std::max(p[i] * (1.0 - p[i]), 1e-10);
        num += weights[i] * residuals[i];
        den += weights[i] * h;
    }
    den += 1e-10;
    return clip(num / den, -kMaxNewtonLeaf, kMaxNewtonLeaf);
}

double mse_leaf_value(
    const double* residuals,
    const double* weights,
    int n
) {
    double num = 0.0;
    double den = 0.0;
    for (int i = 0; i < n; ++i) {
        num += weights[i] * residuals[i];
        den += weights[i];
    }
    den += 1e-10;
    return num / den;
}

void fit_sefr_linear(
    const double* X,
    int n,
    int p,
    const int* y_bin,
    const double* sample_weight,
    std::vector<double>& coef_out,
    double& intercept_out
) {
    coef_out.assign(static_cast<size_t>(p), 0.0);
    std::vector<double> avg_pos(static_cast<size_t>(p), 0.0);
    std::vector<double> avg_neg(static_cast<size_t>(p), 0.0);
    double w_pos = 0.0;
    double w_neg = 0.0;

    for (int i = 0; i < n; ++i) {
        const double* row = X + static_cast<size_t>(i) * static_cast<size_t>(p);
        const double w = sample_weight[i];
        if (y_bin[i] == 1) {
            w_pos += w;
            for (int j = 0; j < p; ++j) {
                avg_pos[static_cast<size_t>(j)] += w * row[j];
            }
        } else {
            w_neg += w;
            for (int j = 0; j < p; ++j) {
                avg_neg[static_cast<size_t>(j)] += w * row[j];
            }
        }
    }

    if (w_pos <= 0.0 || w_neg <= 0.0) {
        throw std::runtime_error("SEFR requires both classes present");
    }

    for (int j = 0; j < p; ++j) {
        avg_pos[static_cast<size_t>(j)] /= w_pos;
        avg_neg[static_cast<size_t>(j)] /= w_neg;
        const double denom = avg_pos[static_cast<size_t>(j)] + avg_neg[static_cast<size_t>(j)] + 1e-7;
        coef_out[static_cast<size_t>(j)] =
            (avg_pos[static_cast<size_t>(j)] - avg_neg[static_cast<size_t>(j)]) / denom;
    }

    double pos_score_sum = 0.0;
    double neg_score_sum = 0.0;
    int pos_count = 0;
    int neg_count = 0;
    for (int i = 0; i < n; ++i) {
        const double* row = X + static_cast<size_t>(i) * static_cast<size_t>(p);
        const double score = dot_row(row, coef_out, p);
        if (y_bin[i] == 1) {
            pos_score_sum += sample_weight[i] * score;
            ++pos_count;
        } else {
            neg_score_sum += sample_weight[i] * score;
            ++neg_count;
        }
    }
    const double pos_score_avg = pos_score_sum / w_pos;
    const double neg_score_avg = neg_score_sum / w_neg;
    const double bias =
        (static_cast<double>(neg_count) * pos_score_avg + static_cast<double>(pos_count) * neg_score_avg) /
        static_cast<double>(neg_count + pos_count);
    intercept_out = -bias;
}

std::pair<double, double> best_split_threshold(
    const double* proj,
    const double* residuals,
    const double* sample_weight,
    const double* hessian,
    int n,
    int min_samples_leaf
) {
    const int msl = std::max(1, min_samples_leaf);
    if (n < 2 * msl) {
        return {kSplitNone, -std::numeric_limits<double>::infinity()};
    }

    std::vector<int> order(static_cast<size_t>(n));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return proj[a] < proj[b];
    });

    std::vector<double> gr(static_cast<size_t>(n));
    std::vector<double> gh(static_cast<size_t>(n));
    double g_total = 0.0;
    double h_total = 0.0;
    for (int i = 0; i < n; ++i) {
        const int idx = order[static_cast<size_t>(i)];
        gr[static_cast<size_t>(i)] = sample_weight[idx] * residuals[idx];
        gh[static_cast<size_t>(i)] = sample_weight[idx] * hessian[idx];
        g_total += gr[static_cast<size_t>(i)];
        h_total += gh[static_cast<size_t>(i)];
    }
    h_total += kEps;

    int best = -1;
    double best_gain = -std::numeric_limits<double>::infinity();
    double cum_g = 0.0;
    double cum_h = 0.0;
    for (int i = 0; i < n - 1; ++i) {
        cum_g += gr[static_cast<size_t>(i)];
        cum_h += gh[static_cast<size_t>(i)];
        const int n_left = i + 1;
        const int n_right = n - n_left;
        if (n_left < msl || n_right < msl) {
            continue;
        }
        const int idx_a = order[static_cast<size_t>(i)];
        const int idx_b = order[static_cast<size_t>(i + 1)];
        if (!(proj[idx_a] < proj[idx_b])) {
            continue;
        }
        const double g_left = cum_g;
        const double h_left = cum_h + kEps;
        const double g_right = g_total - g_left;
        const double h_right = h_total - cum_h + kEps;
        const double gain =
            (g_left * g_left) / h_left + (g_right * g_right) / h_right -
            (g_total * g_total) / h_total;
        if (std::isfinite(gain) && gain > best_gain) {
            best_gain = gain;
            best = i;
        }
    }

    if (best < 0 || !(best_gain > 0.0)) {
        return {kSplitNone, best_gain};
    }
    const int idx_a = order[static_cast<size_t>(best)];
    const int idx_b = order[static_cast<size_t>(best + 1)];
    const double t = 0.5 * (proj[idx_a] + proj[idx_b]);
    return {t, best_gain};
}

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
) {
    const int msl = std::max(1, min_samples_leaf);
    if (n < 2 * msl) {
        return {-1, 0.0, -std::numeric_limits<double>::infinity()};
    }

    const int p_use = feature_indices ? n_features_subset : p;
    double g_total = 0.0;
    double h_total = 0.0;
    for (int i = 0; i < n; ++i) {
        g_total += sample_weight[i] * residuals[i];
        h_total += sample_weight[i] * hessian[i];
    }
    h_total += kEps;

    int best_j = -1;
    double best_thr = 0.0;
    double best_gain = -std::numeric_limits<double>::infinity();

    std::vector<int> order(static_cast<size_t>(n));
    std::vector<double> gr(static_cast<size_t>(n));
    std::vector<double> gh(static_cast<size_t>(n));

    for (int fj = 0; fj < p_use; ++fj) {
        const int col = feature_indices ? feature_indices[fj] : fj;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return X[static_cast<size_t>(a) * static_cast<size_t>(p) + static_cast<size_t>(col)] <
                   X[static_cast<size_t>(b) * static_cast<size_t>(p) + static_cast<size_t>(col)];
        });

        for (int i = 0; i < n; ++i) {
            const int idx = order[static_cast<size_t>(i)];
            gr[static_cast<size_t>(i)] = sample_weight[idx] * residuals[idx];
            gh[static_cast<size_t>(i)] = sample_weight[idx] * hessian[idx];
        }

        double cum_g = 0.0;
        double cum_h = 0.0;
        for (int i = 0; i < n - 1; ++i) {
            cum_g += gr[static_cast<size_t>(i)];
            cum_h += gh[static_cast<size_t>(i)];
            const int n_left = i + 1;
            const int n_right = n - n_left;
            if (n_left < msl || n_right < msl) {
                continue;
            }
            const int idx_a = order[static_cast<size_t>(i)];
            const int idx_b = order[static_cast<size_t>(i + 1)];
            const double va =
                X[static_cast<size_t>(idx_a) * static_cast<size_t>(p) + static_cast<size_t>(col)];
            const double vb =
                X[static_cast<size_t>(idx_b) * static_cast<size_t>(p) + static_cast<size_t>(col)];
            if (!(va < vb)) {
                continue;
            }
            const double g_left = cum_g;
            const double h_left = cum_h + kEps;
            const double g_right = g_total - g_left;
            const double h_right = h_total - cum_h + kEps;
            const double gain =
                (g_left * g_left) / h_left + (g_right * g_right) / h_right -
                (g_total * g_total) / h_total;
            if (std::isfinite(gain) && gain > best_gain) {
                best_gain = gain;
                best_j = col;
                best_thr = 0.5 * (va + vb);
            }
        }
    }

    if (best_j < 0 || !(best_gain > 0.0)) {
        return {-1, 0.0, best_gain};
    }
    return {best_j, best_thr, best_gain};
}

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
) {
    Tree tree;
    struct Frame {
        std::vector<int> idx;
        int depth;
        int parent;
        bool is_left;
    };

    tree.nodes.emplace_back();
    tree.root = 0;
    std::vector<Frame> stack;
    stack.push_back({idx, depth, 0, false});

    while (!stack.empty()) {
        Frame frame = std::move(stack.back());
        stack.pop_back();
        const int node_id = frame.parent;
        const int n = static_cast<int>(frame.idx.size());

        std::vector<double> r_n(static_cast<size_t>(n));
        std::vector<double> p_n(static_cast<size_t>(n));
        std::vector<double> w_n(static_cast<size_t>(n));
        std::vector<double> hess(static_cast<size_t>(n));
        bool all_pos = true;
        bool all_neg = true;
        int pos_count = 0;
        int neg_count = 0;

        for (int i = 0; i < n; ++i) {
            const int row = frame.idx[static_cast<size_t>(i)];
            r_n[static_cast<size_t>(i)] = residuals[row];
            p_n[static_cast<size_t>(i)] = p[row];
            w_n[static_cast<size_t>(i)] = sample_weight[row];
            if (r_n[static_cast<size_t>(i)] > 0.0) {
                all_neg = false;
                ++pos_count;
            }
            if (r_n[static_cast<size_t>(i)] < 0.0) {
                all_pos = false;
                ++neg_count;
            }
            hess[static_cast<size_t>(i)] =
                regression ? 1.0 : std::max(p_n[static_cast<size_t>(i)] * (1.0 - p_n[static_cast<size_t>(i)]), 1e-10);
        }

        auto make_leaf = [&]() {
            tree.nodes[static_cast<size_t>(node_id)].is_leaf = true;
            if (regression) {
                tree.nodes[static_cast<size_t>(node_id)].value =
                    mse_leaf_value(r_n.data(), w_n.data(), n);
            } else {
                tree.nodes[static_cast<size_t>(node_id)].value =
                    newton_leaf_value(r_n.data(), p_n.data(), w_n.data(), n);
            }
        };

        if (frame.depth >= max_depth || n < min_samples_split || all_pos || all_neg ||
            pos_count == 0 || neg_count == 0) {
            make_leaf();
            continue;
        }

        std::vector<double> rw(static_cast<size_t>(n));
        double rw_sum = 0.0;
        for (int i = 0; i < n; ++i) {
            rw[static_cast<size_t>(i)] = std::abs(r_n[static_cast<size_t>(i)]) * w_n[static_cast<size_t>(i)];
            rw_sum += rw[static_cast<size_t>(i)];
        }
        if (rw_sum <= 1e-15) {
            make_leaf();
            continue;
        }
        for (int i = 0; i < n; ++i) {
            rw[static_cast<size_t>(i)] /= rw_sum;
        }

        std::vector<int> y_bin(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            y_bin[static_cast<size_t>(i)] = r_n[static_cast<size_t>(i)] > 0.0 ? 1 : 0;
        }

        std::vector<double> lo(static_cast<size_t>(n_features), 0.0);
        std::vector<double> hi(static_cast<size_t>(n_features), 0.0);
        std::vector<char> nondegen(static_cast<size_t>(n_features), 0);
        for (int j = 0; j < n_features; ++j) {
            lo[static_cast<size_t>(j)] = std::numeric_limits<double>::infinity();
            hi[static_cast<size_t>(j)] = -std::numeric_limits<double>::infinity();
        }
        for (int i = 0; i < n; ++i) {
            const int row = frame.idx[static_cast<size_t>(i)];
            const double* xrow = X + static_cast<size_t>(row) * static_cast<size_t>(n_features);
            for (int j = 0; j < n_features; ++j) {
                const double v = xrow[j];
                lo[static_cast<size_t>(j)] = std::min(lo[static_cast<size_t>(j)], v);
                hi[static_cast<size_t>(j)] = std::max(hi[static_cast<size_t>(j)], v);
            }
        }
        for (int j = 0; j < n_features; ++j) {
            if (hi[static_cast<size_t>(j)] > lo[static_cast<size_t>(j)]) {
                nondegen[static_cast<size_t>(j)] = 1;
            }
        }

        double oblique_t = kSplitNone;
        double oblique_gain = -std::numeric_limits<double>::infinity();
        std::vector<double> coef(static_cast<size_t>(n_features), 0.0);
        std::vector<double> proj(static_cast<size_t>(n));

        bool any_nondegen = false;
        for (int j = 0; j < n_features; ++j) {
            if (nondegen[static_cast<size_t>(j)]) {
                any_nondegen = true;
                break;
            }
        }

        if (any_nondegen) {
            std::vector<double> X_fit(static_cast<size_t>(n) * static_cast<size_t>(n_features), 0.0);
            for (int i = 0; i < n; ++i) {
                const int row = frame.idx[static_cast<size_t>(i)];
                const double* xrow = X + static_cast<size_t>(row) * static_cast<size_t>(n_features);
                for (int j = 0; j < n_features; ++j) {
                    if (nondegen[static_cast<size_t>(j)]) {
                        const double span = hi[static_cast<size_t>(j)] - lo[static_cast<size_t>(j)];
                        X_fit[static_cast<size_t>(i) * static_cast<size_t>(n_features) + static_cast<size_t>(j)] =
                            (xrow[j] - lo[static_cast<size_t>(j)]) / span;
                    }
                }
            }

            std::vector<double> coef_scaled;
            double intercept0 = 0.0;
            try {
                fit_sefr_linear(
                    X_fit.data(),
                    n,
                    n_features,
                    y_bin.data(),
                    rw.data(),
                    coef_scaled,
                    intercept0
                );
                for (int j = 0; j < n_features; ++j) {
                    if (nondegen[static_cast<size_t>(j)]) {
                        const double span = hi[static_cast<size_t>(j)] - lo[static_cast<size_t>(j)];
                        coef[static_cast<size_t>(j)] = coef_scaled[static_cast<size_t>(j)] / span;
                    }
                }
                sanitize_coef(coef);
                for (int i = 0; i < n; ++i) {
                    const int row = frame.idx[static_cast<size_t>(i)];
                    const double* xrow = X + static_cast<size_t>(row) * static_cast<size_t>(n_features);
                    proj[static_cast<size_t>(i)] = dot_row(xrow, coef, n_features);
                }
                auto split = best_split_threshold(
                    proj.data(),
                    r_n.data(),
                    w_n.data(),
                    hess.data(),
                    n,
                    min_samples_leaf
                );
                oblique_t = split.first;
                oblique_gain = split.second;
            } catch (...) {
                oblique_t = kSplitNone;
                oblique_gain = -std::numeric_limits<double>::infinity();
            }
        }

        int axis_j = -1;
        double axis_thr = 0.0;
        double axis_gain = -std::numeric_limits<double>::infinity();
        if (split_mode != SplitMode::SefrOnly) {
            bool run_axis = split_mode == SplitMode::Hybrid || split_mode == SplitMode::HybridSampled ||
                            (split_mode == SplitMode::AxisFallback &&
                             (std::isnan(oblique_t) || oblique_gain <= 0.0));
            if (run_axis) {
                std::vector<int> feat_idx;
                const int* feat_ptr = nullptr;
                int feat_n = 0;
                if (split_mode == SplitMode::HybridSampled) {
                    feat_idx = axis_feature_subset(n_features, rng);
                    feat_ptr = feat_idx.data();
                    feat_n = static_cast<int>(feat_idx.size());
                }
                std::vector<double> X_n(static_cast<size_t>(n) * static_cast<size_t>(n_features));
                for (int i = 0; i < n; ++i) {
                    const int row = frame.idx[static_cast<size_t>(i)];
                    const double* xrow = X + static_cast<size_t>(row) * static_cast<size_t>(n_features);
                    std::copy(xrow, xrow + n_features, X_n.begin() + static_cast<size_t>(i) * static_cast<size_t>(n_features));
                }
                auto axis = best_axis_split(
                    X_n.data(),
                    n,
                    n_features,
                    r_n.data(),
                    w_n.data(),
                    hess.data(),
                    min_samples_leaf,
                    feat_ptr,
                    feat_n
                );
                axis_j = std::get<0>(axis);
                axis_thr = std::get<1>(axis);
                axis_gain = std::get<2>(axis);
            }
        }

        bool use_axis = false;
        if (split_mode == SplitMode::SefrOnly) {
            use_axis = false;
        } else if (split_mode == SplitMode::AxisFallback) {
            use_axis = axis_j >= 0 && std::isnan(oblique_t);
        } else {
            use_axis = axis_j >= 0 && (std::isnan(oblique_t) || axis_gain > oblique_gain);
        }

        double t_star = kSplitNone;
        if (use_axis) {
            std::fill(coef.begin(), coef.end(), 0.0);
            coef[static_cast<size_t>(axis_j)] = 1.0;
            t_star = axis_thr;
            for (int i = 0; i < n; ++i) {
                const int row = frame.idx[static_cast<size_t>(i)];
                proj[static_cast<size_t>(i)] =
                    X[static_cast<size_t>(row) * static_cast<size_t>(n_features) + static_cast<size_t>(axis_j)];
            }
        } else if (!std::isnan(oblique_t)) {
            t_star = oblique_t;
        } else {
            make_leaf();
            continue;
        }

        const double intercept = clip(-t_star, -1e6, 1e6);
        std::vector<int> idx_left;
        std::vector<int> idx_right;
        idx_left.reserve(static_cast<size_t>(n));
        idx_right.reserve(static_cast<size_t>(n));
        for (int i = 0; i < n; ++i) {
            if (proj[static_cast<size_t>(i)] <= t_star) {
                idx_left.push_back(frame.idx[static_cast<size_t>(i)]);
            } else {
                idx_right.push_back(frame.idx[static_cast<size_t>(i)]);
            }
        }

        if (static_cast<int>(idx_left.size()) < min_samples_leaf ||
            static_cast<int>(idx_right.size()) < min_samples_leaf ||
            idx_left.empty() || idx_right.empty()) {
            make_leaf();
            continue;
        }

        tree.nodes[static_cast<size_t>(node_id)].is_leaf = false;
        tree.nodes[static_cast<size_t>(node_id)].coef = coef;
        tree.nodes[static_cast<size_t>(node_id)].intercept = intercept;

        const int left_id = static_cast<int>(tree.nodes.size());
        tree.nodes.emplace_back();
        tree.nodes[static_cast<size_t>(node_id)].left = left_id;

        const int right_id = static_cast<int>(tree.nodes.size());
        tree.nodes.emplace_back();
        tree.nodes[static_cast<size_t>(node_id)].right = right_id;

        stack.push_back({std::move(idx_right), frame.depth + 1, right_id, false});
        stack.push_back({std::move(idx_left), frame.depth + 1, left_id, true});
    }

    return tree;
}

std::vector<double> predict_tree(
    const Tree& tree,
    const double* X,
    int n_samples,
    int n_features
) {
    std::vector<double> out(static_cast<size_t>(n_samples), 0.0);
    if (tree.root < 0 || tree.nodes.empty()) {
        return out;
    }

    for (int i = 0; i < n_samples; ++i) {
        const double* row = X + static_cast<size_t>(i) * static_cast<size_t>(n_features);
        int node_id = tree.root;
        while (!tree.nodes[static_cast<size_t>(node_id)].is_leaf) {
            const auto& node = tree.nodes[static_cast<size_t>(node_id)];
            const double score = dot_row(row, node.coef, n_features) + node.intercept;
            node_id = score <= 0.0 ? node.left : node.right;
        }
        out[static_cast<size_t>(i)] = tree.nodes[static_cast<size_t>(node_id)].value;
    }
    return out;
}

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
) {
    ClassifierModel model;
    model.n_features = n_features;
    model.n_classes = n_classes;
    model.learning_rate = learning_rate;
    model.multiclass = n_classes > 2;
    model.trees_per_stage = model.multiclass ? n_classes : 1;

    std::mt19937 rng(random_state);

    if (!model.multiclass) {
        double w_sum = 0.0;
        double pos_sum = 0.0;
        for (int i = 0; i < n_samples; ++i) {
            w_sum += sample_weight[i];
            pos_sum += sample_weight[i] * static_cast<double>(y_idx[i]);
        }
        w_sum += 1e-15;
        const double pos_rate = clip(pos_sum / w_sum, 1e-10, 1.0 - 1e-10);
        model.init_score = {std::log(pos_rate / (1.0 - pos_rate))};

        std::vector<double> F(static_cast<size_t>(n_samples), model.init_score[0]);
        model.trees_flat.reserve(static_cast<size_t>(n_estimators));

        for (int stage = 0; stage < n_estimators; ++stage) {
            std::vector<double> p(static_cast<size_t>(n_samples));
            std::vector<double> residuals(static_cast<size_t>(n_samples));
            for (int i = 0; i < n_samples; ++i) {
                p[static_cast<size_t>(i)] = 1.0 / (1.0 + std::exp(-F[static_cast<size_t>(i)]));
                p[static_cast<size_t>(i)] = clip(p[static_cast<size_t>(i)], 1e-10, 1.0 - 1e-10);
                residuals[static_cast<size_t>(i)] =
                    static_cast<double>(y_idx[i]) - p[static_cast<size_t>(i)];
            }

            std::vector<int> fit_idx;
            const double* X_fit = X;
            const double* r_fit = residuals.data();
            const double* p_fit = p.data();
            const double* w_fit = sample_weight;
            int n_fit = n_samples;

            std::vector<double> r_buf;
            std::vector<double> p_buf;
            std::vector<double> w_buf;
            if (subsample < 1.0) {
                const int n_sub = std::max(1, static_cast<int>(subsample * n_samples));
                fit_idx.resize(static_cast<size_t>(n_sub));
                std::vector<int> all(static_cast<size_t>(n_samples));
                std::iota(all.begin(), all.end(), 0);
                std::shuffle(all.begin(), all.end(), rng);
                for (int i = 0; i < n_sub; ++i) {
                    fit_idx[static_cast<size_t>(i)] = all[static_cast<size_t>(i)];
                }
                n_fit = n_sub;
                r_buf.resize(static_cast<size_t>(n_sub));
                p_buf.resize(static_cast<size_t>(n_sub));
                w_buf.resize(static_cast<size_t>(n_sub));
                for (int i = 0; i < n_sub; ++i) {
                    const int idx = fit_idx[static_cast<size_t>(i)];
                    r_buf[static_cast<size_t>(i)] = residuals[static_cast<size_t>(idx)];
                    p_buf[static_cast<size_t>(i)] = p[static_cast<size_t>(idx)];
                    w_buf[static_cast<size_t>(i)] = sample_weight[idx];
                }
                r_fit = r_buf.data();
                p_fit = p_buf.data();
                w_fit = w_buf.data();
            } else {
                fit_idx.resize(static_cast<size_t>(n_samples));
                std::iota(fit_idx.begin(), fit_idx.end(), 0);
            }

            Tree tree = grow_tree(
                X,
                n_samples,
                n_features,
                residuals.data(),
                p.data(),
                sample_weight,
                fit_idx,
                0,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                false,
                split_mode,
                rng
            );
            model.trees_flat.push_back(std::move(tree));

            const std::vector<double> pred =
                predict_tree(model.trees_flat.back(), X, n_samples, n_features);
            for (int i = 0; i < n_samples; ++i) {
                F[static_cast<size_t>(i)] += learning_rate * pred[static_cast<size_t>(i)];
            }
        }
        return model;
    }

    model.mc_leaf_scale = static_cast<double>(n_classes - 1) / static_cast<double>(n_classes);
    model.init_score.assign(static_cast<size_t>(n_classes), 0.0);
    double w_sum = 0.0;
    std::vector<double> class_w(static_cast<size_t>(n_classes), 0.0);
    for (int i = 0; i < n_samples; ++i) {
        w_sum += sample_weight[i];
        class_w[static_cast<size_t>(y_idx[i])] += sample_weight[i];
    }
    w_sum += 1e-15;
    for (int k = 0; k < n_classes; ++k) {
        const double prior = clip(class_w[static_cast<size_t>(k)] / w_sum, 1e-10, 1.0 - 1e-10);
        model.init_score[static_cast<size_t>(k)] = std::log(prior);
    }

    std::vector<std::vector<double>> F(
        static_cast<size_t>(n_samples),
        std::vector<double>(static_cast<size_t>(n_classes))
    );
    for (int i = 0; i < n_samples; ++i) {
        F[static_cast<size_t>(i)] = model.init_score;
    }

    model.trees_flat.reserve(static_cast<size_t>(n_estimators * n_classes));

    for (int stage = 0; stage < n_estimators; ++stage) {
        std::vector<std::vector<double>> P(
            static_cast<size_t>(n_samples),
            std::vector<double>(static_cast<size_t>(n_classes))
        );
        for (int i = 0; i < n_samples; ++i) {
            double fmax = F[static_cast<size_t>(i)][0];
            for (int k = 1; k < n_classes; ++k) {
                fmax = std::max(fmax, F[static_cast<size_t>(i)][static_cast<size_t>(k)]);
            }
            double denom = 0.0;
            for (int k = 0; k < n_classes; ++k) {
                P[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                    std::exp(F[static_cast<size_t>(i)][static_cast<size_t>(k)] - fmax);
                denom += P[static_cast<size_t>(i)][static_cast<size_t>(k)];
            }
            for (int k = 0; k < n_classes; ++k) {
                P[static_cast<size_t>(i)][static_cast<size_t>(k)] /= denom;
                P[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                    clip(P[static_cast<size_t>(i)][static_cast<size_t>(k)], 1e-10, 1.0 - 1e-10);
            }
        }

        std::vector<int> fit_idx;
        if (subsample < 1.0) {
            const int n_sub = std::max(1, static_cast<int>(subsample * n_samples));
            fit_idx.resize(static_cast<size_t>(n_sub));
            std::vector<int> all(static_cast<size_t>(n_samples));
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            for (int i = 0; i < n_sub; ++i) {
                fit_idx[static_cast<size_t>(i)] = all[static_cast<size_t>(i)];
            }
        } else {
            fit_idx.resize(static_cast<size_t>(n_samples));
            std::iota(fit_idx.begin(), fit_idx.end(), 0);
        }

        for (int k = 0; k < n_classes; ++k) {
            std::vector<double> residuals(static_cast<size_t>(n_samples));
            std::vector<double> p_k(static_cast<size_t>(n_samples));
            for (int i = 0; i < n_samples; ++i) {
                const double yk = y_idx[i] == k ? 1.0 : 0.0;
                residuals[static_cast<size_t>(i)] = yk - P[static_cast<size_t>(i)][static_cast<size_t>(k)];
                p_k[static_cast<size_t>(i)] = P[static_cast<size_t>(i)][static_cast<size_t>(k)];
            }

            Tree tree = grow_tree(
                X,
                n_samples,
                n_features,
                residuals.data(),
                p_k.data(),
                sample_weight,
                fit_idx,
                0,
                max_depth,
                min_samples_leaf,
                min_samples_split,
                false,
                split_mode,
                rng
            );
            model.trees_flat.push_back(std::move(tree));

            const std::vector<double> pred =
                predict_tree(model.trees_flat.back(), X, n_samples, n_features);
            for (int i = 0; i < n_samples; ++i) {
                F[static_cast<size_t>(i)][static_cast<size_t>(k)] +=
                    learning_rate * model.mc_leaf_scale * pred[static_cast<size_t>(i)];
            }
        }
    }

    return model;
}

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
) {
    RegressorModel model;
    model.n_features = n_features;
    model.learning_rate = learning_rate;

    double w_sum = 0.0;
    double y_sum = 0.0;
    for (int i = 0; i < n_samples; ++i) {
        w_sum += sample_weight[i];
        y_sum += sample_weight[i] * y[i];
    }
    model.init_score = y_sum / (w_sum + 1e-15);

    std::vector<double> F(static_cast<size_t>(n_samples), model.init_score);
    model.trees.reserve(static_cast<size_t>(n_estimators));
    std::mt19937 rng(random_state);

    for (int stage = 0; stage < n_estimators; ++stage) {
        std::vector<double> residuals(static_cast<size_t>(n_samples));
        for (int i = 0; i < n_samples; ++i) {
            residuals[static_cast<size_t>(i)] = y[i] - F[static_cast<size_t>(i)];
        }

        std::vector<int> fit_idx;
        if (subsample < 1.0) {
            const int n_sub = std::max(1, static_cast<int>(subsample * n_samples));
            fit_idx.resize(static_cast<size_t>(n_sub));
            std::vector<int> all(static_cast<size_t>(n_samples));
            std::iota(all.begin(), all.end(), 0);
            std::shuffle(all.begin(), all.end(), rng);
            for (int i = 0; i < n_sub; ++i) {
                fit_idx[static_cast<size_t>(i)] = all[static_cast<size_t>(i)];
            }
        } else {
            fit_idx.resize(static_cast<size_t>(n_samples));
            std::iota(fit_idx.begin(), fit_idx.end(), 0);
        }

        std::vector<double> p_dummy(static_cast<size_t>(n_samples), 1.0);
        Tree tree = grow_tree(
            X,
            n_samples,
            n_features,
            residuals.data(),
            p_dummy.data(),
            sample_weight,
            fit_idx,
            0,
            max_depth,
            min_samples_leaf,
            min_samples_split,
            true,
            split_mode,
            rng
        );
        model.trees.push_back(std::move(tree));

        const std::vector<double> pred =
            predict_tree(model.trees.back(), X, n_samples, n_features);
        for (int i = 0; i < n_samples; ++i) {
            F[static_cast<size_t>(i)] += learning_rate * pred[static_cast<size_t>(i)];
        }
    }

    return model;
}

std::vector<double> predict_classifier_decision_binary(
    const ClassifierModel& model,
    const double* X,
    int n_samples
) {
    std::vector<double> F(static_cast<size_t>(n_samples), model.init_score[0]);
    for (const Tree& tree : model.trees_flat) {
        const std::vector<double> pred = predict_tree(tree, X, n_samples, model.n_features);
        for (int i = 0; i < n_samples; ++i) {
            F[static_cast<size_t>(i)] += model.learning_rate * pred[static_cast<size_t>(i)];
        }
    }
    return F;
}

std::vector<std::vector<double>> predict_classifier_decision_multiclass(
    const ClassifierModel& model,
    const double* X,
    int n_samples
) {
    std::vector<std::vector<double>> F(
        static_cast<size_t>(n_samples),
        model.init_score
    );
    const int K = model.n_classes;
    for (size_t stage = 0; stage < model.trees_flat.size() / static_cast<size_t>(K); ++stage) {
        for (int k = 0; k < K; ++k) {
            const Tree& tree = model.trees_flat[stage * static_cast<size_t>(K) + static_cast<size_t>(k)];
            const std::vector<double> pred = predict_tree(tree, X, n_samples, model.n_features);
            for (int i = 0; i < n_samples; ++i) {
                F[static_cast<size_t>(i)][static_cast<size_t>(k)] +=
                    model.learning_rate * model.mc_leaf_scale * pred[static_cast<size_t>(i)];
            }
        }
    }
    return F;
}

std::vector<double> predict_classifier_proba_pos(
    const ClassifierModel& model,
    const double* X,
    int n_samples
) {
    const std::vector<double> df = predict_classifier_decision_binary(model, X, n_samples);
    std::vector<double> proba(static_cast<size_t>(n_samples));
    for (int i = 0; i < n_samples; ++i) {
        double p = 1.0 / (1.0 + std::exp(-df[static_cast<size_t>(i)]));
        proba[static_cast<size_t>(i)] = clip(p, 1e-10, 1.0 - 1e-10);
    }
    return proba;
}

std::vector<std::vector<double>> predict_classifier_proba_multiclass(
    const ClassifierModel& model,
    const double* X,
    int n_samples
) {
    const auto F = predict_classifier_decision_multiclass(model, X, n_samples);
    std::vector<std::vector<double>> proba(
        static_cast<size_t>(n_samples),
        std::vector<double>(static_cast<size_t>(model.n_classes))
    );
    for (int i = 0; i < n_samples; ++i) {
        double fmax = F[static_cast<size_t>(i)][0];
        for (int k = 1; k < model.n_classes; ++k) {
            fmax = std::max(fmax, F[static_cast<size_t>(i)][static_cast<size_t>(k)]);
        }
        double denom = 0.0;
        for (int k = 0; k < model.n_classes; ++k) {
            proba[static_cast<size_t>(i)][static_cast<size_t>(k)] =
                std::exp(F[static_cast<size_t>(i)][static_cast<size_t>(k)] - fmax);
            denom += proba[static_cast<size_t>(i)][static_cast<size_t>(k)];
        }
        for (int k = 0; k < model.n_classes; ++k) {
            proba[static_cast<size_t>(i)][static_cast<size_t>(k)] /= denom;
        }
    }
    return proba;
}

std::vector<double> predict_regressor(
    const RegressorModel& model,
    const double* X,
    int n_samples
) {
    std::vector<double> out(static_cast<size_t>(n_samples), model.init_score);
    for (const Tree& tree : model.trees) {
        const std::vector<double> pred = predict_tree(tree, X, n_samples, model.n_features);
        for (int i = 0; i < n_samples; ++i) {
            out[static_cast<size_t>(i)] += model.learning_rate * pred[static_cast<size_t>(i)];
        }
    }
    return out;
}

std::size_t tree_size_bytes(const Tree& tree) {
    std::size_t bytes = sizeof(Tree);
    bytes += tree.nodes.size() * sizeof(TreeNode);
    for (const TreeNode& node : tree.nodes) {
        bytes += node.coef.size() * sizeof(double);
    }
    return bytes;
}

std::size_t classifier_model_size_bytes(const ClassifierModel& model) {
    std::size_t bytes = sizeof(ClassifierModel);
    bytes += model.init_score.size() * sizeof(double);
    bytes += model.classes.size() * sizeof(int64_t);
    for (const Tree& tree : model.trees_flat) {
        bytes += tree_size_bytes(tree);
    }
    return bytes;
}

std::size_t regressor_model_size_bytes(const RegressorModel& model) {
    std::size_t bytes = sizeof(RegressorModel);
    for (const Tree& tree : model.trees) {
        bytes += tree_size_bytes(tree);
    }
    return bytes;
}

}  // namespace sefrboost
