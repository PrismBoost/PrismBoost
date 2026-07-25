#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "sefr_boost.hpp"

namespace sefrboost {

constexpr char kClassifierMagic[] = "SEFRBC1";
constexpr char kRegressorMagic[] = "SEFRBR1";
constexpr std::size_t kFormatMagicLen = 7;

struct ClassifierCoreState {
    int n_estimators = 0;
    double learning_rate = 0.1;
    int max_depth = 3;
    int min_samples_leaf = 10;
    int min_samples_split = 2;
    double subsample = 1.0;
    SplitMode split_mode = SplitMode::HybridSampled;
    uint32_t random_state = 0;
    bool fitted = false;
    int n_features_in = 0;
    int n_classes = 0;
    ClassifierModel model;
};

struct RegressorCoreState {
    int n_estimators = 0;
    double learning_rate = 0.1;
    int max_depth = 3;
    int min_samples_leaf = 10;
    int min_samples_split = 2;
    double subsample = 1.0;
    SplitMode split_mode = SplitMode::HybridSampled;
    uint32_t random_state = 0;
    bool fitted = false;
    int n_features_in = 0;
    RegressorModel model;
};

std::vector<uint8_t> serialize_tree(const Tree& tree);

Tree deserialize_tree(const uint8_t* data, std::size_t size, std::size_t& offset);

std::vector<uint8_t> serialize_classifier_model(const ClassifierModel& model);

ClassifierModel deserialize_classifier_model(
    const uint8_t* data,
    std::size_t size,
    std::size_t& offset
);

std::vector<uint8_t> serialize_regressor_model(const RegressorModel& model);

RegressorModel deserialize_regressor_model(
    const uint8_t* data,
    std::size_t size,
    std::size_t& offset
);

std::vector<uint8_t> serialize_classifier_core(const ClassifierCoreState& state);

ClassifierCoreState deserialize_classifier_core(const std::vector<uint8_t>& bytes);

std::vector<uint8_t> serialize_regressor_core(const RegressorCoreState& state);

RegressorCoreState deserialize_regressor_core(const std::vector<uint8_t>& bytes);

}  // namespace sefrboost
