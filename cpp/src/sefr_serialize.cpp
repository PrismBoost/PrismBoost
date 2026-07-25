#include "sefr_serialize.hpp"

#include <cstring>
#include <stdexcept>

namespace sefrboost {

namespace {

class BufferWriter {
public:
    void write_bytes(const void* ptr, std::size_t n) {
        const auto* p = static_cast<const uint8_t*>(ptr);
        out_.insert(out_.end(), p, p + n);
    }

    void write_u8(uint8_t v) { write_bytes(&v, 1); }
    void write_i32(int32_t v) { write_bytes(&v, sizeof(v)); }
    void write_i64(int64_t v) { write_bytes(&v, sizeof(v)); }
    void write_u32(uint32_t v) { write_bytes(&v, sizeof(v)); }
    void write_f64(double v) { write_bytes(&v, sizeof(v)); }

    std::vector<uint8_t> take() { return std::move(out_); }

private:
    std::vector<uint8_t> out_;
};

class BufferReader {
public:
    BufferReader(const uint8_t* data, std::size_t size, std::size_t& offset)
        : data_(data), size_(size), offset_(offset) {}

    void read_bytes(void* ptr, std::size_t n) {
        if (offset_ + n > size_) {
            throw std::runtime_error("SEFRBoost deserialize: unexpected end of buffer");
        }
        std::memcpy(ptr, data_ + offset_, n);
        offset_ += n;
    }

    uint8_t read_u8() {
        uint8_t v = 0;
        read_bytes(&v, 1);
        return v;
    }

    int32_t read_i32() {
        int32_t v = 0;
        read_bytes(&v, sizeof(v));
        return v;
    }

    int64_t read_i64() {
        int64_t v = 0;
        read_bytes(&v, sizeof(v));
        return v;
    }

    uint32_t read_u32() {
        uint32_t v = 0;
        read_bytes(&v, sizeof(v));
        return v;
    }

    double read_f64() {
        double v = 0.0;
        read_bytes(&v, sizeof(v));
        return v;
    }

private:
    const uint8_t* data_;
    std::size_t size_;
    std::size_t& offset_;
};

void expect_magic(const std::vector<uint8_t>& bytes, const char* magic) {
    if (bytes.size() < kFormatMagicLen) {
        throw std::runtime_error("SEFRBoost deserialize: buffer too small");
    }
    if (std::memcmp(bytes.data(), magic, kFormatMagicLen) != 0) {
        throw std::runtime_error("SEFRBoost deserialize: invalid magic header");
    }
}

void write_tree(BufferWriter& w, const Tree& tree) {
    w.write_i32(static_cast<int32_t>(tree.nodes.size()));
    w.write_i32(tree.root);
    for (const TreeNode& node : tree.nodes) {
        w.write_u8(node.is_leaf ? 1 : 0);
        w.write_f64(node.value);
        w.write_f64(node.intercept);
        w.write_i32(node.left);
        w.write_i32(node.right);
        w.write_i32(static_cast<int32_t>(node.coef.size()));
        for (double c : node.coef) {
            w.write_f64(c);
        }
    }
}

Tree read_tree(BufferReader& r) {
    Tree tree;
    const int32_t n_nodes = r.read_i32();
    tree.root = r.read_i32();
    tree.nodes.resize(static_cast<size_t>(n_nodes));
    for (int32_t i = 0; i < n_nodes; ++i) {
        TreeNode& node = tree.nodes[static_cast<size_t>(i)];
        node.is_leaf = r.read_u8() != 0;
        node.value = r.read_f64();
        node.intercept = r.read_f64();
        node.left = r.read_i32();
        node.right = r.read_i32();
        const int32_t n_coef = r.read_i32();
        node.coef.resize(static_cast<size_t>(n_coef));
        for (int32_t j = 0; j < n_coef; ++j) {
            node.coef[static_cast<size_t>(j)] = r.read_f64();
        }
    }
    return tree;
}

}  // namespace

std::vector<uint8_t> serialize_tree(const Tree& tree) {
    BufferWriter w;
    write_tree(w, tree);
    return w.take();
}

Tree deserialize_tree(const uint8_t* data, std::size_t size, std::size_t& offset) {
    BufferReader r(data, size, offset);
    return read_tree(r);
}

std::vector<uint8_t> serialize_classifier_model(const ClassifierModel& model) {
    BufferWriter w;
    w.write_i32(model.n_features);
    w.write_i32(model.n_classes);
    w.write_f64(model.learning_rate);
    w.write_f64(model.mc_leaf_scale);
    w.write_u8(model.multiclass ? 1 : 0);
    w.write_i32(model.trees_per_stage);
    w.write_i32(static_cast<int32_t>(model.init_score.size()));
    for (double v : model.init_score) {
        w.write_f64(v);
    }
    w.write_i32(static_cast<int32_t>(model.classes.size()));
    for (int64_t c : model.classes) {
        w.write_i64(c);
    }
    w.write_i32(static_cast<int32_t>(model.trees_flat.size()));
    for (const Tree& tree : model.trees_flat) {
        write_tree(w, tree);
    }
    return w.take();
}

ClassifierModel deserialize_classifier_model(
    const uint8_t* data,
    std::size_t size,
    std::size_t& offset
) {
    BufferReader r(data, size, offset);
    ClassifierModel model;
    model.n_features = r.read_i32();
    model.n_classes = r.read_i32();
    model.learning_rate = r.read_f64();
    model.mc_leaf_scale = r.read_f64();
    model.multiclass = r.read_u8() != 0;
    model.trees_per_stage = r.read_i32();
    const int32_t n_init = r.read_i32();
    model.init_score.resize(static_cast<size_t>(n_init));
    for (int32_t i = 0; i < n_init; ++i) {
        model.init_score[static_cast<size_t>(i)] = r.read_f64();
    }
    const int32_t n_classes = r.read_i32();
    model.classes.resize(static_cast<size_t>(n_classes));
    for (int32_t i = 0; i < n_classes; ++i) {
        model.classes[static_cast<size_t>(i)] = r.read_i64();
    }
    const int32_t n_trees = r.read_i32();
    model.trees_flat.resize(static_cast<size_t>(n_trees));
    for (int32_t t = 0; t < n_trees; ++t) {
        model.trees_flat[static_cast<size_t>(t)] = read_tree(r);
    }
    return model;
}

std::vector<uint8_t> serialize_regressor_model(const RegressorModel& model) {
    BufferWriter w;
    w.write_i32(model.n_features);
    w.write_f64(model.learning_rate);
    w.write_f64(model.init_score);
    w.write_i32(static_cast<int32_t>(model.trees.size()));
    for (const Tree& tree : model.trees) {
        write_tree(w, tree);
    }
    return w.take();
}

RegressorModel deserialize_regressor_model(
    const uint8_t* data,
    std::size_t size,
    std::size_t& offset
) {
    BufferReader r(data, size, offset);
    RegressorModel model;
    model.n_features = r.read_i32();
    model.learning_rate = r.read_f64();
    model.init_score = r.read_f64();
    const int32_t n_trees = r.read_i32();
    model.trees.resize(static_cast<size_t>(n_trees));
    for (int32_t t = 0; t < n_trees; ++t) {
        model.trees[static_cast<size_t>(t)] = read_tree(r);
    }
    return model;
}

std::vector<uint8_t> serialize_classifier_core(const ClassifierCoreState& state) {
    BufferWriter w;
    w.write_bytes(kClassifierMagic, kFormatMagicLen);
    w.write_i32(state.n_estimators);
    w.write_f64(state.learning_rate);
    w.write_i32(state.max_depth);
    w.write_i32(state.min_samples_leaf);
    w.write_i32(state.min_samples_split);
    w.write_f64(state.subsample);
    w.write_i32(static_cast<int32_t>(state.split_mode));
    w.write_u32(state.random_state);
    w.write_u8(state.fitted ? 1 : 0);
    w.write_i32(state.n_features_in);
    w.write_i32(state.n_classes);
    const std::vector<uint8_t> model_bytes = serialize_classifier_model(state.model);
    w.write_i32(static_cast<int32_t>(model_bytes.size()));
    w.write_bytes(model_bytes.data(), model_bytes.size());
    return w.take();
}

ClassifierCoreState deserialize_classifier_core(const std::vector<uint8_t>& bytes) {
    expect_magic(bytes, kClassifierMagic);
    std::size_t offset = kFormatMagicLen;
    BufferReader r(bytes.data(), bytes.size(), offset);
    ClassifierCoreState state;
    state.n_estimators = r.read_i32();
    state.learning_rate = r.read_f64();
    state.max_depth = r.read_i32();
    state.min_samples_leaf = r.read_i32();
    state.min_samples_split = r.read_i32();
    state.subsample = r.read_f64();
    state.split_mode = static_cast<SplitMode>(r.read_i32());
    state.random_state = r.read_u32();
    state.fitted = r.read_u8() != 0;
    state.n_features_in = r.read_i32();
    state.n_classes = r.read_i32();
    const int32_t model_len = r.read_i32();
    std::size_t model_offset = offset;
    state.model = deserialize_classifier_model(bytes.data(), bytes.size(), model_offset);
    if (model_offset != offset + static_cast<std::size_t>(model_len)) {
        throw std::runtime_error("SEFRBoost deserialize: classifier model length mismatch");
    }
    offset = model_offset;
    return state;
}

std::vector<uint8_t> serialize_regressor_core(const RegressorCoreState& state) {
    BufferWriter w;
    w.write_bytes(kRegressorMagic, kFormatMagicLen);
    w.write_i32(state.n_estimators);
    w.write_f64(state.learning_rate);
    w.write_i32(state.max_depth);
    w.write_i32(state.min_samples_leaf);
    w.write_i32(state.min_samples_split);
    w.write_f64(state.subsample);
    w.write_i32(static_cast<int32_t>(state.split_mode));
    w.write_u32(state.random_state);
    w.write_u8(state.fitted ? 1 : 0);
    w.write_i32(state.n_features_in);
    const std::vector<uint8_t> model_bytes = serialize_regressor_model(state.model);
    w.write_i32(static_cast<int32_t>(model_bytes.size()));
    w.write_bytes(model_bytes.data(), model_bytes.size());
    return w.take();
}

RegressorCoreState deserialize_regressor_core(const std::vector<uint8_t>& bytes) {
    expect_magic(bytes, kRegressorMagic);
    std::size_t offset = kFormatMagicLen;
    BufferReader r(bytes.data(), bytes.size(), offset);
    RegressorCoreState state;
    state.n_estimators = r.read_i32();
    state.learning_rate = r.read_f64();
    state.max_depth = r.read_i32();
    state.min_samples_leaf = r.read_i32();
    state.min_samples_split = r.read_i32();
    state.subsample = r.read_f64();
    state.split_mode = static_cast<SplitMode>(r.read_i32());
    state.random_state = r.read_u32();
    state.fitted = r.read_u8() != 0;
    state.n_features_in = r.read_i32();
    const int32_t model_len = r.read_i32();
    std::size_t model_offset = offset;
    state.model = deserialize_regressor_model(bytes.data(), bytes.size(), model_offset);
    if (model_offset != offset + static_cast<std::size_t>(model_len)) {
        throw std::runtime_error("SEFRBoost deserialize: regressor model length mismatch");
    }
    offset = model_offset;
    return state;
}

}  // namespace sefrboost
