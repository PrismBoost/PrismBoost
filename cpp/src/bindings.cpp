#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "sefr_boost.hpp"
#include "sefr_serialize.hpp"

namespace py = pybind11;

namespace {

std::string split_mode_to_string(sefrboost::SplitMode mode) {
    switch (mode) {
        case sefrboost::SplitMode::SefrOnly:
            return "sefr_only";
        case sefrboost::SplitMode::AxisFallback:
            return "axis_fallback";
        case sefrboost::SplitMode::HybridSampled:
            return "hybrid_sampled";
        case sefrboost::SplitMode::Hybrid:
            return "hybrid";
    }
    return "hybrid_sampled";
}

py::array_t<double> as_2d_c_contiguous(py::array_t<double> arr, const char* name) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument(std::string(name) + " must be a 2-D array");
    }
    return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(arr);
}

py::array_t<double> as_1d(py::array_t<double> arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    return py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(arr);
}

py::array_t<int64_t> as_1d_int64(py::array_t<int64_t> arr, const char* name) {
    if (arr.ndim() != 1) {
        throw std::invalid_argument(std::string(name) + " must be a 1-D array");
    }
    return py::array_t<int64_t, py::array::c_style | py::array::forcecast>::ensure(arr);
}

}  // namespace

class SEFRBoostClassifierCore {
public:
    SEFRBoostClassifierCore(
        int n_estimators = 100,
        double learning_rate = 0.1,
        int max_depth = 3,
        int min_samples_leaf = 10,
        int min_samples_split = 2,
        double subsample = 1.0,
        const std::string& split_mode = "hybrid_sampled",
        uint32_t random_state = 0
    )
        : n_estimators_(n_estimators),
          learning_rate_(learning_rate),
          max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          min_samples_split_(min_samples_split),
          subsample_(subsample),
          split_mode_(sefrboost::parse_split_mode(split_mode)),
          random_state_(random_state) {}

    void fit(
        py::array_t<double> X,
        py::array_t<int64_t> y_idx,
        py::object sample_weight = py::none()
    ) {
        X = as_2d_c_contiguous(X, "X");
        y_idx = as_1d_int64(y_idx, "y_idx");
        const auto xbuf = X.request();
        const auto ybuf = y_idx.request();
        if (xbuf.shape[0] != ybuf.shape[0]) {
            throw std::invalid_argument("X and y_idx must have the same number of rows");
        }

        const int n_samples = static_cast<int>(xbuf.shape[0]);
        const int n_features = static_cast<int>(xbuf.shape[1]);

        std::vector<double> sw(static_cast<size_t>(n_samples), 1.0);
        if (!sample_weight.is_none()) {
            auto sw_arr = as_1d(sample_weight.cast<py::array_t<double>>(), "sample_weight");
            if (sw_arr.shape(0) != n_samples) {
                throw std::invalid_argument("sample_weight length must match X rows");
            }
            const auto swbuf = sw_arr.request();
            const double* sw_ptr = static_cast<const double*>(swbuf.ptr);
            std::copy(sw_ptr, sw_ptr + n_samples, sw.begin());
        }

        const int64_t* y_ptr = static_cast<const int64_t*>(ybuf.ptr);
        int n_classes = 0;
        for (int i = 0; i < n_samples; ++i) {
            n_classes = std::max(n_classes, static_cast<int>(y_ptr[i]) + 1);
        }
        n_classes_ = n_classes;
        n_features_in_ = n_features;

        model_ = sefrboost::fit_classifier(
            static_cast<const double*>(xbuf.ptr),
            n_samples,
            n_features,
            y_ptr,
            sw.data(),
            n_classes,
            n_estimators_,
            learning_rate_,
            max_depth_,
            min_samples_leaf_,
            min_samples_split_,
            subsample_,
            split_mode_,
            random_state_
        );
        fitted_ = true;
    }

    py::array_t<double> decision_function(py::array_t<double> X) const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted");
        }
        X = as_2d_c_contiguous(X, "X");
        const auto xbuf = X.request();
        const int n_samples = static_cast<int>(xbuf.shape[0]);
        if (static_cast<int>(xbuf.shape[1]) != n_features_in_) {
            throw std::invalid_argument("X has incorrect number of features");
        }
        const double* xptr = static_cast<const double*>(xbuf.ptr);

        if (n_classes_ == 2) {
            const std::vector<double> df =
                sefrboost::predict_classifier_decision_binary(model_, xptr, n_samples);
            py::array_t<double> out(n_samples);
            std::copy(df.begin(), df.end(), out.mutable_data());
            return out;
        }

        const auto df = sefrboost::predict_classifier_decision_multiclass(model_, xptr, n_samples);
        py::array_t<double> out({n_samples, n_classes_});
        auto outbuf = out.request();
        double* optr = static_cast<double*>(outbuf.ptr);
        for (int i = 0; i < n_samples; ++i) {
            for (int k = 0; k < n_classes_; ++k) {
                optr[static_cast<size_t>(i) * static_cast<size_t>(n_classes_) + static_cast<size_t>(k)] =
                    df[static_cast<size_t>(i)][static_cast<size_t>(k)];
            }
        }
        return out;
    }

    py::array_t<double> predict_proba(py::array_t<double> X) const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted");
        }
        X = as_2d_c_contiguous(X, "X");
        const auto xbuf = X.request();
        const int n_samples = static_cast<int>(xbuf.shape[0]);
        if (static_cast<int>(xbuf.shape[1]) != n_features_in_) {
            throw std::invalid_argument("X has incorrect number of features");
        }
        const double* xptr = static_cast<const double*>(xbuf.ptr);

        if (n_classes_ == 2) {
            const std::vector<double> ppos =
                sefrboost::predict_classifier_proba_pos(model_, xptr, n_samples);
            py::array_t<double> out({n_samples, 2});
            auto outbuf = out.request();
            double* optr = static_cast<double*>(outbuf.ptr);
            for (int i = 0; i < n_samples; ++i) {
                const double p1 = ppos[static_cast<size_t>(i)];
                optr[static_cast<size_t>(i) * 2] = 1.0 - p1;
                optr[static_cast<size_t>(i) * 2 + 1] = p1;
            }
            return out;
        }

        const auto proba =
            sefrboost::predict_classifier_proba_multiclass(model_, xptr, n_samples);
        py::array_t<double> out({n_samples, n_classes_});
        auto outbuf = out.request();
        double* optr = static_cast<double*>(outbuf.ptr);
        for (int i = 0; i < n_samples; ++i) {
            for (int k = 0; k < n_classes_; ++k) {
                optr[static_cast<size_t>(i) * static_cast<size_t>(n_classes_) + static_cast<size_t>(k)] =
                    proba[static_cast<size_t>(i)][static_cast<size_t>(k)];
            }
        }
        return out;
    }

    py::array_t<int64_t> predict(py::array_t<double> X) const {
        const py::array_t<double> proba = predict_proba(X);
        const auto pbuf = proba.request();
        const int n_samples = static_cast<int>(pbuf.shape[0]);
        const int n_classes = static_cast<int>(pbuf.shape[1]);
        const double* pptr = static_cast<const double*>(pbuf.ptr);

        py::array_t<int64_t> out(n_samples);
        int64_t* optr = out.mutable_data();
        for (int i = 0; i < n_samples; ++i) {
            int best = 0;
            double best_p = pptr[static_cast<size_t>(i) * static_cast<size_t>(n_classes)];
            for (int k = 1; k < n_classes; ++k) {
                const double pk =
                    pptr[static_cast<size_t>(i) * static_cast<size_t>(n_classes) + static_cast<size_t>(k)];
                if (pk > best_p) {
                    best_p = pk;
                    best = k;
                }
            }
            optr[i] = best;
        }
        return out;
    }

    bool fitted() const { return fitted_; }
    int n_features_in() const { return n_features_in_; }
    int n_classes() const { return n_classes_; }
    std::size_t model_size_bytes() const {
        if (!fitted_) {
            return 0;
        }
        return sefrboost::classifier_model_size_bytes(model_);
    }

    py::bytes to_bytes() const {
        sefrboost::ClassifierCoreState state;
        state.n_estimators = n_estimators_;
        state.learning_rate = learning_rate_;
        state.max_depth = max_depth_;
        state.min_samples_leaf = min_samples_leaf_;
        state.min_samples_split = min_samples_split_;
        state.subsample = subsample_;
        state.split_mode = split_mode_;
        state.random_state = random_state_;
        state.fitted = fitted_;
        state.n_features_in = n_features_in_;
        state.n_classes = n_classes_;
        state.model = model_;
        const std::vector<uint8_t> blob = sefrboost::serialize_classifier_core(state);
        return py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size());
    }

    static SEFRBoostClassifierCore from_bytes(py::bytes data) {
        const std::string raw = static_cast<std::string>(data);
        const std::vector<uint8_t> blob(raw.begin(), raw.end());
        const sefrboost::ClassifierCoreState state = sefrboost::deserialize_classifier_core(blob);
        SEFRBoostClassifierCore obj(
            state.n_estimators,
            state.learning_rate,
            state.max_depth,
            state.min_samples_leaf,
            state.min_samples_split,
            state.subsample,
            split_mode_to_string(state.split_mode),
            state.random_state
        );
        obj.fitted_ = state.fitted;
        obj.n_features_in_ = state.n_features_in;
        obj.n_classes_ = state.n_classes;
        obj.model_ = state.model;
        return obj;
    }

private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_leaf_;
    int min_samples_split_;
    double subsample_;
    sefrboost::SplitMode split_mode_;
    uint32_t random_state_;
    bool fitted_ = false;
    int n_features_in_ = 0;
    int n_classes_ = 0;
    sefrboost::ClassifierModel model_;
};

class SEFRBoostRegressorCore {
public:
    SEFRBoostRegressorCore(
        int n_estimators = 100,
        double learning_rate = 0.1,
        int max_depth = 3,
        int min_samples_leaf = 10,
        int min_samples_split = 2,
        double subsample = 1.0,
        const std::string& split_mode = "hybrid_sampled",
        uint32_t random_state = 0
    )
        : n_estimators_(n_estimators),
          learning_rate_(learning_rate),
          max_depth_(max_depth),
          min_samples_leaf_(min_samples_leaf),
          min_samples_split_(min_samples_split),
          subsample_(subsample),
          split_mode_(sefrboost::parse_split_mode(split_mode)),
          random_state_(random_state) {}

    void fit(
        py::array_t<double> X,
        py::array_t<double> y,
        py::object sample_weight = py::none()
    ) {
        X = as_2d_c_contiguous(X, "X");
        y = as_1d(y, "y");
        const auto xbuf = X.request();
        const auto ybuf = y.request();
        if (xbuf.shape[0] != ybuf.shape[0]) {
            throw std::invalid_argument("X and y must have the same number of rows");
        }

        const int n_samples = static_cast<int>(xbuf.shape[0]);
        const int n_features = static_cast<int>(xbuf.shape[1]);
        n_features_in_ = n_features;

        std::vector<double> sw(static_cast<size_t>(n_samples), 1.0);
        if (!sample_weight.is_none()) {
            auto sw_arr = as_1d(sample_weight.cast<py::array_t<double>>(), "sample_weight");
            if (sw_arr.shape(0) != n_samples) {
                throw std::invalid_argument("sample_weight length must match X rows");
            }
            const auto swbuf = sw_arr.request();
            const double* sw_ptr = static_cast<const double*>(swbuf.ptr);
            std::copy(sw_ptr, sw_ptr + n_samples, sw.begin());
        }

        model_ = sefrboost::fit_regressor(
            static_cast<const double*>(xbuf.ptr),
            n_samples,
            n_features,
            static_cast<const double*>(ybuf.ptr),
            sw.data(),
            n_estimators_,
            learning_rate_,
            max_depth_,
            min_samples_leaf_,
            min_samples_split_,
            subsample_,
            split_mode_,
            random_state_
        );
        fitted_ = true;
    }

    py::array_t<double> predict(py::array_t<double> X) const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted");
        }
        X = as_2d_c_contiguous(X, "X");
        const auto xbuf = X.request();
        const int n_samples = static_cast<int>(xbuf.shape[0]);
        if (static_cast<int>(xbuf.shape[1]) != n_features_in_) {
            throw std::invalid_argument("X has incorrect number of features");
        }
        const std::vector<double> pred = sefrboost::predict_regressor(
            model_,
            static_cast<const double*>(xbuf.ptr),
            n_samples
        );
        py::array_t<double> out(n_samples);
        std::copy(pred.begin(), pred.end(), out.mutable_data());
        return out;
    }

    bool fitted() const { return fitted_; }
    int n_features_in() const { return n_features_in_; }
    std::size_t model_size_bytes() const {
        if (!fitted_) {
            return 0;
        }
        return sefrboost::regressor_model_size_bytes(model_);
    }

    py::bytes to_bytes() const {
        sefrboost::RegressorCoreState state;
        state.n_estimators = n_estimators_;
        state.learning_rate = learning_rate_;
        state.max_depth = max_depth_;
        state.min_samples_leaf = min_samples_leaf_;
        state.min_samples_split = min_samples_split_;
        state.subsample = subsample_;
        state.split_mode = split_mode_;
        state.random_state = random_state_;
        state.fitted = fitted_;
        state.n_features_in = n_features_in_;
        state.model = model_;
        const std::vector<uint8_t> blob = sefrboost::serialize_regressor_core(state);
        return py::bytes(reinterpret_cast<const char*>(blob.data()), blob.size());
    }

    static SEFRBoostRegressorCore from_bytes(py::bytes data) {
        const std::string raw = static_cast<std::string>(data);
        const std::vector<uint8_t> blob(raw.begin(), raw.end());
        const sefrboost::RegressorCoreState state = sefrboost::deserialize_regressor_core(blob);
        SEFRBoostRegressorCore obj(
            state.n_estimators,
            state.learning_rate,
            state.max_depth,
            state.min_samples_leaf,
            state.min_samples_split,
            state.subsample,
            split_mode_to_string(state.split_mode),
            state.random_state
        );
        obj.fitted_ = state.fitted;
        obj.n_features_in_ = state.n_features_in;
        obj.model_ = state.model;
        return obj;
    }

private:
    int n_estimators_;
    double learning_rate_;
    int max_depth_;
    int min_samples_leaf_;
    int min_samples_split_;
    double subsample_;
    sefrboost::SplitMode split_mode_;
    uint32_t random_state_;
    bool fitted_ = false;
    int n_features_in_ = 0;
    sefrboost::RegressorModel model_;
};

PYBIND11_MODULE(_sefr_boost_core, m) {
    m.doc() = "C++ core for SEFRBoost gradient boosting";

    py::class_<SEFRBoostClassifierCore>(m, "SEFRBoostClassifierCore")
        .def(
            py::init<int, double, int, int, int, double, std::string, uint32_t>(),
            py::arg("n_estimators") = 100,
            py::arg("learning_rate") = 0.1,
            py::arg("max_depth") = 3,
            py::arg("min_samples_leaf") = 10,
            py::arg("min_samples_split") = 2,
            py::arg("subsample") = 1.0,
            py::arg("split_mode") = "hybrid_sampled",
            py::arg("random_state") = 0
        )
        .def("fit", &SEFRBoostClassifierCore::fit, py::arg("X"), py::arg("y_idx"), py::arg("sample_weight") = py::none())
        .def("predict", &SEFRBoostClassifierCore::predict, py::arg("X"))
        .def("predict_proba", &SEFRBoostClassifierCore::predict_proba, py::arg("X"))
        .def("decision_function", &SEFRBoostClassifierCore::decision_function, py::arg("X"))
        .def_property_readonly("fitted", &SEFRBoostClassifierCore::fitted)
        .def_property_readonly("n_features_in", &SEFRBoostClassifierCore::n_features_in)
        .def_property_readonly("n_classes", &SEFRBoostClassifierCore::n_classes)
        .def("model_size_bytes", &SEFRBoostClassifierCore::model_size_bytes)
        .def("to_bytes", &SEFRBoostClassifierCore::to_bytes)
        .def_static("from_bytes", &SEFRBoostClassifierCore::from_bytes);

    py::class_<SEFRBoostRegressorCore>(m, "SEFRBoostRegressorCore")
        .def(
            py::init<int, double, int, int, int, double, std::string, uint32_t>(),
            py::arg("n_estimators") = 100,
            py::arg("learning_rate") = 0.1,
            py::arg("max_depth") = 3,
            py::arg("min_samples_leaf") = 10,
            py::arg("min_samples_split") = 2,
            py::arg("subsample") = 1.0,
            py::arg("split_mode") = "hybrid_sampled",
            py::arg("random_state") = 0
        )
        .def("fit", &SEFRBoostRegressorCore::fit, py::arg("X"), py::arg("y"), py::arg("sample_weight") = py::none())
        .def("predict", &SEFRBoostRegressorCore::predict, py::arg("X"))
        .def_property_readonly("fitted", &SEFRBoostRegressorCore::fitted)
        .def_property_readonly("n_features_in", &SEFRBoostRegressorCore::n_features_in)
        .def("model_size_bytes", &SEFRBoostRegressorCore::model_size_bytes)
        .def("to_bytes", &SEFRBoostRegressorCore::to_bytes)
        .def_static("from_bytes", &SEFRBoostRegressorCore::from_bytes);
}
