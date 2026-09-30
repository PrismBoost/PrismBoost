cmake_minimum_required(VERSION 3.15)
project(prismboost_core LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(
    _sefr_boost_core
    src/bindings.cpp
    src/sefr_boost.cpp
    src/sefr_serialize.cpp
)

target_include_directories(_sefr_boost_core PRIVATE include)

# The split search uses NaN (`kSplitNone`) and -infinity (the initial best gain) as sentinels,
# which fast-math tells the compiler cannot occur: `-ffast-math` implies `-ffinite-math-only`, so
# `std::isnan` folds to false and comparisons against -infinity are free to be reordered. Under
# `-O3 -ffast-math` clang dropped the best-candidate bookkeeping in `best_split_threshold`, so
# every tree came back as a single leaf. Keep `-O3`; reassociation is allowed, infinities are not.
if (MSVC)
    target_compile_options(_sefr_boost_core PRIVATE /O2 /fp:contract=fast)
else()
    target_compile_options(_sefr_boost_core PRIVATE -O3 -fno-math-errno -ffp-contract=fast)
endif()

install(TARGETS _sefr_boost_core DESTINATION prismboost)
