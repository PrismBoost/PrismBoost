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

if (MSVC)
    target_compile_options(_sefr_boost_core PRIVATE /O2 /fp:fast)
else()
    target_compile_options(_sefr_boost_core PRIVATE -O3 -ffast-math)
endif()

install(TARGETS _sefr_boost_core DESTINATION prismboost)
