# Copyright (c) Huawei Technologies Co., Ltd. 2024-2026. All rights reserved.

cmake_minimum_required(VERSION 3.14)
project(sssd_speculator LANGUAGES C CXX)

# ——— C++ settings ———
set(CMAKE_CXX_STANDARD      14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS    OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ——— Python & pybind11 ———
# try 3.9, 3.10, 3.11 in turn
find_package(Python 3.9 EXACT COMPONENTS Interpreter Development QUIET)
if (NOT Python_FOUND)
  find_package(Python 3.10 EXACT COMPONENTS Interpreter Development QUIET)
endif()
if (NOT Python_FOUND)
  find_package(Python 3.11 EXACT COMPONENTS Interpreter Development QUIET)
endif()
if (NOT Python_FOUND)
  find_package(Python 3.12 EXACT COMPONENTS Interpreter Development QUIET)
endif()
if (NOT Python_FOUND)
  message(FATAL_ERROR "Python ≥ 3.9 (≤ 3.12) is required")
endif()

# Make sure you pass the right pybind11 dir (see setup.py)
set(pybind11_DIR ${PYBIND11_INCLUDE_DIR})
find_package(pybind11 REQUIRED)


# ——— Offline dependencies ———
#
# The benchmark server cannot access GitHub.  The original upstream build used
# FetchContent with remote Git URLs, which made an otherwise local native build
# fail before compiling any SSSD source.  Prefer an explicitly supplied local
# checkout, then fall back to the copy already present in this repository.
set(_SSSD_BUNDLED_LIBSAIS_DIR
  "${CMAKE_CURRENT_LIST_DIR}/../evaluation/REST/DraftRetriever/src/libsais")
set(SSSD_LIBSAIS_SOURCE_DIR "${_SSSD_BUNDLED_LIBSAIS_DIR}" CACHE PATH
  "Local libsais checkout (must contain libsais.c and libsais.h)")
if(NOT IS_ABSOLUTE "${SSSD_LIBSAIS_SOURCE_DIR}")
  get_filename_component(SSSD_LIBSAIS_SOURCE_DIR "${SSSD_LIBSAIS_SOURCE_DIR}"
    ABSOLUTE BASE_DIR "${CMAKE_CURRENT_LIST_DIR}")
endif()

# Support both the upstream checkout layout (src/ + include/) and the
# repository's vendored REST copy (libsais.c + libsais.h in one directory).
if(EXISTS "${SSSD_LIBSAIS_SOURCE_DIR}/src/libsais.c")
  set(_SSSD_LIBSAIS_C "${SSSD_LIBSAIS_SOURCE_DIR}/src/libsais.c")
  set(_SSSD_LIBSAIS_INCLUDE_DIR "${SSSD_LIBSAIS_SOURCE_DIR}/include")
elseif(EXISTS "${SSSD_LIBSAIS_SOURCE_DIR}/libsais.c")
  set(_SSSD_LIBSAIS_C "${SSSD_LIBSAIS_SOURCE_DIR}/libsais.c")
  set(_SSSD_LIBSAIS_INCLUDE_DIR "${SSSD_LIBSAIS_SOURCE_DIR}")
else()
  message(FATAL_ERROR
    "Local libsais source was not found at '${SSSD_LIBSAIS_SOURCE_DIR}'. "
    "Set -DSSSD_LIBSAIS_SOURCE_DIR=/path/to/libsais; network downloads are "
    "intentionally disabled.")
endif()

add_library(libsais_static STATIC "${_SSSD_LIBSAIS_C}")
target_include_directories(libsais_static PUBLIC "${_SSSD_LIBSAIS_INCLUDE_DIR}")

# spdlog is used only for native diagnostic logging.  A full local checkout can
# be supplied when available; otherwise use the small header-only compatibility
# implementation shipped with this repository.  Both paths are fully offline.
set(SSSD_SPDLOG_SOURCE_DIR "" CACHE PATH
  "Optional local spdlog checkout; empty uses the bundled compatibility headers")
if(SSSD_SPDLOG_SOURCE_DIR)
  if(NOT EXISTS "${SSSD_SPDLOG_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR
      "SSSD_SPDLOG_SOURCE_DIR='${SSSD_SPDLOG_SOURCE_DIR}' has no CMakeLists.txt")
  endif()
  set(SPDLOG_BUILD_EXAMPLE OFF CACHE BOOL "" FORCE)
  set(SPDLOG_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
  add_subdirectory("${SSSD_SPDLOG_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/sssd_spdlog"
    EXCLUDE_FROM_ALL)
  if(NOT TARGET spdlog::spdlog AND TARGET spdlog)
    add_library(spdlog::spdlog ALIAS spdlog)
  endif()
else()
  set(_SSSD_SPDLOG_COMPAT_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../third_party/spdlog_compat/include")
  if(NOT EXISTS "${_SSSD_SPDLOG_COMPAT_DIR}/spdlog/spdlog.h")
    message(FATAL_ERROR
      "Bundled spdlog compatibility headers are missing at "
      "'${_SSSD_SPDLOG_COMPAT_DIR}'")
  endif()
  add_library(sssd_spdlog_compat INTERFACE)
  target_include_directories(sssd_spdlog_compat INTERFACE
    "${_SSSD_SPDLOG_COMPAT_DIR}")
  add_library(spdlog::spdlog ALIAS sssd_spdlog_compat)
endif()

# ——— Include source files ———
set(SOURCES
  src/writer.cpp
  src/reader.cpp
  src/trie.cpp
  src/prompt_cache.cpp
  src/utils.cpp
  src/memory_pool.cpp
  src/sssd_speculator.cpp
)

add_definitions(-DLIBSAIS_STATIC)
pybind11_add_module(sssd_speculator ${SOURCES})

target_include_directories(sssd_speculator PRIVATE
  ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(sssd_speculator PRIVATE
  libsais_static
  spdlog::spdlog
)

# ——— Optional: NUMA support on Linux ———
if (UNIX AND NOT APPLE)
  # Try to find libnuma (from numactl)
  find_library(NUMA_LIB numa)

  if (NUMA_LIB)
    message(STATUS "libnuma found: ${NUMA_LIB} — enabling NUMA-aware code")
    target_compile_definitions(sssd_speculator PRIVATE HAVE_LIBNUMA=1)
    target_link_libraries(sssd_speculator PRIVATE ${NUMA_LIB})
  else()
    message(STATUS "libnuma not found — building without NUMA support")
  endif()
endif()

find_package(Threads REQUIRED)
target_link_libraries(sssd_speculator PRIVATE Threads::Threads)
