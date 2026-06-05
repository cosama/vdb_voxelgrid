# MIT License
#
# # Copyright (c) 2022 Ignacio Vizzo, Cyrill Stachniss, University of Bonn
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

include(ExternalProject)
set(_BLOSC_INSTALL_DIR "${VDBVOXELGRID_EXTERNAL_ROOT}/blosc")
set(_BLOSC_INCLUDE_DIR "${_BLOSC_INSTALL_DIR}/include")
set(_BLOSC_LIBRARY "${_BLOSC_INSTALL_DIR}/lib/libblosc.a")
set(_BLOSC_CACHE_READY FALSE)

if(EXISTS "${_BLOSC_INCLUDE_DIR}/blosc.h" AND EXISTS "${_BLOSC_LIBRARY}")
  set(_BLOSC_CACHE_READY TRUE)
  message(STATUS "Reusing cached vendored Blosc from ${_BLOSC_INSTALL_DIR}")
else()
  ExternalProject_Add(
    external_blosc
    PREFIX "${_BLOSC_INSTALL_DIR}"
    URL https://github.com/Blosc/c-blosc/archive/refs/tags/v1.17.0.tar.gz
    URL_HASH SHA256=75d98c752b8cf0d4a6380a3089d56523f175b0afa2d0cf724a1bd0a1a8f975a4
    UPDATE_COMMAND ""
    CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
               ${ExternalProject_CMAKE_ARGS}
               ${ExternalProject_CMAKE_CXX_FLAGS}
               # Custom OpenVDB build settings
               -DCMAKE_POLICY_VERSION_MINIMUM=3.5
               -DBUILD_STATIC=ON
               -DBUILD_TESTS=OFF
               -DBUILD_BENCHMARKS=OFF
               -DPREFER_EXTERNAL_COMPLIBS=OFF
    BUILD_BYPRODUCTS "${_BLOSC_LIBRARY}")
endif()

# Simulate importing Blosc::blosc for OpenVDBHelper target
set(BLOSC_ROOT ${_BLOSC_INSTALL_DIR} CACHE INTERNAL "BLOSC_ROOT Install directory")
add_library(BloscHelper INTERFACE)
if(NOT _BLOSC_CACHE_READY)
  add_dependencies(BloscHelper external_blosc)
endif()
target_include_directories(BloscHelper INTERFACE "${_BLOSC_INCLUDE_DIR}")
target_link_libraries(BloscHelper INTERFACE "${_BLOSC_LIBRARY}")
add_library(Blosc::blosc ALIAS BloscHelper)
