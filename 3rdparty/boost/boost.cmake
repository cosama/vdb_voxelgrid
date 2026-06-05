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

set(BOOST_URL "https://archives.boost.io/release/1.80.0/source/boost_1_80_0.tar.gz")
set(BOOST_URL_SHA256 "4b2136f98bdd1f5857f1c3dea9ac2018effe65286cf251534b6ae20cc45e1847")
set(BOOST_CONFIGURE <SOURCE_DIR>/bootstrap.sh --with-libraries=iostreams,regex)
set(BOOST_INSTALL
    <SOURCE_DIR>/b2
    install
    link=static
    warnings=off
    cxxflags=-fPIC
    cflags=-fPIC
    --prefix=<INSTALL_DIR>)

include(ExternalProject)
set(_BOOST_INSTALL_DIR "${VDBVOXELGRID_EXTERNAL_ROOT}/boost")
set(_BOOST_INCLUDE_DIR "${_BOOST_INSTALL_DIR}/include")
set(_BOOST_IOSTREAMS_LIBRARY "${_BOOST_INSTALL_DIR}/lib/libboost_iostreams.a")
set(_BOOST_REGEX_LIBRARY "${_BOOST_INSTALL_DIR}/lib/libboost_regex.a")
set(_BOOST_CACHE_READY FALSE)

if(EXISTS "${_BOOST_INCLUDE_DIR}/boost/version.hpp"
   AND EXISTS "${_BOOST_IOSTREAMS_LIBRARY}"
   AND EXISTS "${_BOOST_REGEX_LIBRARY}")
  set(_BOOST_CACHE_READY TRUE)
  message(STATUS "Reusing cached vendored Boost from ${_BOOST_INSTALL_DIR}")
else()
  ExternalProject_Add(
    external_boost
    PREFIX "${_BOOST_INSTALL_DIR}"
    URL ${BOOST_URL}
    URL_HASH SHA256=${BOOST_URL_SHA256}
    BUILD_IN_SOURCE true
    CONFIGURE_COMMAND "${BOOST_CONFIGURE}"
    BUILD_COMMAND ""
    INSTALL_COMMAND "${BOOST_INSTALL}"
    BUILD_BYPRODUCTS "${_BOOST_IOSTREAMS_LIBRARY}" "${_BOOST_REGEX_LIBRARY}")
endif()

# Simulate importing Boost::iostreams for OpenVDBHelper target
set(BOOST_ROOT ${_BOOST_INSTALL_DIR} CACHE INTERNAL "Boost libraries Install directory")
add_library(BoostIostreamsHelper INTERFACE)
if(NOT _BOOST_CACHE_READY)
  add_dependencies(BoostIostreamsHelper external_boost)
endif()
target_include_directories(BoostIostreamsHelper INTERFACE "${_BOOST_INCLUDE_DIR}")
target_link_libraries(BoostIostreamsHelper INTERFACE "${_BOOST_IOSTREAMS_LIBRARY}")
add_library(Boost::iostreams ALIAS BoostIostreamsHelper)
