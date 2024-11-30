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

cmake_minimum_required(VERSION 3.10)
project(VDBVoxelgrid VERSION 0.1.6 LANGUAGES CXX)

add_library(vdbvoxelgrid STATIC)
add_library(Voxelgrid::vdbvoxelgrid ALIAS vdbvoxelgrid)

# In case we are building this as a standalone project, we should look here the dependencies
if(NOT TARGET OpenVDB::openvdb)
  message(STATUS "Building only VDBVoxelgrid C++ module")
  include(GNUInstallDirs)
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_INSTALL_FULL_LIBDIR}/cmake/OpenVDB")
  set(VDBVOXELGRID_STANDALONE ON)
  find_package(OpenVDB REQUIRED)
endif()

find_package(OpenMP REQUIRED)

target_sources(vdbvoxelgrid PRIVATE VoxelGrid.cpp)
target_compile_features(vdbvoxelgrid PUBLIC cxx_std_17)
target_compile_options(vdbvoxelgrid PRIVATE -Wall -Wextra -O3)
set_target_properties(vdbvoxelgrid PROPERTIES PUBLIC_HEADER "VoxelGrid.h")
target_include_directories(
  vdbvoxelgrid PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
                                                       $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(vdbvoxelgrid PUBLIC OpenVDB::openvdb PRIVATE OpenMP::OpenMP_CXX)

# NOTE: Installation is still work in progress, therefore we only support dev machines with all the 3rdparty
# dependencies locally installed
if((USE_SYSTEM_OPENVDB) OR VDBVOXELGRID_STANDALONE)
  include(cmake/Install.cmake)
endif()
