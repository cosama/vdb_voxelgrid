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
ExternalProject_Add(
  external_tbb
  PREFIX tbb
  URL https://github.com/uxlfoundation/oneTBB/archive/refs/tags/v2022.0.0.tar.gz
  URL_HASH SHA256=e8e89c9c345415b17b30a2db3095ba9d47647611662073f7fbf54ad48b7f3c2a
  UPDATE_COMMAND ""
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=<INSTALL_DIR>
             ${ExternalProject_CMAKE_ARGS}
             ${ExternalProject_CMAKE_CXX_FLAGS}
             # custom flags
             -DTBB_STRICT=OFF
             -DTBBMALLOC_BUILD=ON
             -DBUILD_SHARED_LIBS=OFF
             -DTBB_TEST=OFF)

# Simulate importing TBB::tbb for OpenVDBHelper target
ExternalProject_Get_Property(external_tbb INSTALL_DIR)
set(TBB_ROOT ${INSTALL_DIR} CACHE INTERNAL "TBB_ROOT Install directory")
add_library(TBBHelper INTERFACE)
add_dependencies(TBBHelper external_tbb)
target_include_directories(TBBHelper INTERFACE ${INSTALL_DIR}/include)
target_link_directories(TBBHelper INTERFACE ${INSTALL_DIR}/lib)
target_link_libraries(TBBHelper INTERFACE tbb)
add_library(TBB::tbb ALIAS TBBHelper)
