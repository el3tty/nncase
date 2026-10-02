cmake_minimum_required(VERSION 3.18)
project(ai2d_test LANGUAGES CXX)

include(FetchContent)
 
FetchContent_Declare(cxxopts
  GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
  GIT_TAG        v3.3.1
  GIT_SHALLOW    TRUE)
 
FetchContent_Declare(xxhash
  GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
  GIT_TAG        v0.8.4
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  build/cmake)

set(LIBMMZ_REV ea816b267c431edd9a1a80574bded229010b1f15)
set(LIBMMZ_BASE https://raw.githubusercontent.com/kendryte/k230_linux_sdk/${LIBMMZ_REV}/buildroot-overlay/package/libmmz)
FetchContent_Declare(mmz_c URL ${LIBMMZ_BASE}/mmz.c DOWNLOAD_NO_EXTRACT TRUE)
FetchContent_Declare(mmz_h URL ${LIBMMZ_BASE}/mmz.h DOWNLOAD_NO_EXTRACT TRUE)

FetchContent_MakeAvailable(cxxopts xxhash mmz_c mmz_h)

add_library(mmz STATIC ${mmz_c_SOURCE_DIR}/mmz.c)
target_include_directories(mmz PUBLIC ${mmz_h_SOURCE_DIR})

add_executable(ai2d_test
  ai2d_test.cpp  
)

target_link_libraries(ai2d_test PRIVATE cxxopts)
target_link_libraries(ai2d_test PRIVATE xxhash)
target_link_libraries(ai2d_test PRIVATE nncaseruntime)
target_link_libraries(ai2d_test PRIVATE functional_k230)
target_link_libraries(ai2d_test PRIVATE nncase.rt_modules.k230)
target_link_libraries(ai2d_test PRIVATE mmz)
