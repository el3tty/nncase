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

FetchContent_MakeAvailable(cxxopts xxhash)

add_executable(ai2d_test
  ai2d_test.cpp  
)

target_link_libraries(ai2d_test PRIVATE cxxopts)
target_link_libraries(ai2d_test PRIVATE xxhash)
target_link_libraries(ai2d_test PRIVATE nncaseruntime)
target_link_libraries(ai2d_test PRIVATE functional_k230)
target_link_libraries(ai2d_test PRIVATE nncase.rt_modules.k230)
target_link_libraries(ai2d_test PRIVATE mmz)
