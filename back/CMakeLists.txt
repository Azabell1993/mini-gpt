cmake_minimum_required(VERSION 3.15)
project(mini_gpt_back LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 컴파일러 최적화 플래그
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

add_executable(back 
    main.cpp 
    advanced_tensor.cpp
)

# Crow + standalone ASIO 헤더 경로
target_include_directories(back PRIVATE
    ../third_party/crow/include
    ../third_party/asio/asio/include
    .
)

# standalone ASIO 사용 선언
target_compile_definitions(back PRIVATE 
    ASIO_STANDALONE
    _WIN32_WINNT=0x0601  # Windows 호환성
)