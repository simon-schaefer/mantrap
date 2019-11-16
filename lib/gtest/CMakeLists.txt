# The following code to fetch googletest is inspired by and adapted after https://crascit.com/2015/07/25/cmake-gtest/.
# It downloads and unpacks googletest at configure time.
set(GOOGLETEST_DOWNLOAD_ROOT ${PROJECT_BINARY_DIR}/googletest)
configure_file(googletest_download.cmake ${PROJECT_BINARY_DIR}/googletest/CMakeLists.txt @ONLY)
unset(GOOGLETEST_DOWNLOAD_ROOT)

# Create cmake and build it (i.e. download and build gtest).
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest
        )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY ${PROJECT_BINARY_DIR}/googletest
        )

# Add the target: gtest, gtest_main, gmock, gmock_main
add_subdirectory(
        ${PROJECT_BINARY_DIR}/googletest/googletest-src
        ${PROJECT_BINARY_DIR}/googletest/googletest-build
)
