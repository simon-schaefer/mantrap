# Add gtest include directories.
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})

# Unit testing exectuables.
add_executable(unit_tests
    test_types.cpp
    test_2.cpp
)
target_link_libraries(unit_tests gtest_main gtest_main)
add_test(NAME unit COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests)