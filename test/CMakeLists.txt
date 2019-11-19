# Add gtest include directories.
enable_testing()

# Unit testing exectuables.
add_executable(unit_tests
        test_gmm_stats.cpp
        test_ados.cpp
        test_egos.cpp
)
target_link_libraries(unit_tests gtest gtest_main gmm_stats mantrap)
add_test(NAME unit COMMAND ${CMAKE_BINARY_DIR}/${CMAKE_INSTALL_BINDIR}/unit_tests)
