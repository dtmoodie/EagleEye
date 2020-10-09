SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}")
foreach(test ${tests})
    file(GLOB_RECURSE test_hdr "${test}/*.hpp" "${test}/*.h")
    file(GLOB_RECURSE test_src "${test}/*.cpp")
    file(GLOB_RECURSE test_knl "${test}/*.cu")
    cuda_add_executable(${test} ${test_hdr} ${test_src} ${test_knl})
    add_dependencies(${test} aqframegrabbers)
    target_link_libraries(${test} aqframegrabbers
        aquila_types
        gtest
        gtest_main
    )
    set_target_properties(${test} PROPERTIES FOLDER Tests/Aquila/frame_grabbers)
    add_test(${test} ${test})
    if(MSVC)
        CONFIGURE_FILE("../../../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/Plugins/framegrabbers/tests/${test}.vcxproj.user @ONLY)
    endif()
endforeach()
