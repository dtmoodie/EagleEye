SUBDIRLIST(tests "${CMAKE_CURRENT_LIST_DIR}")
foreach(test ${tests})
    file(GLOB_RECURSE test_hdr "${test}/*.hpp" "${test}/*.h")
    file(GLOB_RECURSE test_src "${test}/*.cpp")
    file(GLOB_RECURSE test_knl "${test}/*.cu")
    cuda_add_executable(${test} ${test_hdr} ${test_src} ${test_knl})
    add_dependencies(${test} frame_grabbers)
    target_link_libraries(${test} frame_grabbers
        aquila_types
        aquila_metatypes
    )
    set_target_properties(${test} PROPERTIES FOLDER Tests/Aquila/frame_grabbers)
    add_test(${test} ${test})
    if(MSVC)
        CONFIGURE_FILE("../../../Aquila/tests/Test.vcxproj.user.in" ${CMAKE_BINARY_DIR}/Plugins/frame_grabbers/tests/${test}.vcxproj.user @ONLY)
    endif()
endforeach()
