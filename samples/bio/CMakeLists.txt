if(TARGET aqbio)
    set(CMAKE_CXX_STANDARD 14)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    add_executable(bio_example src/main.cpp)

    target_link_libraries(bio_example aqbio aqframegrabbers aqcore ${Boost_LIBRARIES})
endif()
