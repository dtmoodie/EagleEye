find_path(cereal_INCLUDE_DIRS 
			cereal/cereal.hpp 
			PATHS ${cereal_DIR})
if(cereal_INCLUDE_DIRS)
  message("found cereal")
  set(cereal_FOUND true)
endif()