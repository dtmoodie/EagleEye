cmake_minimum_required(VERSION 2.8)

project(cntk)

##### find cntk
set(CNTK_DIR "" CACHE PATH "Folder that contains cntk")
find_path(CNTK_INCLUDE_DIR EvalDll/CNTKEval.h PATHS ${CNTK_DIR})
find_path(CNTK_LIB_DIR EvalDll.lib PATHS ${CNTK_DIR}/lib)
set(CNTK_LIBS "")
find_library(CNTK_CNTK_LIBRARY NAMES CNTK HINTS "${CNTK_DIR}/lib")
find_library(CNTK_SequenceTraining_LIBRARY NAMES SequenceTrainingLib HINTS "${CNTK_DIR}/lib")
find_library(CNTK_MathCUDA_LIBRARY NAMES MathCUDA HINTS "${CNTK_DIR}/lib")
find_library(CNTK_Math_LIBRARY NAMES Math HINTS "${CNTK_DIR}/lib")
find_library(CNTK_Reader_LIBRARY NAMES ReaderLib HINTS "${CNTK_DIR}/lib")
find_library(CNTK_BinaryReader_LIBRARY NAMES BinaryReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_DSSMReader_LIBRARY NAMES DSSMReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_LMSequenceReader_LIBRARY NAMES LMSequenceReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_LUSequenceReader_LIBRARY NAMES LUSequenceReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_SparsePCReader_LIBRARY NAMES SparsePCReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_SVMBinaryReader_LIBRARY NAMES LIBSVMBinaryReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_UCIFastReader_LIBRARY NAMES UCIFastReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_HTKMLFReader_LIBRARY NAMES HTKMLFReader HINTS "${CNTK_DIR}/lib")
find_library(CNTK_ComputationNetwork_LIBRARY NAMES ComputationNetworkLib HINTS "${CNTK_DIR}/lib")
find_library(CNTK_SGD_LIBRARY NAMES SGDLib HINTS "${CNTK_DIR}/lib")
find_library(CNTK_Eval_LIBRARY NAMES EvalDll HINTS "${CNTK_DIR}/lib")
find_library(CNTK_Actions_LIBRARY NAMES ActionsLib HINTS "${CNTK_DIR}/lib")
if(CNTK_CNTK_LIBRARY AND 
   CNTK_SequenceTraining_LIBRARY AND 
   CNTK_MathCUDA_LIBRARY AND 
   CNTK_Reader_LIBRARY AND 
   CNTK_BinaryReader_LIBRARY AND 
   CNTK_DSSMReader_LIBRARY AND 
   CNTK_LMSequenceReader_LIBRARY AND
   CNTK_LUSequenceReader_LIBRARY AND
   CNTK_SparsePCReader_LIBRARY AND
   CNTK_SVMBinaryReader_LIBRARY AND
   CNTK_UCIFastReader_LIBRARY AND
   CNTK_HTKMLFReader_LIBRARY AND
   CNTK_ComputationNetwork_LIBRARY AND
   CNTK_SGD_LIBRARY AND
   CNTK_Eval_LIBRARY AND 
   CNTK_Actions_LIBRARY)

  list(APPEND CNTK_LIBS ${CNTK_CNTK_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_SequenceTraining_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_MathCUDA_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_Reader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_BinaryReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_DSSMReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_LMSequenceReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_LUSequenceReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_SparsePCReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_SVMBinaryReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_UCIFastReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_HTKMLFReader_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_ComputationNetwork_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_SGD_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_Eval_LIBRARY})
  list(APPEND CNTK_LIBS ${CNTK_Actions_LIBRARY})

##### end find cntk

  include_directories(${CNTK_INCLUDE_DIR})
  link_directories(${CNTK_LIB_DIR})
  link_directories(${CMAKE_ARCHIVE_OUTPUT_DIRECTORY})
  add_library(cntk SHARED cntk.cpp)
  RCC_LINK_LIB(cntk ${CNTK_LIBS} Aquila)
  INCLUDE(../PluginTemplate.cmake)
endif()