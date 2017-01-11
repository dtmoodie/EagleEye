#include <MetaObject/Parameters/IO/CerealPolicy.hpp>
#include <MetaObject/Parameters/IO/CerealMemory.hpp>
#include "CaffeNetHandler.hpp"
#include "EagleLib/IO/JsonArchive.hpp"
#include <cereal/types/vector.hpp>
#include <cereal/types/string.hpp>

INSTANTIATE_META_PARAMETER(std::vector<rcc::shared_ptr<EagleLib::Caffe::NetHandler>>)
