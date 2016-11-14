#pragma once

#include "EagleLib/Detail/Export.hpp"

namespace cereal
{
    class BinaryOutputArchive;
    class BinaryInputArchive;
    class XMLOutputArchive;
    class XMLInputArchive;
    class JSONOutputArchive;
    class JSONInputArchive;
}

namespace EagleLib
{
    namespace Nodes
    {
        class Node;
    }
    EAGLE_EXPORTS bool Serialize(cereal::BinaryOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::BinaryInputArchive& ar, EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool Serialize(cereal::XMLOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::XMLInputArchive& ar, EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool Serialize(cereal::JSONOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::JSONInputArchive& ar, EagleLib::Nodes::Node* obj);
}