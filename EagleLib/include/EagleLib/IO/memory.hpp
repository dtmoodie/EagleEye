#pragma once
#include "EagleLib/Nodes/Node.h"
#include "shared_ptr.hpp"
#include <MetaObject/Logging/Log.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/xml.hpp>
#include <cereal/types/string.hpp>
namespace EagleLib
{
    EAGLE_EXPORTS bool Serialize(cereal::BinaryOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::BinaryInputArchive& ar, EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool Serialize(cereal::XMLOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::XMLInputArchive& ar, EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool Serialize(cereal::JSONOutputArchive& ar, const EagleLib::Nodes::Node* obj);
    EAGLE_EXPORTS bool DeSerialize(cereal::JSONInputArchive& ar, EagleLib::Nodes::Node* obj);
}
namespace cereal
{
    template<class AR, class T> void save(AR& ar, rcc::shared_ptr<T> const & m)
    {
        if(std::is_base_of<EagleLib::Nodes::Node, T>::value)
        {
            EagleLib::Serialize(ar, m.Get());
        }else
        {
            mo::Serialize(ar, m.Get());
        }
    }

    template<class AR, class T> void load(AR& ar, rcc::shared_ptr<T> & m)
    {
        if (!m)
        {
            std::string type;
            ar(make_nvp("TypeName", type));
            m = mo::MetaObjectFactory::Instance()->Create(type.c_str());
        }
        if (std::is_base_of<EagleLib::Nodes::Node, T>::value)
        {
            EagleLib::DeSerialize(ar, m.Get());
        }else
        {
            mo::DeSerialize(ar, m.Get());
        }
    }
    template<class AR, class T> void save(AR& ar, rcc::weak_ptr<T> const & m)
    {

    }

    template<class AR, class T> void load(AR& ar, rcc::weak_ptr<T> & m)
    {

    }
}