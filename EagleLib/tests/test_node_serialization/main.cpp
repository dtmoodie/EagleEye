#define BOOST_TEST_MAIN
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>
#include <MetaObject/Parameters/IO/TextPolicy.hpp>
#include <MetaObject/Parameters/Types.hpp>
#include "MetaObject/Parameters/MetaParameter.hpp"
#include "MetaObject/IMetaObject.hpp"
#include "MetaObject/Signals/TypedSignal.hpp"
#include "MetaObject/Detail/Counter.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Signals/detail/SignalMacros.hpp"
#include "MetaObject/Signals/detail/SlotMacros.hpp"
#include "MetaObject/Parameters//ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedParameterPtr.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/Logging/CompileLogger.hpp"
#include "MetaObject/Parameters/Buffers/BufferFactory.hpp"
#include "MetaObject/IO/Policy.hpp"
#include "MetaObject/IO/memory.hpp"
#include "shared_ptr.hpp"
#include "RuntimeObjectSystem.h"
#include "shared_ptr.hpp"
#include "IObjectFactorySystem.h"
#include "cereal/archives/xml.hpp"
#include "cereal/archives/portable_binary.hpp"
#include <fstream>
#include "../MetaObject/instantiations/instantiate.hpp"

#include <opencv2/core.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <EagleLib/SyncedMemory.h>
#include <boost/thread.hpp>
#include <iostream>

using namespace mo;

struct serializable_object : public IMetaObject
{
    MO_BEGIN(serializable_object);
    PARAM(int, test, 5);
    PARAM(int, test2, 6);
    MO_END;
};



BuildCallback* cb = nullptr;
MO_REGISTER_OBJECT(serializable_object);

BOOST_AUTO_TEST_CASE(synced_mem_to_json)
{
    EagleLib::SyncedMemory synced_mem(cv::Mat(320,240, CV_32FC3));
    
    mo::TypedParameterPtr<EagleLib::SyncedMemory> param;
    param.SetName("Matrix");
    param.UpdatePtr(&synced_mem);
    auto func = mo::SerializationFunctionRegistry::Instance()->GetJsonSerializationFunction(param.GetTypeInfo());
    BOOST_REQUIRE(func);
    std::ofstream ofs("synced_memory_json.json");
    BOOST_REQUIRE(ofs.is_open());
    cereal::JSONOutputArchive ar(ofs);
    func(&param,ar);
}