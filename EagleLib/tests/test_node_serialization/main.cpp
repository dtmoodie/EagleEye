#define BOOST_TEST_MAIN
#include <EagleLib/IDataStream.hpp>
#include <EagleLib/IO/JsonArchive.hpp>
#include <EagleLib/DataStream.hpp>

#include <MetaObject/Thread/ThreadPool.hpp>
#include <MetaObject/MetaObject.hpp>
#include <MetaObject/Parameters/IO/SerializationFunctionRegistry.hpp>

#ifdef _MSC_VER
#include <boost/test/unit_test.hpp>
#else
#define BOOST_TEST_MODULE __FILE__
#include <boost/test/included/unit_test.hpp>
#endif
#include <EagleLib/SyncedMemory.h>
#include <boost/thread.hpp>
#include <boost/filesystem.hpp>
#include <iostream>
#include <fstream>

using namespace mo;

//MO_REGISTER_OBJECT(serializable_object);
BOOST_AUTO_TEST_CASE(initialize)
{
    boost::filesystem::path currentDir = boost::filesystem::current_path();
#ifdef _MSC_VER
#ifdef _DEBUG
    currentDir = boost::filesystem::path(currentDir.string() + "/../Debug/");
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/../RelWithDebInfo/");
#endif
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
#endif
    LOG(info) << "Looking for plugins in: " << currentDir.string();
    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(currentDir))
    {
        for (boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
#ifdef _MSC_VER
                if (itr->path().extension() == ".dll")
#else
                if (itr->path().extension() == ".so")
#endif
                {
                    std::string file = itr->path().string();
                    mo::MetaObjectFactory::Instance()->LoadPlugin(file);
                }
            }
        }
    }
}

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
    EagleLib::JSONOutputArchive ar(ofs);
    func(&param,ar);
}

BOOST_AUTO_TEST_CASE(datastream)
{
    auto ds = EagleLib::IDataStream::Create("", "TestFrameGrabber");
    std::ofstream ofs("datastream.json");
    BOOST_REQUIRE(ofs.is_open());
    EagleLib::JSONOutputArchive ar(ofs);
    ds->AddNode("QtImageDisplay");
    auto disp = ds->GetNode("QtImageDisplay0");
    auto fg = ds->GetNode("TestFrameGrabber0");
    disp->ConnectInput(fg, "current_frame", "image");
    ar(ds);
}

BOOST_AUTO_TEST_CASE(read_datastream)
{
    rcc::shared_ptr<EagleLib::IDataStream> stream = rcc::shared_ptr<EagleLib::DataStream>::Create();
    std::ifstream ifs("datastream.json");
    BOOST_REQUIRE(ifs.is_open());
    std::map<std::string, std::string> dummy;
    EagleLib::JSONInputArchive ar(ifs, dummy, dummy);
    ar(stream);
}

BOOST_AUTO_TEST_CASE(cleanup)
{
    mo::ThreadPool::Instance()->Cleanup();
}
