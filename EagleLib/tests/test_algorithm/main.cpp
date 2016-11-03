#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <EagleLib/Algorithm.h>

#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"
#include "MetaObject/Parameters/Buffers/BufferPolicy.hpp"
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibAlgorithm"

#include <boost/test/unit_test.hpp>
#include <boost/thread.hpp>
#include <iostream>
using namespace EagleLib;


class int_output: public Algorithm
{
public:
    bool ProcessImpl()
    {
        ++value;
        return true;
    }
    MO_BEGIN(int_output);
        OUTPUT(int, value, 0);
    MO_END;
};

class int_input: public Algorithm
{
public:
    bool ProcessImpl()
    {
        if(input)
            value = *input;
        return true;
    }

    MO_BEGIN(int_input);
        INPUT(int, input, nullptr);
    MO_END;
    int value;
};

class synced_input: public Algorithm
{
public:
    bool ProcessImpl()
    {
        BOOST_REQUIRE_EQUAL(timestamp, input_param.GetTimestamp());
        return true;
    }
    MO_BEGIN(synced_input);
        INPUT(int, input, nullptr);
    MO_END;
    int timestamp;
};

class multi_input: public Algorithm
{
public:
    bool ProcessImpl()
    {
        BOOST_REQUIRE_EQUAL(*input1, *input2);
        return true;
    }

    MO_BEGIN(multi_input)
        INPUT(int, input1, nullptr);
        INPUT(int, input2, nullptr);
    MO_END;
};


MO_REGISTER_OBJECT(int_output);
MO_REGISTER_OBJECT(int_input);
MO_REGISTER_OBJECT(multi_input);

BOOST_AUTO_TEST_CASE(initialize)
{
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
}


BOOST_AUTO_TEST_CASE(test_no_input)
{
    auto obj = rcc::shared_ptr<int_output>::Create();
    for(int i = 0; i < 100; ++i)
    {
        obj->Process();
    }
    
    BOOST_REQUIRE_EQUAL(obj->value, 100);
}

BOOST_AUTO_TEST_CASE(test_counting_input)
{
    auto output = rcc::shared_ptr<int_output>::Create();
    auto input = rcc::shared_ptr<int_input>::Create();
    auto output_param = output->GetOutput<int>("value");
    auto input_param = input->GetInput<int>("input");
    BOOST_REQUIRE(output_param);
    BOOST_REQUIRE(input_param);
    BOOST_REQUIRE(input_param->SetInput(output_param));
    for(int i = 0; i < 100; ++i)
    {
        output->Process();
        input->Process();
        BOOST_REQUIRE_EQUAL(output->value, *input->input);
    }
}

BOOST_AUTO_TEST_CASE(test_synced_input)
{
    mo::TypedParameter<int> output;
    output.UpdateData(10, 0);
    auto input = rcc::shared_ptr<int_input>::Create();
    input->input_param.SetInput(&output);
    input->SetSyncInput("input");

    for(int i = 0; i < 100; ++i)
    {
        output.UpdateData(i+ 1, i);
        BOOST_REQUIRE(input->Process());
        BOOST_REQUIRE_EQUAL(input->value, output.GetData((long long)i));
    }
}

BOOST_AUTO_TEST_CASE(test_threaded_input)
{
    mo::Context ctx;
    mo::TypedParameter<int> output("test", 0, mo::Control_e, 0, &ctx);

    auto obj = rcc::shared_ptr<int_input>::Create();
    boost::thread thread([&obj, &output]()->void
    {
        mo::Context _ctx;
        obj->SetContext(&_ctx);
        BOOST_REQUIRE(obj->ConnectInput("input", nullptr, &output));
        int success_count = 0;
        while(success_count < 1000)
        {
            if(obj->Process())
            {
                ++success_count;
            }
        }
        obj->SetContext(nullptr, true);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    });

    for(int i = 0; i < 1000; ++i)
    {
        output.UpdateData(i, i, &ctx);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }

    thread.join();
}

BOOST_AUTO_TEST_CASE(test_desynced_nput)
{
    mo::Context ctx;
    mo::TypedParameter<int> fast_output("test", 0, mo::Control_e, 0, &ctx);
    mo::TypedParameter<int> slow_output("test2", 0, mo::Control_e, 0, &ctx);
    int* addr1 = fast_output.GetDataPtr();
    int* addr2 = slow_output.GetDataPtr();

    auto obj = rcc::shared_ptr<multi_input>::Create();

    bool thread1_done = false;
    bool thread2_done = false;

    boost::thread thread([&obj, &fast_output, &slow_output, &thread1_done, addr1, addr2]()->void
    {
        mo::Context _ctx;
        obj->SetContext(&_ctx);
        BOOST_REQUIRE(obj->ConnectInput("input1", nullptr, &fast_output));
        BOOST_REQUIRE(obj->ConnectInput("input2", nullptr, &slow_output));

        int success_count = 0;
        while(success_count < 999)
        {
            if(obj->Process())
            {
                ++success_count;
                if(boost::this_thread::interruption_requested())
                    break;
            }
        }
        thread1_done = true;
        obj->SetContext(nullptr, true);
    });
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    boost::thread slow_thread(
        [&slow_output, &thread2_done]()->void
    {
        mo::Context _ctx;
        slow_output.SetContext(&_ctx);
        for(int i = 1; i < 1000; ++i)
        {
            slow_output.UpdateData(i, i, &_ctx);
            boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
            if(boost::this_thread::interruption_requested())
                break;
        }
        thread2_done = true;
        slow_output.SetContext(nullptr);
    });
    

    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    for(int i = 1; i < 1000; ++i)
    {
        fast_output.UpdateData(i, i, &ctx);
        boost::this_thread::sleep_for(boost::chrono::milliseconds(1));
    }
    while(thread2_done == false)
    {
        boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    }
    boost::this_thread::sleep_for(boost::chrono::milliseconds(100));
    BOOST_REQUIRE(thread1_done);
    BOOST_REQUIRE(thread2_done);
    slow_thread.join();
    thread.join();
}

BOOST_AUTO_TEST_CASE(cleanup)
{
    
}
