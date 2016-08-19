#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include <EagleLib/Algorithm.h>

#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/IMetaObjectImpl.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibNodes"

#include <boost/test/unit_test.hpp>
#include <iostream>
using namespace EagleLib;


class int_output: public Algorithm
{
public:
    void ProcessImpl()
    {
        ++value;
    }
    MO_BEGIN(int_output);
        OUTPUT(int, value, 0);
    MO_END;
};

class int_input: public Algorithm
{
public:
    void ProcessImpl()
    {
        if(input)
            value = *input;
    }

    MO_BEGIN(int_input);
        INPUT(int, input, nullptr);
    MO_END;
    int value;
};
class synced_input: public Algorithm
{
public:
    void ProcessImpl()
    {
    
    }
    MO_BEGIN(synced_input);
        INPUT(int, input, nullptr);
    MO_END;
};


MO_REGISTER_OBJECT(int_output);
MO_REGISTER_OBJECT(int_input);


BOOST_AUTO_TEST_CASE(initialize)
{
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
}


BOOST_AUTO_TEST_CASE(test_no_input)
{
    auto obj = rcc::shared_ptr<int_output>::Create();
    for(int i = 0; i < 100; ++i)
    {
        obj->CheckInputs();
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
        if(output->CheckInputs())
        {
            output->Process();
        }
        if(input->CheckInputs())
        {
            input->Process();
        }
        BOOST_REQUIRE_EQUAL(output->value, *input->input);
    }
}

BOOST_AUTO_TEST_CASE(test_synced_input)
{
    mo::TypedParameter<int> output;
    output.UpdateData(10, 0);
    auto input = rcc::shared_ptr<int_input>::Create();
    input->input_param.SetInput(&output);
    input->SetSyncInput(&input->input_param);

    for(int i = 0; i < 100; ++i)
    {
        output.UpdateData(i+ 1, i);
        BOOST_REQUIRE(input->CheckInputs());
        input->Process();
        BOOST_REQUIRE_EQUAL(input->value, output.GetData(i));
    }
}
