#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include "EagleLib/nodes/Node.h"


#include "EagleLib/Logging.h"
#include "EagleLib/nodes/NodeInfo.hpp"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Detail/MetaObjectMacros.hpp"
#include "MetaObject/MetaObjectFactory.hpp"
#include "MetaObject/Parameters/Buffers/StreamBuffer.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibFrameGrabbers"
#include <boost/thread.hpp>
#include <boost/test/unit_test.hpp>
#include <iostream>

using namespace EagleLib;
using namespace EagleLib::Nodes;

struct test_node: public Node
{
    static std::vector<std::string> GetNodeCategory()
    {
        return {"test1", "test2"};
    }

    bool ProcessImpl()
    {
        return true;
    }

    MO_DERIVE(test_node, Node)
        MO_SLOT(void, node_slot, int);
        MO_SIGNAL(void, node_signal, int);
        PARAM(int, test_param, 5);
    MO_END;
};
void test_node::node_slot(int val)
{
    
}


MO_REGISTER_CLASS(test_node);

BOOST_AUTO_TEST_CASE(test_node_reflection)
{
    mo::MetaObjectFactory::Instance()->RegisterTranslationUnit();
    auto info = mo::MetaObjectFactory::Instance()->GetObjectInfo("test_node");
    auto node_info = dynamic_cast<NodeInfo*>(info);
    BOOST_REQUIRE(node_info);
    BOOST_REQUIRE_EQUAL(node_info->GetNodeCategory().size(), 2);
    BOOST_REQUIRE_EQUAL(node_info->ListParameters().size(), 1);
    BOOST_REQUIRE_EQUAL(node_info->ListSignalInfo().size(), 2);
    BOOST_REQUIRE_EQUAL(node_info->ListSlotInfo().size(), 2);
}