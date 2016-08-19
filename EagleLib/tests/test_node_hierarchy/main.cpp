#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN
#include "EagleLib/nodes/Node.h"

#include "MetaObject/Parameters/ParameterMacros.hpp"
#include "MetaObject/Parameters/TypedInputParameter.hpp"
#include "MetaObject/MetaObjectFactory.hpp"

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE "EagleLibNodes"

#include <boost/test/unit_test.hpp>
#include <iostream>
using namespace EagleLib;
using namespace EagleLib::Nodes;

struct node_a: public Nodes::Node
{

};

struct node_b: public Nodes::Node
{

};



BOOST_AUTO_TEST_CASE(no_branching)
{

}

BOOST_AUTO_TEST_CASE(branching)
{

}

BOOST_AUTO_TEST_CASE(diamond)
{

}

BOOST_AUTO_TEST_CASE(delete_node)
{

}
