#include "mil_boost.hpp"

using namespace aq;
using namespace aq::ML;
using namespace aq::ML::classifiers;
using namespace aq::ML::classifiers::MIL;

void mil_tree::nodeInit(bool firstInit)
{
    if (firstInit)
    {
        num_features = -1;
    }
}

/*std::vector<Parameters::Parameter::Ptr> mil_tree::GetParameters()
{
    return this->parameters;
}*/
