#include "mil_boost.hpp"

using namespace EagleLib;
using namespace EagleLib::ML;
using namespace EagleLib::ML::classifiers;
using namespace EagleLib::ML::classifiers::MIL;


void mil_tree::Init(bool firstInit)
{
    if(firstInit)
    {
        num_features = -1;
    }
    
}
void mil_tree::Serialize(ISimpleSerializer* pSerializer)
{

}
std::vector<Parameters::Parameter::Ptr> mil_tree::GetParameters()
{
    return this->parameters;
}
