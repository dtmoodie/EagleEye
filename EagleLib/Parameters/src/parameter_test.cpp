#include "../include/Parameters.hpp"
#include "../include/UI/Qt.hpp"

int main()
{
    Parameters::TypedParameterPolicy<float> param;
    param.Data() = 10;
    //Parameters::PointerParameterPolicy<float> ptrParam;
    //ptrParam.UpdateData(param.Data());
    //ptrParam.Data();
    //Parameters::ITypedParameter<float>* ptr = &ptrParam;

    //Parameters::ParameterBase<std::vector<float>, TypedParameterPolicy, WrapperPolicy> test;


    return 0;
}
