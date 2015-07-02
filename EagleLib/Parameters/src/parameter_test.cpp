#include "../include/Parameters.hpp"
#include "../include/UI/Qt.hpp"

int main()
{
	{
		// In this use case, there are two userspace variables, and two Parameter objects.  The parameter objects
		// are constructed with pointers to the user space variables, and to automatically update the user space input
		// based on the Output parameter object.  
		int* userInput = nullptr;
		int userOutput = 50;
		int userOutput2 = 500;
		Parameters::TypedInputParameterPtr<int> typedInputParam(&userInput);
		Parameters::TypedParameterPtr<int>::Ptr typedParameter(new Parameters::TypedParameterPtr<int>(&userOutput));
		if (typedInputParam.SetInput(typedParameter))
		{
			userOutput = 60;
		}
		typedParameter->UpdateData(&userOutput2);
		*userInput == userOutput2;
	}

	{
		// In this example, param is the holder of the data.  Useful if data is a large object and we want to avoid large copies with serialization
		int* userInput = nullptr;
		auto param = Parameters::TypedParameter<int>::create(20);
		auto inputParam = Parameters::TypedInputParameterPtr<int>(&userInput);
		inputParam.SetInput(param);
		param->UpdateData(500);
		*userInput == *param->Data();
	}
	{
		// In this case everything is stored in the parameters
		auto param = Parameters::TypedParameter<double>::create(20.0);
		auto inputParameter = Parameters::TypedInputParameter<double>();
		inputParameter.SetInput(param);
		bool eqVal = *inputParameter.Data() == *param->Data();
		bool eqPtr = inputParameter.Data() == param->Data();
	}


    return 0;
}
