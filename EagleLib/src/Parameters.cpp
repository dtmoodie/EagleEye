#include "Parameters.h"
namespace EagleLib
{
//#define SERIALIZE_TYPE(type) template<> inline void TypedParameter<type>::Serialize(cv::FileStorage& fs){    \
//    Parameter::Serialize(fs);                                                                         \
//    fs << "Data" << data;                                                                             \
//}

//    SERIALIZE_TYPE(double)
//    SERIALIZE_TYPE(float)
//    SERIALIZE_TYPE(char)
//    SERIALIZE_TYPE(unsigned char)
//    SERIALIZE_TYPE(short)
//    SERIALIZE_TYPE(unsigned short)
//    SERIALIZE_TYPE(int)
//    template<> void TypedParameter<EnumParameter>::Serialize(cv::FileStorage& fs)
//    {
//        Parameter::Serialize(fs);
//        fs << "Enumerations" << "[:";
//        for(int i = 0; i < data.enumerations.size(); ++i)
//        {
//            fs << data.enumerations[i];
//        }
//        fs << "]";
//        fs << "Values" << "[:";
//        for(int i = 0; i < data.values.size(); ++i)
//        {
//            fs << data.values[i];
//        }
//        fs << "]";
//        fs << "CurrentSelection" << data.currentSelection;
//    }
}


