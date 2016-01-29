#ifndef TYPE_HPP
#define TYPE_HPP
//#define CVAPI_EXPORTS
#include "opencv2/core/cvdef.h"
#include <string>
#include <typeinfo>
namespace TypeInfo
{
std::string CV_EXPORTS demangle(const char* name);

template <class T>
std::string type(const T& t) {

    return demangle(typeid(t).name());
}
}
//#endif
#endif
