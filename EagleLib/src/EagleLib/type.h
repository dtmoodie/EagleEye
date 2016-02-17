#ifndef TYPE_HPP
#define TYPE_HPP

#include <EagleLib/Defs.hpp>
#include <string>
#include <typeinfo>
namespace TypeInfo
{
	std::string EAGLE_EXPORTS demangle(const char* name);

template <class T>
std::string type(const T& t) {

    return demangle(typeid(t).name());
}
}
//#endif
#endif
