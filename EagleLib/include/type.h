#ifndef TYPE_HPP
#define TYPE_HPP
#ifndef _MSC_VER

#include <string>
#include <typeinfo>
namespace type_info
{
std::string demangle(const char* name);

template <class T>
std::string type(const T& t) {

    return demangle(typeid(t).name());
}
}
#endif
#endif
