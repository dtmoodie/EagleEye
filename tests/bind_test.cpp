
#include <boost/function.hpp>
#include <boost/bind.hpp>
#include <iostream>
void testFunfionct(int x)
{
    std::cout << x << std::endl;
}

int main()
{
    boost::function<void(void)> f =  boost::bind(testFunfionct, 5);
    f();

    return 0;
}
