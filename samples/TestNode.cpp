#include <nodes/Node.h>


namespace EagleLib
{
    class TestObject: public TInterface<IID_NodeObject, IObject>
    {
    public:
        TestObject()
        {
            //int x = 0;
        }
        virtual ~TestObject() {}
    };

}
using namespace EagleLib;
REGISTERCLASS(TestObject)
