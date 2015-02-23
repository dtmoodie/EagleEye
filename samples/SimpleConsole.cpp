
#include "EagleLib.h"
#include "Manager.h"
#include "nodes/Node.h"

int main()
{
    EagleLib::NodeManager manager;
    manager.addNode("TestNode");
    while(1)
    {
        manager.CheckRecompile();
    }
}
