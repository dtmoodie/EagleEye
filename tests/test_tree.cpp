#include "Manager.h"
#include "Nodes/Node.h"

int main()
{
    auto node = EagleLib::NodeManager::getInstance().addNode("SerialStack");
    return 0;
}
