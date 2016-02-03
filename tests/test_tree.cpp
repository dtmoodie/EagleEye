#include <EagleLib/nodes/NodeManager.h>
#include "EagleLib/nodes/Node.h"

int main()
{
    auto node = EagleLib::NodeManager::getInstance().addNode("SerialStack");
    return 0;
}
