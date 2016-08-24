#include "nodes/Node.h"

namespace EagleLib
{
    bool EAGLE_EXPORTS loadPlugin(const std::string& fullPluginPath);
    std::vector<std::string> EAGLE_EXPORTS ListLoadedPlugins();
    void EAGLE_EXPORTS Init();
}
