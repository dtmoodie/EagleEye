#include "nodes/Node.h"

namespace EagleLib
{
    bool CV_EXPORTS loadPlugin(const std::string& fullPluginPath);
	std::vector<std::string> EAGLE_EXPORTS ListLoadedPlugins();
}
