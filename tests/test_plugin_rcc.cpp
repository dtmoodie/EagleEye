#include "MetaObject/core/SystemTable.hpp"
#include "MetaObject/object/MetaObjectFactory.hpp"
#include <RuntimeObjectSystem/IObjectFactorySystem.h>
#include <RuntimeObjectSystem/IRuntimeObjectSystem.h>
#include <boost/filesystem.hpp>
#include <iostream>

class BuildCallback : public ITestBuildNotifier
{
  public:
    BuildCallback(std::vector<std::string>& failures, std::vector<std::string>& success)
        : m_failures(failures), m_success(success)
    {
    }

    virtual bool TestBuildCallback(const char* file, TestBuildResult type)
    {
        std::cout << "[" << file << "] - ";
        switch (type)
        {
        case TESTBUILDRRESULT_SUCCESS:
            std::cout << "TESTBUILDRRESULT_SUCCESS\n";
            m_success.push_back(file);
            return true;
        case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
            std::cout << "TESTBUILDRRESULT_NO_FILES_TO_BUILD\n";
            break;
        case TESTBUILDRRESULT_BUILD_FILE_GONE:
            std::cout << "TESTBUILDRRESULT_BUILD_FILE_GONE\n";
            break;
        case TESTBUILDRRESULT_BUILD_NOT_STARTED:
            std::cout << "TESTBUILDRRESULT_BUILD_NOT_STARTED\n";
            break;
        case TESTBUILDRRESULT_BUILD_FAILED:
            std::cout << "TESTBUILDRRESULT_BUILD_FAILED\n";
            break;
        case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
            std::cout << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL\n";
            break;
        }
        m_failures.push_back(file);
        return false;
    }
    virtual bool TestBuildWaitAndUpdate() { return true; }
    std::vector<std::string>& m_failures;
    std::vector<std::string>& m_success;
};

int main(int argc, char** argv)
{
    (void)argc;
    SystemTable table;
    mo::MetaObjectFactory factory(&table);
    std::vector<std::string> failures, success;
    BuildCallback cb(failures, success);
    boost::filesystem::path currentDir = boost::filesystem::path(argv[0]).parent_path();
#ifdef _MSC_VER
    currentDir = boost::filesystem::path(currentDir.string());
#else
    currentDir = boost::filesystem::path(currentDir.string() + "/Plugins");
#endif

    boost::filesystem::directory_iterator end_itr;
    if (boost::filesystem::is_directory(currentDir))
    {
        for (boost::filesystem::directory_iterator itr(currentDir); itr != end_itr; ++itr)
        {
            if (boost::filesystem::is_regular_file(itr->path()))
            {
#ifdef _MSC_VER
                if (itr->path().extension() == ".dll")
#else
                if (itr->path().extension() == ".so")
#endif
                {
                    std::string file = itr->path().string();
                    mo::MetaObjectFactory::instance().loadPlugin(file);
                }
            }
        }
    }
    auto plugins = mo::MetaObjectFactory::instance().listLoadedPlugins();
    std::cout << "Testing the following plugins:\n";
    for (auto plugin : plugins)
        std::cout << plugin << std::endl;
    mo::MetaObjectFactory::instance().getObjectSystem()->TestBuildAllRuntimeSourceFiles(&cb, true);
    if (failures.size())
        std::cout << "The following files failed RCC:\n";
    for (auto file : failures)
    {
        std::cout << file << '\n';
    }
    std::cout << std::endl;
    if (success.size())
        std::cout << "The following files succeded:\n";
    for (auto file : success)
    {
        std::cout << file << '\n';
    }
    return 0;
}
