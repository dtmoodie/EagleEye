#include "ObjectManager.h"
#include <RuntimeObjectSystem.h>
#include <EagleLib/rcc/SystemTable.hpp>
#include <EagleLib/rcc/IObjectNotifiable.h>

#include "parameters/Parameter_def.hpp"

#include <MetaObject/Logging/Log.hpp>

#include <boost/tokenizer.hpp>
#include <boost/foreach.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/date_time/posix_time/ptime.hpp>
#include <boost/date_time/microsec_time_clock.hpp>
#include <boost/thread.hpp>
#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>

#include <opencv2/core/cuda.hpp>

#include "remotery/lib/Remotery.h"

#include "cuda.h"
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"

#include <stdarg.h>


using namespace EagleLib;

#ifdef _WIN32
#include "Windows.h"
#pragma warning( disable : 4996 4800 )
#endif

void CompileLogger::log(int level, const char *format, va_list args)
{
    vsnprintf(m_buff, LOGSYSTEM_MAX_BUFFER - 1, format, args);
    // Make sure there's a limit to the amount of rubbish we can output
    m_buff[LOGSYSTEM_MAX_BUFFER - 1] = '\0';
    if (callback)
        callback(std::string(m_buff), level);
    switch (level)
    {
    case 0:
    {
        LOG(trace) << "[RCC] " << m_buff;
        break;
    }
    case 1:
    {
        LOG(debug) << "[RCC] " << m_buff;
        break;
    }
    case 2:
    {
        LOG(info) << "[RCC] " << m_buff;
        break;
    }
    }


#ifdef _WIN32
    OutputDebugStringA(m_buff);
#endif
}

void CompileLogger::LogError(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(2, format, args);
}

void CompileLogger::LogWarning(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(1, format, args);

}

void CompileLogger::LogInfo(const char * format, ...)
{
    va_list args;
    va_start(args, format);
    log(0, format, args);
}
bool TestCallback::TestBuildCallback(const char* file, TestBuildResult type)
{
    bool success = true;
    switch (type)
    {
    case TESTBUILDRRESULT_SUCCESS:
        LOG(info) << "TESTBUILDRRESULT_SUCCESS - " << file;
        break;
    case TESTBUILDRRESULT_NO_FILES_TO_BUILD:
        LOG(info) << "TESTBUILDRRESULT_NO_FILES_TO_BUILD - " << file;
        success = false;
        break;
    case TESTBUILDRRESULT_BUILD_FILE_GONE:
        LOG(info) << "TESTBUILDRRESULT_BUILD_FILE_GONE - " << file;
        success = false;
        break;
    case TESTBUILDRRESULT_BUILD_NOT_STARTED:
        LOG(info) << "TESTBUILDRRESULT_BUILD_NOT_STARTED - " << file;
        success = false;
        break;
    case TESTBUILDRRESULT_BUILD_FAILED:
        LOG(info) << "TESTBUILDRRESULT_BUILD_FAILED - " << file;
        success = false;
        break;
    case TESTBUILDRRESULT_OBJECT_SWAP_FAIL:
        LOG(info) << "TESTBUILDRRESULT_OBJECT_SWAP_FAIL - " << file;
        success = false;
        break;
    }
    //BOOST_LOG(info) << file;
    return success;
}

bool TestCallback::TestBuildWaitAndUpdate()
{
    boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    return true;
}
// ******************************************************************************************************
//                                Object Manager
// ******************************************************************************************************
ObjectManager::ObjectManager()
{
    m_pTestCallback = nullptr;
    m_pRuntimeObjectSystem.reset(new RuntimeObjectSystem);
    m_pCompileLogger.reset(new CompileLogger());
    m_systemTable.reset(new SystemTable());
    m_pRuntimeObjectSystem->Initialise(m_pCompileLogger.get(), m_systemTable.get());
    m_systemTable->SetSingleton<ObjectManager>(this);
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddListener(this);
    boost::filesystem::path workingDir(__FILE__);
    std::string includePath = workingDir.parent_path().parent_path().string();
    //m_pRuntimeObjectSystem->SetAdditionalLinkOptions(" -DPARAMETERS_NO_UI ");
    m_pRuntimeObjectSystem->SetAdditionalCompileOptions(" -DPARAMTERS_NO_UI ");
#ifdef _MSC_VER

#else
    m_pRuntimeObjectSystem->SetAdditionalCompileOptions("-std=c++11");
#endif // _MSC_VER

#ifdef _DEBUG
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_DEBUG);
#else
    m_pRuntimeObjectSystem->SetOptimizationLevel(RCCPPOPTIMIZATIONLEVEL_PERF);
#endif // _DEBUG
#ifdef _MSC_VER
    includePath += "\\include";
#else
    includePath += "/include";
#endif
    m_pRuntimeObjectSystem->AddIncludeDir(includePath.c_str());
#ifdef NVCC_PATH
    m_pRuntimeObjectSystem->SetCompilerLocation(NVCC_PATH);
#endif
    cv::cuda::GpuMat mat(10, 10, CV_32F);

    try
    {
        Remotery* rmt;
        rmt_CreateGlobalInstance(&rmt);

        CUcontext ctx;
        cuCtxGetCurrent(&ctx);

        rmtCUDABind bind;
        bind.context = ctx;
        bind.CtxSetCurrent = (void*)&cuCtxSetCurrent;
        bind.CtxGetCurrent = (void*)&cuCtxGetCurrent;
        bind.EventCreate = (void*)&cuEventCreate;
        bind.EventDestroy = (void*)&cuEventDestroy;
        bind.EventRecord = (void*)&cuEventRecord;
        bind.EventQuery = (void*)&cuEventQuery;
        bind.EventElapsedTime = (void*)&cuEventElapsedTime;
        rmt_BindCUDA(&bind);
        rmt_BindOpenGL();
    }catch(...)
    {
        LOG(warning) << "Unable to start remotery";
    }
    
}
ObjectManager& ObjectManager::Instance()
{
    static ObjectManager* inst = nullptr;
    if (inst == nullptr)
    {
        inst = new ObjectManager();
    }
    return *inst;
}
rcc::shared_ptr<IObject> ObjectManager::GetObject(const std::string& object_name)
{
    auto constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(object_name.c_str());
    if (constructor)
    {
        assert(!constructor->GetIsSingleton());
        auto object = constructor->Construct();
        return rcc::shared_ptr<IObject>(object);
    }
    LOG(warning) << "Constructor for " << object_name << " doesn't exist";
    return rcc::shared_ptr<IObject>();
}
IObject* ObjectManager::GetObject(ObjectId oid)
{
    if(!oid.IsValid())
        return nullptr;
    auto constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(oid.m_ConstructorId);
    return constructor->GetConstructedObject(oid.m_PerTypeId);
}
rcc::weak_ptr<IObject> ObjectManager::GetSingleton(const std::string& object_name)
{
    auto constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(object_name.c_str());
    if (constructor)
    {
        assert(constructor->GetIsSingleton());
        auto object = constructor->Construct();
        return rcc::weak_ptr<IObject>(object);
    }
    LOG(warning) << "Constructor for " << object_name << " doesn't exist";
    return rcc::weak_ptr<IObject>();
}
bool ObjectManager::TestRuntimeCompilation()
{
    
    if (m_pTestCallback == nullptr)
        m_pTestCallback = new TestCallback();
    m_pRuntimeObjectSystem->TestBuildAllRuntimeHeaders(m_pTestCallback, true);
    m_pRuntimeObjectSystem->TestBuildAllRuntimeSourceFiles(m_pTestCallback, true);
    return true;
}
void ObjectManager::addIncludeDir(const std::string& dir, unsigned short projId)
{
    m_pRuntimeObjectSystem->AddIncludeDir(dir.c_str(), projId);
}
void ObjectManager::addIncludeDirs(const std::string& dirs, unsigned short projId)
{
    if (!dirs.size())
        return;
    boost::char_separator<char> sep("+");
    boost::tokenizer<boost::char_separator<char>> tokens(dirs, sep);
    BOOST_FOREACH(const std::string& t, tokens)
    {
        addIncludeDir(t, projId);
    }
}
void ObjectManager::addLinkDirs(const std::string& dirs, unsigned short projId)
{
    
    if (!dirs.size())
        return;
    boost::char_separator<char> sep("+");
    boost::tokenizer<boost::char_separator<char>> tokens(dirs, sep);
    BOOST_FOREACH(const std::string& t, tokens)
    {
        addLinkDir(t, projId);
    }
}
void ObjectManager::addLinkDir(const std::string& dir, unsigned short projId)
{
    m_pRuntimeObjectSystem->AddLibraryDir(dir.c_str(), projId);
}
void ObjectManager::addDefinitions(const std::string& defs, unsigned short projId)
{
    
    if (!defs.size())
        return;
    boost::char_separator<char> sep("+");
    boost::tokenizer<boost::char_separator<char>> tokens(defs, sep);
    BOOST_FOREACH(const std::string& t, tokens)
    {
        m_pRuntimeObjectSystem->SetAdditionalCompileOptions(t.c_str(), projId);
    }
}
int ObjectManager::parseProjectConfig(const std::string& file)
{
    if (file.size())
    {
        boost::filesystem::path pfile(file);
        std::string root = pfile.filename().string();
        std::string projectName = root.substr(0, root.size() - 11);

        int id = m_pRuntimeObjectSystem->ParseConfigFile(file.c_str());
        if (id != -1)
        {
            m_projectNames[id] = projectName;
            return id;
        }
    }
    return 0;
}


std::vector<std::string> ObjectManager::getLinkDirs(unsigned short projId)
{
    std::vector<std::string> output;
    auto inc = m_pRuntimeObjectSystem->GetLinkDirList(projId);
    for (int i = 0; i < inc.size(); ++i)
    {
        output.push_back(inc[i].m_string);
    }
    return output;
}

std::vector<std::string> ObjectManager::getIncludeDirs(unsigned short projId)
{
    
    std::vector<std::string> output;
    auto inc = m_pRuntimeObjectSystem->GetIncludeDirList(projId);
    for (int i = 0; i < inc.size(); ++i)
    {
        output.push_back(inc[i].m_string);
    }
    return output;
}
RCppOptimizationLevel ObjectManager::getOptimizationLevel()
{
    
    return m_pRuntimeObjectSystem->GetOptimizationLevel();
}

void ObjectManager::setOptimizationLevel(RCppOptimizationLevel level)
{
    
    m_pRuntimeObjectSystem->SetOptimizationLevel(level);
}
int ObjectManager::getNumLoadedModules()
{
    return m_pRuntimeObjectSystem->GetNumberLoadedModules();
}

bool
ObjectManager::CheckRecompile(bool swapAllowed)
{
    
    static boost::posix_time::ptime prevTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::ptime currentTime = boost::posix_time::microsec_clock::universal_time();
    boost::posix_time::time_duration delta = currentTime - prevTime;
    // Prevent checking too often
    if (delta.total_milliseconds() < 10)
        return false;
    prevTime = currentTime;
    if (m_pRuntimeObjectSystem->GetIsCompiledComplete() && swapAllowed)
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
        std::lock_guard<std::mutex> lock(mtx);
        for(auto itr = m_sharedPtrs.begin(); itr !=  m_sharedPtrs.end(); ++itr)
        {
            (*itr)->notify_swap();    
        }
    }
    if (m_pRuntimeObjectSystem->GetIsCompiling())
    {
        return true;
    }
    else
    {
        m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(float(delta.total_milliseconds()) / 1000.0);
    }
    return false;
}
bool ObjectManager::CheckRecompile() const
{
    m_pRuntimeObjectSystem->GetFileChangeNotifier()->Update(float(100));
    return CheckIsCompiling();
}
bool ObjectManager::PerformSwap()
{
    if (m_pRuntimeObjectSystem->GetIsCompiledComplete())
    {
        m_pRuntimeObjectSystem->LoadCompiledModule();
        std::lock_guard<std::mutex> lock(mtx);
        for(auto itr = m_sharedPtrs.begin(); itr !=  m_sharedPtrs.end(); ++itr)
        {
            (*itr)->notify_swap();    
        }
        return true;
    }
    return false;
}
bool ObjectManager::CheckIsCompiling() const
{
    return m_pRuntimeObjectSystem->GetIsCompiling();
}

void ObjectManager::RegisterConstructorAddedCallback(std::function<void(void)> f)
{
    if (f)
        onConstructorsAddedCallbacks.push_back(f);
}
void
ObjectManager::setupModule(IPerModuleInterface* pPerModuleInterface)
{
    
    auto constructors = pPerModuleInterface->GetConstructors();
    int projectId = 0;
    if (constructors.size())
    {
        projectId = constructors[0]->GetProjectId();
#ifdef BUILD_DIR
#ifdef _DEBUG

        addLinkDir(BUILD_DIR "/Debug", projectId);
#else
        addLinkDir(BUILD_DIR "/RelWithDebInfo", projectId);
#endif
#endif
        if(build_dir.size())
        {
            addLinkDir(build_dir, projectId);
        }
    }
    m_pRuntimeObjectSystem->SetupObjectConstructors(pPerModuleInterface);
}
void ObjectManager::set_build_directory(const std::string& build_directory)
{
    build_dir = build_directory;
}
void ObjectManager::addSourceFile(const std::string &file)
{
    m_pRuntimeObjectSystem->AddToRuntimeFileList(file.c_str());
}
void ObjectManager::setCompileCallback(std::function<void(const std::string &, int)> &f)
{
    
    m_pCompileLogger->callback = f;
}
std::vector<std::pair<std::string, int>> ObjectManager::getObjectList()
{
    
    std::vector<std::pair<std::string, int>> output;
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    for (int i = 0; i < constructors.Size(); ++i)
    {
        output.push_back(std::make_pair(std::string(constructors[i]->GetName()), int(constructors[i]->GetProjectId())));
    }
    return output;
}
int ObjectManager::getProjectCount()
{
    return    m_pRuntimeObjectSystem->GetProjectCount();
}
std::string ObjectManager::getProjectName(int idx)
{
    auto itr = m_projectNames.find(idx);
    if (itr != m_projectNames.end())
    {
        return itr->second;
    }
    return boost::lexical_cast<std::string>(idx);
}

std::vector<std::string> ObjectManager::getLinkDependencies(const std::string& objectName)
{
    
    IObjectConstructor* constructor = m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetConstructor(objectName.c_str());
    
    std::vector<std::string> linkDependency;
    if (constructor)
    {
        int linkLibCount = constructor->GetMaxNumLinkLibraries();
        linkDependency.reserve(linkLibCount);
        for (int i = 0; i < linkLibCount; ++i)
        {
            const char* lib = constructor->GetLinkLibrary(i);
            if (lib)
                linkDependency.push_back(std::string(lib));
        }
    }
    return linkDependency;
}
void
ObjectManager::OnConstructorsAdded()
{
    
    for (int i = 0; i < onConstructorsAddedCallbacks.size(); ++i)
    {
        onConstructorsAddedCallbacks[i]();
    }
}
void
ObjectManager::addConstructors(IAUDynArray<IObjectConstructor*> & constructors)
{
    
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->AddConstructors(constructors);
}
void ObjectManager::abort_compilation()
{
    m_pRuntimeObjectSystem->AbortCompilation();
}
std::vector<IObjectConstructor*> ObjectManager::GetConstructorsForInterface(int interface_id)
{
    AUDynArray<IObjectConstructor*> constructors;
    m_pRuntimeObjectSystem->GetObjectFactorySystem()->GetAll(constructors);
    std::vector<IObjectConstructor*> output;
    for(int i = 0; i < constructors.Size(); ++i)
    {
        if(constructors[i]->GetInterfaceId() == interface_id)
        {
            output.push_back(constructors[i]);
        }
    }
    return output;
}
void ObjectManager::register_notifier(IObjectNotifiable* obj)
{
    std::lock_guard<std::mutex> lock(mtx);
    m_sharedPtrs.push_back(obj);
}
void ObjectManager::remove_notifier(IObjectNotifiable* obj)
{
    std::lock_guard<std::mutex> lock(mtx);
    auto itr = std::find(m_sharedPtrs.begin(), m_sharedPtrs.end(), obj);
    if(itr != m_sharedPtrs.end())
        m_sharedPtrs.erase(itr);
}