#pragma once


#include <src/precompiled.hpp>
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE
namespace aq
{
    namespace Nodes
    {
    
    class ImageWriter: public Node
    {
        enum Extensions
        {
            jpg = 0,
            png,
            tiff,
            bmp
        };
        
        bool writeRequested;
        int frameSkip;
    public:
        
        MO_DERIVE(ImageWriter, Node)
            INPUT(SyncedMemory, input_image, nullptr)
            PARAM(std::string, base_name, "Image-")
            ENUM_PARAM(extension, jpg, png, tiff, bmp)
            PARAM(int, frequency, 30)
        #ifdef _MSC_VER
            PARAM(mo::WriteDirectory, save_directory, mo::WriteDirectory("C:/tmp"))
        #else
            PARAM(mo::WriteDirectory, save_directory, mo::WriteDirectory("/tmp"))
        #endif
            STATUS(int, frame_count, 0)
            PARAM(bool, request_write, false)
            MO_SLOT(void, snap)
        MO_END;
    protected:
        bool ProcessImpl();
        
    };
    }
}
