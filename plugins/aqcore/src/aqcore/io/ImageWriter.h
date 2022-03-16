#pragma once
#include <aqcore_export.hpp>

#include <Aquila/types/SyncedImage.hpp>
#include <Aquila/nodes/Node.hpp>
#include <MetaObject/thread/Thread.hpp>

#include <RuntimeObjectSystem/RuntimeInclude.h>
#include <RuntimeObjectSystem/RuntimeSourceDependency.h>
RUNTIME_COMPILER_SOURCEDEPENDENCY
RUNTIME_MODIFIABLE_INCLUDE

namespace aq
{
    namespace nodes
    {

        class ImageWriter : public Node
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
                INPUT(SyncedImage, input_image)
                PARAM(std::string, base_name, "Image-")
                ENUM_PARAM(extension, png, jpg, tiff, bmp)
                PARAM(int, frequency, 30)
#ifdef _MSC_VER
                PARAM(mo::WriteDirectory, save_directory, mo::WriteDirectory("C:/tmp"))
#else
                PARAM(mo::WriteDirectory, save_directory, mo::WriteDirectory("/tmp"))
#endif
                STATE(int32_t, frame_count, 0)
                PARAM(bool, request_write, false)
                MO_SLOT(void, snap)
            MO_END;

          protected:

            void nodeInit(bool first) override;
            bool processImpl() override;
            mo::Thread m_worker_thread;
            mo::IAsyncStream::Ptr_t m_worker_stream;
        };
    } // namespace nodes
} // namespace aq
