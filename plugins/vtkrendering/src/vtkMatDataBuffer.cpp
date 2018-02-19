#include "vtkMatDataBuffer.h"
#include <MetaObject/Logging/Log.hpp>
#include <vtkOpenGLRenderWindow.h>
#ifdef _MSC_VER
#include <Windows.h>
#endif
#include <gl/GL.h>

vtkMatDataBuffer::vtkMatDataBuffer() : vtkTextureObject()
{
}
vtkMatDataBuffer* vtkMatDataBuffer::New()
{
    return new vtkMatDataBuffer();
}
vtkTextureDataBuffer* vtkTextureDataBuffer::New()
{
    return new vtkTextureDataBuffer();
}
vtkTextureDataBuffer::vtkTextureDataBuffer() : vtkMatDataBuffer()
{
}
void vtkTextureDataBuffer::compile_texture()
{
    try
    {
#ifdef AUTO_BUFFER
        if (image_buffer)
        {

            image_buffer->bind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
            if ((this->Width != image_buffer->cols() || this->Height != image_buffer->rows() ||
                 this->Components != image_buffer->channels()) &&
                Context)
            {
                InternalFormat = GL_RGB8;
                int vtk_type = 0;
                switch (image_buffer->depth())
                {
                case CV_8U:
                    vtk_type = VTK_UNSIGNED_CHAR;
                    break;
                case CV_16U:
                    vtk_type = VTK_UNSIGNED_SHORT;
                    break;
                case CV_32S:
                    vtk_type = VTK_INT;
                    break;
                case CV_32F:
                    vtk_type = VTK_FLOAT;
                    break;
                case CV_64F:
                    vtk_type = VTK_DOUBLE;
                    break;
                }
                Allocate2D(image_buffer->cols(), image_buffer->rows(), image_buffer->channels(), vtk_type);
            }
            else
            {
                this->Activate();
                glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, this->Format, this->Type, NULL);
            }
            this->Deactivate();
            image_buffer->unbind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
            Modified();
        }
#else
        data_buffer.bind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
        if ((this->Width != data_buffer.cols() || this->Height != data_buffer.rows() ||
             this->Components != data_buffer.channels()) &&
            Context)
        {
            switch (data_buffer.type())
            {
            case CV_8U:
                InternalFormat = GL_INTENSITY8;
                break;
            case CV_8UC3:
                InternalFormat = GL_RGB8;
                break;
            default:
                throw "Invalid datatype";
            }
            InternalFormat = GL_RGB8;
            int vtk_type = 0;
            switch (data_buffer.depth())
            {
            case CV_8U:
                vtk_type = VTK_UNSIGNED_CHAR;
                break;
            case CV_16U:
                vtk_type = VTK_UNSIGNED_SHORT;
                break;
            case CV_32S:
                vtk_type = VTK_INT;
                break;
            case CV_32F:
                vtk_type = VTK_FLOAT;
                break;
            case CV_64F:
                vtk_type = VTK_DOUBLE;
                break;
            }
            Allocate2D(data_buffer.cols(), data_buffer.rows(), data_buffer.channels(), vtk_type);
        }
        else
        {
            this->Activate();
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->Width, this->Height, this->Format, this->Type, NULL);
        }
        this->Deactivate();
        data_buffer.unbind(cv::ogl::Buffer::PIXEL_UNPACK_BUFFER);
        Modified();

#endif
    }
    catch (cv::Exception& e)
    {
        LOG(error) << e.what();
    }
    catch (...)
    {
    }
}

vtkVertexDataBuffer::vtkVertexDataBuffer() : vtkMatDataBuffer()
{
}
vtkVertexDataBuffer* vtkVertexDataBuffer::New()
{
    return new vtkVertexDataBuffer();
}
