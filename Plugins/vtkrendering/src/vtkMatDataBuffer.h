#pragma once
#include <vtkTextureObject.h>
#include <opencv2/core/opengl.hpp>

class PLUGIN_EXPORTS vtkMatDataBuffer: public vtkTextureObject
{
public:
    static vtkMatDataBuffer* New();
    vtkTypeMacro(vtkMatDataBuffer, vtkTextureObject);

    
#ifdef AUTO_BUFFERS
    EagleLib::pool::Ptr<cv::ogl::Buffer> data_buffer;
#else
    cv::ogl::Buffer data_buffer;
#endif
protected:
    vtkMatDataBuffer();
};

class PLUGIN_EXPORTS vtkTextureDataBuffer: public vtkMatDataBuffer
{
public:
    static vtkTextureDataBuffer* New();
    vtkTypeMacro(vtkTextureDataBuffer, vtkMatDataBuffer);
    void compile_texture();
protected:
    vtkTextureDataBuffer();
};

class PLUGIN_EXPORTS vtkVertexDataBuffer: public vtkMatDataBuffer
{
public:
    static vtkVertexDataBuffer* New();
    vtkTypeMacro(vtkVertexDataBuffer, vtkMatDataBuffer);
protected:
    vtkVertexDataBuffer();
};