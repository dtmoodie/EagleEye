#include "ptCloudStreamer.h"
#include <QtNetwork/qhostaddress.h>
#include <parameters/ParameteredObjectImpl.hpp>
using namespace EagleLib;
using namespace EagleLib::Nodes;
ServerHandler::ServerHandler()
{
	server = new QTcpServer();
	//server->listen(, 10000);
	server->listen(QHostAddress::Any, 10000);
	
	connect(server, SIGNAL(newConnection()), this, SLOT(on_newConnection()));
	
}
void ServerHandler::on_newConnection()
{
	connections.push_back(server->nextPendingConnection());

}
void ServerHandler::send()
{
	std::vector<cv::Vec3f> binaryData;
	int size = 100;
	binaryData.reserve(size);
	for (int i = 0; i < size; ++i)
	{
		binaryData.push_back(cv::Vec3f(1000 * rand() / RAND_MAX, 1000 * rand() / RAND_MAX, 1000 * rand() / RAND_MAX));
	}
	QByteArray buffer; 
	
	buffer.reserve(sizeof(int) + sizeof(float) * 3 * binaryData.size());
	memcpy(buffer.data(), &size, sizeof(int));
	memcpy(buffer.data() + 4, binaryData.data(), sizeof(float) * 3);
	for (int i = 0; i < connections.size(); ++i) 
	{
		connections[i]->write(buffer); 
	}
}
void Server::NodeInit(bool firstInit)
{
	handler.reset(new ServerHandler); 
}

cv::cuda::GpuMat Server::doProcess(cv::cuda::GpuMat& img, cv::cuda::Stream& stream)
{
	
	if (handler)
	{
		handler->send();
	}
	return img; 
}

NODE_DEFAULT_CONSTRUCTOR_IMPL(Server, PtCloud, Sink);
