#pragma once
#include "EaglePython_defs.h"
#include <EagleLib/rcc/shared_ptr.hpp>
#include <EagleLib/nodes/Node.h>
#include <EagleLib/nodes/NodeManager.h>

#include "PythonQt/PythonQtCppWrapperFactory.h"
#include <PythonQt/PythonQt.h>

#include <QObject>

#include <memory>

namespace EagleLib
{
	class DataStream;
	class DataStreamManager;
	namespace Nodes
	{
		class Node;
	}
	class NodeManager;
	class ObjectManager;

	namespace python
	{
		namespace wrappers
		{
			typedef rcc::shared_ptr<EagleLib::Nodes::Node> NodePtr;
			class EAGLEPYTHON_EXPORTS NodeWrapper: public QObject
			{
				Q_OBJECT
				public Q_SLOTS:
					void delete_NodePtr(NodePtr* o);
					QString GetName(NodePtr* node);
					QString GetFullName(NodePtr* node);
					QStringList ListParameters(NodePtr* node);
					void AddNode(NodePtr* node, QString name);
			};
			class EAGLEPYTHON_EXPORTS DataStreamWrapper: public QObject
			{
				Q_OBJECT
			public Q_SLOTS:
				void Step(DataStream* stream);
				void Play(DataStream* stream);
				void Pause(DataStream* stream);
				QStringList ListNodes(DataStream* stream);
				NodePtr GetNode(DataStream* stream, QString name);
				NodePtr GetNode(DataStream* stream, int index);
				void AddNode(DataStream* stream, QString name);
			};
			void EAGLEPYTHON_EXPORTS RegisterMetaTypes();
		}
		class EAGLEPYTHON_EXPORTS EaglePython: public QObject
		{
			Q_OBJECT
		public:
			EaglePython();
		public Q_SLOTS:
			void LoadPlugin(QString path);
			QStringList ListPlugins();
			QStringList ListDevices();
			QStringList ListConstructableNodes();
			QStringList ListConstructableNodes(QString filter);
			
			void EaglePython::SetLogLevel(QString level);
			bool EaglePython::CheckRecompile();
			void EaglePython::AbortRecompile();
			DataStream* GetDataStream(int index);
			void ReleaseDataStream(int index);
			QStringList ListDataStreams();
			DataStream* OpenDataSource(QString source);
			

		protected:
			std::vector<std::shared_ptr<EagleLib::DataStream>> _streams;
		};
	}
}
