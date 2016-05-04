#pragma once

#include <PythonQt/PythonQt.h>
#include <QObject>

namespace EagleLib
{
	namespace python
	{
		class NodeManager: public QObject
		{
			Q_OBJECT
		public:
			NodeManager();

		public Q_SLOTS:
			QStringList ListConstructableNodes();
			


		};

		class DataStream: public QObject
		{
			Q_OBJECT
		public:
			DataStream();

		public Q_SLOTS:
			QStringList ListChildren();
			QStringList ListParameters();
		};


		class Node: public QObject
		{
			Q_OBJECT
		public:
			Node();

		public Q_SLOTS:
			QStringList ListChildren();
			QStringList ListParameters();

		};
	}
}
