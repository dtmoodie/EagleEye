
#include <QtWidgets/QWidget>
#include <type_traits>

#include "Parameters.hpp"
#include <QtWidgets/QDoubleSpinBox>
#include "qwidget.h"
#include "qlayout.h"
#include "qlabel.h"

namespace Parameters{
	namespace UI{
		namespace qt{
			class SignalProxy;
			class IHandler;

			class QtRegistry
			{
			public:
				typedef std::function<IHandler*(Parameters::Parameter*)> HandlerFunctor;

			private:
				std::map<Loki::TypeInfo, HandlerFunctor> registry;
			};

			// IHandler class is the interface for all parmeter handlers.  It handles updating the user interface on parameter changes
			// as well as updating parameters on user interface changes
			class IHandler
			{
			protected:
				SignalProxy* proxy;
			public:
				virtual QWidget* GetWidget(QWidget* parent)
				{
					QWidget* widget = new QWidget(parent);

				}

			};

			// Relays signals from the QObject ui elements and sends them to the IHandler object
			class SignalProxy : QObject
			{
				Q_OBJECT
				IHandler* handler;
			public:
				SignalProxy(IHandler* handler_);

			public slots:
				void on_update();
				void on_update(int);
				void on_update(double);
				void on_update(QString);
			};

			template<typename T> class Handler : public IHandler
			{

			};


		} /* namespace qt */
	} /* namespace UI */
} /* namespace Perameters */

