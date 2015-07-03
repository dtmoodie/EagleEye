
#include <QtWidgets/QWidget>
#include <type_traits>

#include "Parameters.hpp"
#include <QtWidgets/QDoubleSpinBox>

namespace Parameters
{
	class QtRegistry
	{
	public:
		typedef std::function<QWidget*(Parameters::Parameter*)> WidgetMaker;

	private:
		std::map<Loki::TypeInfo, WidgetMaker> registry;
	};









	/*
    class QtPolicy
    {
    public:
        virtual std::vector<QWidget*> GetSettingWidget() = 0;
    };

    template<typename T, template<typename> class ContainerPolicy = TypedParameterPolicy, typename Enable = void> class TypedQtPolicy: public QtPolicy, public ContainerPolicy<T>
    {
    public:
        typedef QtPolicy super_t;
        virtual std::vector<QWidget*> GetSettingWidget()
        {
            return std::vector<QWidget*>();
        }
    };

    // Specializations for flaots / doubles / etc
    template<typename T, template<typename> class ContainerPolicy>
    class TypedQtPolicy<T, ContainerPolicy,
            typename std::enable_if<std::is_floating_point<T>::value, void>::type>: public QtPolicy, public ContainerPolicy<T>
    {
        typedef QtPolicy super_t;
        virtual std::vector<QWidget*> GetSettingWidget()
        {
            return std::vector<QWidget*>(new QDoubleSpinBox());
        }

    };


    // Wrapper policy specializations are used when indexing is required, such as when there are a vector of floats.
    // The vector specialization is used for handling the indexing and the TypedQtPolicy<float> specialization is used to handle the
    // Inner data
    template<typename T, template<typename> class ItemPolicy, typename Enable = void>
    class WrapperPolicy: public ItemPolicy<T>
    {

    };

    template<typename T, template<typename> class ItemPolicy>
    class WrapperPolicy<std::vector<T>, ItemPolicy, void>: public QtPolicy, public ItemPolicy<T>
    {
        virtual std::vector<QWidget*> GetSettingWidget()
        {
            std::vector<QWidget*> widgets;
            widgets.push_back(new QSpinBox());
            auto superWidgets = ItemPolicy<T>::GetSettingWidget();
            widgets.insert(widgets.end(), superWidgets.begin(), superWidgets.end());
            return widgets;
        }
    };*/


}

