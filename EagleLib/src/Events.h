#pragma once

#include <boost/signals2.hpp>
#include <memory>
#include "IObject.h"
#include "IRuntimeObjectSystem.h"
namespace EagleLib
{
	struct EventBase
	{
		enum EventType
		{
			standard = 0, // Standard message
			image_object = 1,
			video_action = 2
		};
		virtual std::string& name() const;
		virtual std::string& message() const;
		virtual EventType type() const;
	};

	struct standardEvent : public EventBase
	{

	};
	struct ImageEvent : public EventBase
	{

	};

	struct VideoEvent : public EventBase
	{
		
	};

	class EventFilter
	{
	public:
		EventFilter();
		~EventFilter();
		bool acceptsEvent(EventBase* ev);
		bool handleEvent(EventBase* ev);
	};

	class EventHandler : public TInterface<IID_EventHandler, IObject>
	{
	public:
		EventHandler();
		static EventHandler* instance();
		void post_event(std::shared_ptr<EventBase>& ev);
		void register_filter(EventFilter* filter);
		void deregister_filter(EventFilter* filter);
	private:
		std::vector<EventFilter*> filters;
	};
}