#pragma once

namespace EagleLib
{
	class EventHandler;
}
namespace Freenect
{
	class Freenect;
}
struct SystemTable
{
	SystemTable();

	EagleLib::EventHandler* eventHandler;
	Freenect::Freenect* freenect;

};