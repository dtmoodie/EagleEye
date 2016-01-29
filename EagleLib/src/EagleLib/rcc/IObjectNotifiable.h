#pragma once
struct IObject;
class IObjectNotifiable
{
protected:
	friend struct IObject;
	virtual void updateObject(IObject* ptr) = 0;
};