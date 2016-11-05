#pragma once

#define CATCH_MACRO                                                         \
    catch(Signals::ExceptionWithCallStack<cv::Exception>& e)                \
{                                                                            \
    NODE_LOG(error) << e.what() << "\n" << e.CallStack();                    \
}                                                                            \
catch (boost::thread_resource_error& err)                                    \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
catch (boost::thread_interrupted& err)                                      \
{                                                                           \
    NODE_LOG(error) << "Thread interrupted";                                \
    /* Needs to pass this back up to the chain to the processing thread.*/  \
    /* That way it knowns it needs to exit this thread */                   \
    throw err;                                                              \
}                                                                           \
catch (boost::thread_exception& err)                                        \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (cv::Exception &err)                                              \
{                                                                           \
    NODE_LOG(error) << err.what();                                          \
}                                                                           \
    catch (boost::exception &err)                                           \
{                                                                           \
    NODE_LOG(error) << "Boost error";                                       \
}                                                                           \
catch (std::exception &err)                                                 \
{                                                                           \
    NODE_LOG(error) << err.what();                                            \
}                                                                           \
catch (...)                                                                 \
{                                                                           \
    NODE_LOG(error) << "Unknown exception";                                 \
}
