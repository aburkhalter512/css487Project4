#include <iostream>

//#define INCLUDE_LOGGING

#ifdef _DEBUG
#ifdef INCLUDE_LOGGING
#define LOG(str) std::cout << "[DEBUG]: " << str << std::endl
#else
#define LOG(str)
#endif
#else
#define LOG(str)
#endif