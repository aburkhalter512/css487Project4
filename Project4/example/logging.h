#include <iostream>

#ifdef _DEBUG
#define LOG(str) std::cout << "[DEBUG]: " << str << std::endl
#else
#define LOG(str)
#endif