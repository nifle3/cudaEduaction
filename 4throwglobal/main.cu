#include <iostream>

__global__ void throw_check() {
    // this is not working shit (
    throw "something";
}

int main() {
    try {
        throw_check<<<1, 1>>>();

    } catch(...) {
        std::cout << "caught!" << std::endl;
    }

    return 0;
}