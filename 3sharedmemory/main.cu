#include <iostream>

template <typename T>
class CudaMemory 
{
private:
    T* pointer;
    size_t memSize;

    void memCpy(T* dst, const T* src, cudaMemcpyKind kind)
    {
        auto result = cudaMemcpy(dst, src, this->memSize, kind)
        if (result != cudaSuccess)
        {
            throw "Cuda memcpy error!";
        } 
    }
public:
    CudeMemory(size_t mem) : memSize(mem) 
    {
        auto result = cudaMalloc(&this->pointer, mem);
        if (result != cudaSuccess)
        {
            throw "Cuda malloc error!";
        }
    }

    ~CudaMemory()
    {
        cudaFree(this->pointer);
    }

    T* getPointer() { return this->pointer; }

    void memCpyToHost(T* dst)
    {
        this->memCpy(dst, this->pointer, cudaMemcpyDeviceToHost);
    }

    void memCpyToDevice(const T* src)
    {
        this->memCpy(this->pointer, src, cudaMemcpyHostToDevice);
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
void testCase()
{

}

int main() 
{
    testCase<int>();
    testCase<float>();
    testCase<double>();
    testCase<long long>();
    
    return 0;
}