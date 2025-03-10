#include <iostream>
#include <string>
#include <random>
#include <functional>
#include <type_traits>
#include <vector>
#include <thread>

template <typename A, typename = typename std::enable_if<std::is_arithmetic<A>::value>::type>
__global__ void addTwoVector(A* aArr, A* bArr, A* cArr, int N);

template <typename A>
void printVector(const std::vector<A>& vec, const std::string& vectorName);

template <typename A, typename = typename std::enable_if<std::is_arithmetic<A>::value>::type>
void fillRandomVector(std::vector<A>& vec, int N);

template <typename A, typename = typename std::enable_if<std::is_arithmetic<A>::value>::type>
std::function<A()> getRandomGenerator();

template <typename A, typename = typename std::enable_if<std::is_arithmetic<A>::value>::type>
void calculateThirdVector(int N);

void cudaMallocWrapper(void** ptr, size_t mem);

int main() 
{
    std::cout << "For float" << std::endl;
    constexpr int NFirst = 100000;
    calculateThirdVector<float>(NFirst);
    std::cout << std::endl << std::endl << std::endl;


    std::cout << "For int" << std::endl;
    constexpr int NSecond = 100000;
    calculateThirdVector<int>(NSecond);
    std::cout << std::endl << std::endl << std::endl;
}

template <typename A, typename>
void calculateThirdVector(int N) 
{
    const size_t memory = N * sizeof(A);
    std::vector<A> h_A(N);
    std::vector<A> h_B(N);
    std::vector<A> h_C(N);

    std::thread t1(fillRandomVector<A>, std::ref(h_A), N);
    std::thread t2(fillRandomVector<A>, std::ref(h_B), N);
    if (t1.joinable()) 
    {
        t1.join();
    }

    if (t2.joinable())
    {
        t2.join();
    }

    A* d_A;
    A* d_B;
    A* d_C; 
    if (cudaMalloc(&d_A, memory) != cudaSuccess || 
        cudaMalloc(&d_B, memory) != cudaSuccess || 
        cudaMalloc(&d_C, memory) != cudaSuccess) 
    {
        std::cerr << "Cuda malloc failed!" << std::endl;
        return;
    }


   if (cudaMemcpy(d_A, h_A.data(), memory, cudaMemcpyHostToDevice) != cudaSuccess || 
        cudaMemcpy(d_B, h_B.data(), memory, cudaMemcpyHostToDevice) != cudaSuccess) 
    {
        std::cerr << "Cuda memcpy failed!" << std::endl;
        return;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addTwoVector<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    if (cudaMemcpy(h_C.data(), d_C, memory, cudaMemcpyDeviceToHost) != cudaSuccess) 
    {
        std::cerr << "Cuda memcpy failed!" << std::endl;
        return;
    }

    printVector<A>(h_A, "A");
    printVector<A>(h_B, "B");
    printVector(h_C, "C");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename A, typename>
void fillRandomVector(std::vector<A>& vec, int N)
{
    thread_local static auto randomGenerator = getRandomGenerator<A>();
    for (int i = 0; i < N; i++) 
    {
        vec[i] = randomGenerator();
    }
}

template <typename A, typename>
std::function<A()> getRandomGenerator() {
    std::random_device rd;
    std::default_random_engine dre(rd());

    if constexpr (std::is_floating_point_v<A>) 
    {
        std::uniform_real_distribution<A> dist_real(1, 100);
        return [dre = std::move(dre), dist_real]() mutable {
            return dist_real(dre);
        };
    }
    else 
    {
        std::uniform_int_distribution<A> dist_int(1, 100);
        return [dre = std::move(dre), dist_int]() mutable {
            return dist_int(dre);
        };
    }
}

template <typename A>
void printVector(const std::vector<A>& vec, const std::string& vectorName) 
{
    std::cout << "Vector name is: " << vectorName << std::endl;

    for (const A& element : vec) 
    {
        std::cout << element << " ";
    }

    std::cout << std::endl;
}

template <typename A, typename>
__global__ void addTwoVector(A* aArr, A* bArr, A* cArr, int N) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
    {
        cArr[i] = aArr[i] + bArr[i];
    }
}