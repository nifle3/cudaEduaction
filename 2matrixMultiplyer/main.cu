#include <vector>
#include <functional>
#include <random>
#include <iostream>
#include <chrono>
#include <string>
#include <memory>
#include <curand_kernel.h>

template <typename T>
class CudaMemory
{
private:
    T* _pointer;
public:
    CudaMemory(size_t mem) 
    {
        auto result = cudaMalloc(&_pointer, mem);
        if (result != cudaSuccess)
        {
            throw "Cuda malloc exception";
        }
    }

    ~CudaMemory() 
    {
        cudaFree(_pointer);
    }

    T* getPointer() 
    {
        return this->_pointer;
    }

    void memCpyToDevice(const T* src, size_t memSize)
    {
        auto result = cudaMemcpy(this->_pointer, src, memSize, cudaMemcpyHostToDevice);
    
        if (result != cudaSuccess)
        {
            throw "Cuda memcpy error!";
        }
    }

    void memCpyFromDeviceToHost(T* dst, size_t memSize)
    {
        auto result = cudaMemcpy(dst, this->_pointer, memSize, cudaMemcpyDeviceToHost);
    
        if (result != cudaSuccess)
        {
            throw "Cuda memcpy error!";
        }
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class Matrix {
private:
    std::vector<T> _data;
    size_t _col;
    size_t _row;

public:
    Matrix(size_t row, size_t col): _data(), _col(col), _row(row) 
    {
        this->_data.resize(row * col);
    }

    ~Matrix() = default;

    Matrix(const Matrix& other) = default;

    Matrix(Matrix&& other) = default;

    T* operator[](size_t i) {
        if (i >= _row)
        {
            throw std::out_of_range("Row index out of bounds!");
        }

        return _data.data() + i * _col;
    }

    size_t getRowCount() const noexcept
    {
        return this->_row;
    }

    size_t getColCount() const noexcept
    {
        return this->_col;
    }

    size_t getMemSize() const noexcept
    {
        return sizeof(T) * this->_col * this->_row;
    }

    void print() const 
    {
        std::cout << "Matrix \n";
        for (size_t i = 0; i < this->_row; i++)
        {
            for (size_t j = 0; j < this->_col; j++)
            {
                std::cout << this->_data[i * _col + j] << " ";
            }

            std::cout << std::endl;
        }
    }

    T* getData() 
    {
        return this->_data.data();
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class RandomMatrixFactory
{
public:
    virtual Matrix<T> generateRandom(size_t row, size_t col) = 0;
};

__global__ void initializeStates2D(curandState* states, int rows, int cols, unsigned long long seed) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        curand_init(seed, idx, 0, &states[idx]);
    }
}

template <typename T>
__global__ void generateRandomNumbers2D(T* matrix, curandState* states, int rows, int cols, T min_val, T max_val)  
{
    const size_t i = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t j = threadIdx.y + blockIdx.y * blockDim.y;
    const size_t idx = i * cols + j;
    if (i < rows && j < cols) {
        if constexpr (std::is_floating_point_v<T>) {
            matrix[idx] = min_val + (max_val - min_val) * curand_uniform(&states[idx]);
        }
        else if constexpr (std::is_integral_v<T>) {
            matrix[idx] = min_val + (curand(&states[idx]) % (max_val - min_val + 1));
        }
    }
}

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class GpuRandomMatrixFactory : public RandomMatrixFactory<T>
{
private:
    T _min_val;
    T _max_val;
    unsigned long long _seed;

    static constexpr size_t MAX_STATES = 1024 * 1024;
    
public:
    GpuRandomMatrixFactory(T min_val = 0, T max_val = 100, unsigned long long seed = 1234)
        : _min_val(min_val), _max_val(max_val), _seed(seed) {
    }
    
    bool checkAvailableMemory(size_t requiredMem) {
        size_t free, total;
        cudaMemGetInfo(&free, &total);
        return free >= requiredMem;
    }
    
    Matrix<T> generateRandom(size_t rows, size_t cols) override {
        Matrix<T> result(rows, cols);
        size_t matrixMemSize = result.getMemSize();
        size_t totalElements = rows * cols;
    
        CudaMemory<T> deviceMatrix(matrixMemSize);
        
        size_t stateMemorySize = sizeof(curandState) * totalElements;
        bool useChunks = !checkAvailableMemory(stateMemorySize) || totalElements > MAX_STATES;
        
        dim3 threadsPerBlock(16, 16);
        
        if (!useChunks) {
            
            CudaMemory<curandState> deviceStates(stateMemorySize);
            
            dim3 blocksPerGrid(
                (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                (rows + threadsPerBlock.y - 1) / threadsPerBlock.y
            );
            
            int maxGridSize[3];
            cudaDeviceGetAttribute(&maxGridSize[0], cudaDevAttrMaxGridDimX, 0);
            cudaDeviceGetAttribute(&maxGridSize[1], cudaDevAttrMaxGridDimY, 0);
            
            if (blocksPerGrid.x > maxGridSize[0] || blocksPerGrid.y > maxGridSize[1]) {
                throw "Grid size exceeds device limits!";
            }
            
            initializeStates2D<<<blocksPerGrid, threadsPerBlock>>>(
                deviceStates.getPointer(), 
                rows, 
                cols, 
                _seed
            );
            cudaDeviceSynchronize();
            
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                throw cudaGetErrorString(error);
            }
            
            generateRandomNumbers2D<<<blocksPerGrid, threadsPerBlock>>>(
                deviceMatrix.getPointer(), 
                deviceStates.getPointer(), 
                rows, 
                cols, 
                _min_val, 
                _max_val
            );
            cudaDeviceSynchronize();
            
            error = cudaGetLastError();
            if (error != cudaSuccess) {
                throw cudaGetErrorString(error);
            }
        } 
        else {
            size_t chunkRows = MAX_STATES / cols;
            if (chunkRows == 0) chunkRows = 1;
            
            size_t chunkStateSize = sizeof(curandState) * chunkRows * cols;
            CudaMemory<curandState> deviceStates(chunkStateSize);
            
            for (size_t startRow = 0; startRow < rows; startRow += chunkRows) {
                size_t currentChunkRows = std::min(chunkRows, rows - startRow);
                
                dim3 blocksPerGrid(
                    (cols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                    (currentChunkRows + threadsPerBlock.y - 1) / threadsPerBlock.y
                );
                
                size_t offset = startRow * cols;
                
                initializeStates2D<<<blocksPerGrid, threadsPerBlock>>>(
                    deviceStates.getPointer(), 
                    currentChunkRows, 
                    cols, 
                    _seed + startRow
                );
                cudaDeviceSynchronize();
                
                cudaError_t error = cudaGetLastError();
                if (error != cudaSuccess) {
                    throw cudaGetErrorString(error);
                }
                
                generateRandomNumbers2D<<<blocksPerGrid, threadsPerBlock>>>(
                    deviceMatrix.getPointer() + offset, 
                    deviceStates.getPointer(), 
                    currentChunkRows, 
                    cols, 
                    _min_val, 
                    _max_val
                );
                cudaDeviceSynchronize();
                
                error = cudaGetLastError();
                if (error != cudaSuccess) {
                    throw cudaGetErrorString(error);
                }
            }
        }
        
        deviceMatrix.memCpyFromDeviceToHost(result.getData(), matrixMemSize);
        
        return result;
    }
};


template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class RandomMatrixFactoryPrintResultDecorator : public RandomMatrixFactory<T>
{
private:
    std::unique_ptr<RandomMatrixFactory<T>> _base;

public:
    RandomMatrixFactoryPrintResultDecorator(std::unique_ptr<RandomMatrixFactory<T>> factory): _base(std::move(factory)) {}

    Matrix<T> generateRandom(size_t row, size_t col) override
    {
        Matrix<T> result = _base->generateRandom(row, col);
    
        std::cout << "Generated matrix: " << std::endl;
        result.print();

        return result;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class RandomCpuMatrixFactory : public RandomMatrixFactory<T>
{
private:
    std::function<T()> _randomizer;

    std::function<T()> GetRandomizer() 
    {
        std::random_device rd;
        std::default_random_engine dre(rd());

        if constexpr (std::is_floating_point_v<T>) 
        {
            std::uniform_real_distribution<T> dist_real(1, 100);
            return [dre = std::move(dre), dist_real]() mutable {
                return dist_real(dre);
            };
        }
        else if constexpr (std::is_integral_v<T>)
        {
            std::uniform_int_distribution<T> dist_int(1, 100);
            return [dre = std::move(dre), dist_int]() mutable {
                return dist_int(dre);
            };
        } 
    }
public:
    RandomCpuMatrixFactory()
    {
        this->_randomizer = this->GetRandomizer();
    }

    Matrix<T> generateRandom(size_t col, size_t row) override
    {
        Matrix<T> matrix(row, col);
        for (size_t i = 0; i < row; i++)
        {
            for (size_t j = 0; j < col; j++)
            {
                matrix[i][j] = this->_randomizer();
            }
        }

        return matrix;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class MultiplyStrategy
{
public:
    virtual Matrix<T> exec(Matrix<T> first, Matrix<T> second) = 0;
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class CpuMultiplyStrategy: public MultiplyStrategy<T>
{
public:
    Matrix<T> exec(Matrix<T> first, Matrix<T> second) override
    {
        if (first.getColCount() != second.getRowCount())
        {
            throw "Error multiply";
        }

        Matrix<T> result(first.getRowCount(), second.getColCount());

        for (size_t i = 0; i < result.getRowCount(); i++)
        {
            for (size_t j = 0; j < result.getColCount(); j++)
            {
                auto count = first.getColCount();
                T tmp {};
                for (size_t k = 0; k < count; k++)
                {
                    tmp += first[i][k] * second[k][j]; 
                }

                result[i][j] = tmp;
            }
        }

        return result;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
__global__ void multiply(
        T* first, 
        T* second, 
        T* result,
        size_t resultCol,
        size_t firstCol,
        size_t secondCol
    )
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    T tmp {};

    for (size_t k = 0; k < firstCol; k++) 
    {
        tmp += first[i * firstCol + k] * second[k * secondCol + j];
    }

    result[i * resultCol + j] = tmp; 
}

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class GpuMultiplyStrategy: public MultiplyStrategy<T>
{
public:
    Matrix<T> exec(Matrix<T> first, Matrix<T> second) override
    {
        if (first.getColCount() != second.getRowCount())
        {
            throw "Error multiply";
        }

        Matrix<T> result(first.getRowCount(), second.getColCount());
    
        CudaMemory<T> d_First(first.getMemSize());
        CudaMemory<T> d_Second(second.getMemSize());
        CudaMemory<T> d_Result(result.getMemSize());

        d_First.memCpyToDevice(first.getData(), first.getMemSize());
        d_Second.memCpyToDevice(second.getData(), second.getMemSize());

        dim3 threadsPerBlock(16, 16);
        dim3 blocksPerGrid(
            (result.getColCount() + threadsPerBlock.x - 1) / threadsPerBlock.x,
            (result.getRowCount() + threadsPerBlock.y - 1) / threadsPerBlock.y
        );
        multiply<T><<<blocksPerGrid, threadsPerBlock>>>(
            d_First.getPointer(),
            d_Second.getPointer(),
            d_Result.getPointer(),
            result.getColCount(),
            first.getColCount(),
            second.getColCount()
        );

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }

        cudaDeviceSynchronize();

        d_Result.memCpyFromDeviceToHost(result.getData(), result.getMemSize());
    
        return result;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class CalculationContext 
{
public:
    virtual Matrix<T> calculate(std::unique_ptr<MultiplyStrategy<T>> strategy, size_t firstRow, size_t firstCol, size_t secondRow, size_t secondCol) = 0;
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class CalculationContextMultiplayer : public CalculationContext<T>
{
private:
    std::unique_ptr<RandomMatrixFactory<T>> _factory;
public:
    CalculationContextMultiplayer(std::unique_ptr<RandomMatrixFactory<T>> factory): _factory(std::move(factory)) {}

    Matrix<T> calculate(std::unique_ptr<MultiplyStrategy<T>> strategy, size_t firstRow, size_t firstCol, size_t secondRow, size_t secondCol) override 
    {
        Matrix<T> first = this->_factory->generateRandom(firstRow, firstCol);
        Matrix<T> second = this->_factory->generateRandom(secondRow, secondCol);

        Matrix<T> result = strategy->exec(first, second);

        return result;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class CalculationContextPrintResultDecorator : public CalculationContext<T>
{
private:
    std::unique_ptr<CalculationContext<T>> _base;

public:
    CalculationContextPrintResultDecorator(std::unique_ptr<CalculationContext<T>> base): _base(std::move(base)) { }

    Matrix<T> calculate(std::unique_ptr<MultiplyStrategy<T>> strategy, size_t firstRow, size_t firstCol, size_t secondRow, size_t secondCol) override 
    {
        Matrix<T> result = this->_base->calculate(std::move(strategy), firstRow, firstCol, secondRow, secondCol);
        std::cout << "Result of multiply" << std::endl;
        result.print();

        return result;
    }
};

template <typename T, typename = typename std::enable_if_t<std::is_arithmetic_v<T>>>
class CalculationContextTimerExecutionDecorator : public CalculationContext<T>
{
private:
    std::unique_ptr<CalculationContext<T>> _base;
    std::string _nameOfCalculation;
public:
    CalculationContextTimerExecutionDecorator(std::unique_ptr<CalculationContext<T>> base, std::string name): 
        _nameOfCalculation(name), 
        _base(std::move(base)) { }

    Matrix<T> calculate(std::unique_ptr<MultiplyStrategy<T>> strategy, size_t firstRow, size_t firstCol, size_t secondRow, size_t secondCol) override
    {
        const auto now = std::chrono::system_clock::now();

        Matrix<T> result = this->_base->calculate(std::move(strategy), firstRow, firstCol, secondRow, secondCol);

        const auto after = std::chrono::system_clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(after - now);

        std::cout << "Execution time: " << duration.count() << " ms" << std::endl;

        return result;
    }
};

class CommandLineOptions {
private:
    std::vector<std::string_view> args;
public:
    CommandLineOptions(int argc, char* argv[])
        : args(argv, argv + argc)
    { }

    bool hasFlag(std::string_view flag) const {
        for (const auto& arg : args)
        {
            if (arg == flag)
                return true;
        }
        
        return false;
    }
};

template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T>>>
void testCase(
    size_t firstRow,
    size_t firstCol,
    size_t secondRow,
    size_t secondCol,
    bool isDebug,
    bool isDisableTimer,
    bool isGpu,
    bool isGpuRandom,
    std::string name)
{
    std::cout << name << std::endl;

    std::unique_ptr<RandomMatrixFactory<T>> factory;
    if (isGpuRandom)
    {
        factory = std::make_unique<GpuRandomMatrixFactory<T>>();
    }
    else 
    {
        factory = std::make_unique<RandomCpuMatrixFactory<T>>();
    }
    
    if(isDebug) {
        factory = std::make_unique<RandomMatrixFactoryPrintResultDecorator<T>>(std::move(factory));
    }

    std::unique_ptr<CalculationContext<T>> context = std::make_unique<CalculationContextMultiplayer<T>>(std::move(factory));
    
    if(!isDisableTimer) {
        context = std::make_unique<CalculationContextTimerExecutionDecorator<T>>(std::move(context), name);
    }

    if(isDebug) {
        context = std::make_unique<CalculationContextPrintResultDecorator<T>>(std::move(context));
    }

    std::unique_ptr<MultiplyStrategy<T>> strategy;
    if(isGpu) {
        strategy = std::make_unique<GpuMultiplyStrategy<T>>();
    } else {
        strategy = std::make_unique<CpuMultiplyStrategy<T>>();
    }

    context->calculate(
        std::move(strategy),
        firstRow, firstCol,
        secondRow, secondCol
    );

    std::cout << std::endl;
}

int main(int argc, char* argv[]) {
    CommandLineOptions options(argc, argv);

    bool hasDebug = false;
    bool hasChronoDisable = false;

    if (options.hasFlag("--debug") || options.hasFlag("-d"))
    {
        hasDebug = true;
    }
    if (options.hasFlag("--chrono-disable") || options.hasFlag("-cd"))
    {
        hasChronoDisable = true;
    }

    size_t N1 = 1000;
    size_t N2 = 1500;

    if (options.hasFlag("--small") || options.hasFlag("-s"))
    {
        N1 = 4;
        N2 = 5;
    }

    testCase<int>(
        N1, N2,
        N2, N1,
        hasDebug,
        hasChronoDisable,
        false,
        false,
        "Cpu random and cpu calculate"
    );

    testCase<int>(
        N1, N2,
        N2, N1,
        hasDebug,
        hasChronoDisable,
        true,
        false,
        "Cpu random and gpu calculate"
    );

    testCase<int>(
        N1, N2,
        N2, N1,
        hasDebug,
        hasChronoDisable,
        false,
        true,
        "Gpu random and cpu calculate"
    );

    testCase<int>(
        N1, N2,
        N2, N1,
        hasDebug,
        hasChronoDisable,
        true,
        true,
        "Gpu random and gpu calculate"
    );

    return 0;
}