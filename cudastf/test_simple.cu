#include <cuda/experimental/stf.cuh>
#include <iostream>

using namespace cuda::experimental::stf;

int main() {
    std::cout << "Testing basic STF functionality..." << std::endl;
    
    try {
        context ctx;
        const size_t N = 16;
        
        // Create simple logical data
        auto ldata = ctx.logical_data(shape_of<slice<char>>(N));
        
        // Simple task
        ctx.task(ldata.write())->*[](cudaStream_t s, auto data) {
            // Just write some data
            char* ptr = reinterpret_cast<char*>(data.data_handle());
            for (size_t i = 0; i < data.size(); i++) {
                ptr[i] = static_cast<char>(i);
            }
        };
        
        ctx.finalize();
        std::cout << "STF test completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "STF test failed: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
