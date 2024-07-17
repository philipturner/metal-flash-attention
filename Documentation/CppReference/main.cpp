//
//  main.cpp
//  IOKitProject
//
//  Created by Philip Turner on 7/9/24.
//

#include "ccv_nnc_mfa_error.hpp"
#include "GEMM/CoreCount.hpp"
#include "GEMM/GEMMDescriptor.hpp"
#include "GEMM/GEMMKernel.hpp"
#include "GEMM/GEMMShaderCache.hpp"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <vector>

int main(int argc, const char * argv[]) {
  // insert code here...
  std::cout << "Hello, World!\n";
  
  // M1 Max
  //
  // 511^3, BF16, NN | 5149 GFLOPS
  // 511^3, BF16, NT | 4316 GFLOPS or 5559 GFLOPS
  // 511^3, BF16, TN | 4415 GFLOPS
  // 511^3, BF16, TT | 4282 GFLOPS or 5310 GFLOPS
  //
  // 512^3, BF16, NN | 5201 GFLOPS
  // 512^3, BF16, NT | 5265 GFLOPS
  // 512^3, BF16, TN | 4556 GFLOPS or 5880 GFLOPS
  // 512^3, BF16, TT | 5492 GFLOPS
  //
  // 1488^3, BF16, NN | 8371 GFLOPS
  // 1488^3, BF16, NT | 8683 GFLOPS
  // 1488^3, BF16, TN | 8807 GFLOPS
  // 1488^3, BF16, TT | 9041 GFLOPS
  //
  // 1489^3, BF16, NN | 8039 GFLOPS
  // 1489^3, BF16, NT | 8395 GFLOPS
  // 1489^3, BF16, TN | 8378 GFLOPS
  // 1489^3, BF16, TT | 8642 GFLOPS
  
  // Specify the problem configuration.
  int64_t problemSize = 10;
  
  // Instantiate the descriptor.
  GEMMDescriptor gemmDesc;
  gemmDesc.matrixDimensions = simd::uint3 { 
    uint32_t(problemSize),
    uint32_t(problemSize),
    uint32_t(problemSize),
  };
  gemmDesc.memoryPrecisions = {
    .A = GEMMOperandPrecision::BF16,
    .B = GEMMOperandPrecision::BF16,
    .C = GEMMOperandPrecision::BF16,
  };
  gemmDesc.transposeState = simd::uchar2 { false, false };
  
  // Instantiate the kernel.
  auto pool = NS::AutoreleasePool::alloc()->init();
  GEMMShaderCache::fetchKernel(gemmDesc);
  auto pipelineValue = GEMMShaderCache::fetchKernel(gemmDesc);
  pool->drain();
  auto kernel = pipelineValue->kernel;
  auto pipeline = pipelineValue->pipeline;
  
  // Instantiate the device.
  auto device = NS::TransferPtr(MTL::CreateSystemDefaultDevice());
  
  // Set up the diagonal matrix multiplication.
  std::vector<float> A;
  std::vector<float> B;
  std::vector<float> C;
  {
    // A 5x5 matrix defining the upper submatrix of B.
    std::vector<float> B_contents = {
      1.0, 2.0, 3.0, 4.0, 5.0,
      1.0, 2.0, 3.0, 4.0, 5.0,
      2.0, 3.0, 4.0, 5.0, 6.0,
      2.0, 4.0, 6.0, 8.0, 10.0,
      5.0, 4.0, 3.0, 2.0, 1.0,
    };
    for (int64_t rowID = 0; rowID < problemSize; ++rowID) {
      for (int64_t columnID = 0; columnID < problemSize; ++columnID) {
        if (rowID == columnID) {
          A.push_back(2);
        } else {
          A.push_back(0);
        }
        
        if (rowID < 5 && columnID < 5) {
          int64_t address = rowID * 5 + columnID;
          float value = B_contents[address];
          B.push_back(value);
        } else if (rowID == columnID) {
          B.push_back(1);
        } else {
          B.push_back(0);
        }
        
        C.push_back(0);
      }
    }
  }
  
  // Utility functions for type casting.
  auto memcpyDeviceTransfer =
  [=]
  (void *gpu, void *cpu, int64_t elements, bool isCPUToGPU,
   GEMMOperandPrecision type) {
    for (int64_t i = 0; i < elements; ++i) {
      if (type == GEMMOperandPrecision::FP32) {
        // FP32
        auto* gpuPointer = (float*)gpu + i;
        auto* cpuPointer = (float*)cpu + i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[0];
        } else {
          cpuPointer[0] = gpuPointer[0];
        }
      } else if (type.value == GEMMOperandPrecision::FP16) {
        // FP16
        auto* gpuPointer = (_Float16*)gpu + i;
        auto* cpuPointer = (float*)cpu + i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[0];
        } else {
          cpuPointer[0] = gpuPointer[0];
        }
      } else if (type.value == GEMMOperandPrecision::BF16) {
        // BF16
        auto* gpuPointer = (uint16_t*)gpu + i;
        auto* cpuPointer = (uint16_t*)cpu + 2 * i;
        
        if (isCPUToGPU) {
          gpuPointer[0] = cpuPointer[1];
        } else {
          cpuPointer[0] = 0;
          cpuPointer[1] = gpuPointer[0];
        }
      }
    }
  };
  
  // Allocate and fill the buffers.
  int64_t squareMatrixBytes = problemSize * problemSize * sizeof(float);
  auto bufferA = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  auto bufferB = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  auto bufferC = NS::TransferPtr(device->newBuffer
  (squareMatrixBytes, MTL::ResourceStorageModeShared));
  {
    int64_t elements = problemSize * problemSize;
    memcpyDeviceTransfer
    (bufferA->contents(), A.data(), elements, true,
     gemmDesc.memoryPrecisions.value().A);
    memcpyDeviceTransfer
    (bufferB->contents(), B.data(), elements, true,
     gemmDesc.memoryPrecisions.value().B);
  }
  
  // Instantiate the command queue.
  auto commandQueue = NS::TransferPtr(device->newCommandQueue());
  
  // Multiply A with B.
  int64_t maxGFLOPS = 0;
  int64_t occupancy = pipeline->maxTotalThreadsPerThreadgroup();
  for (int64_t trialID = 0; trialID < 15; ++trialID) {
    int64_t duplicatedCommandCount = 20;
    
    auto commandBuffer = commandQueue->commandBuffer();
    auto encoder = commandBuffer->computeCommandEncoder();
    encoder->setComputePipelineState(pipeline.get());
    encoder->setThreadgroupMemoryLength(kernel->threadgroupMemoryAllocation, 0);
    encoder->setBuffer(bufferA.get(), 0, 0);
    encoder->setBuffer(bufferB.get(), 0, 1);
    encoder->setBuffer(bufferC.get(), 0, 2);
    
    for (int64_t commandID = 0; commandID < duplicatedCommandCount; ++commandID) {
      auto ceilDivide =
      [=](int64_t target, uint16_t granularity) -> int64_t {
        return (target + int64_t(granularity) - 1) / int64_t(granularity);
      };
      MTL::Size gridSize
      (ceilDivide(problemSize, kernel->blockDimensions[1]),
       ceilDivide(problemSize, kernel->blockDimensions[0]),
       1);
      MTL::Size groupSize
      (int64_t(kernel->threadgroupSize), 1, 1);
      encoder->dispatchThreadgroups(gridSize, groupSize);
    }
    encoder->endEncoding();
    commandBuffer->commit();
    commandBuffer->waitUntilCompleted();
    
    // Determine the time taken.
    double start = commandBuffer->GPUStartTime();
    double end = commandBuffer->GPUEndTime();
    double latency = end - start;
    
    // Determine the amount of work done.
    int64_t operations = 2 * problemSize * problemSize * problemSize;
    operations *= duplicatedCommandCount;
    int64_t gflops = int64_t(double(operations) / double(latency) / 1e9);
    
    // Report the results.
    maxGFLOPS = std::max(maxGFLOPS, gflops);
  }
  
  // Copy the results to C.
  {
    int64_t elements = problemSize * problemSize;
    memcpyDeviceTransfer
    (bufferC->contents(), C.data(), elements, false,
     gemmDesc.memoryPrecisions.value().C);
  }
  
  if (true) {
    // Display the matrices.
    auto displayMatrix =
    [=](float* matrix) {
      int64_t loopCount = std::min(int64_t(problemSize), int64_t(10));
      for (int64_t rowID = 0; rowID < loopCount; ++rowID) {
        for (int64_t columnID = 0; columnID < loopCount; ++columnID) {
          auto address = rowID * problemSize + columnID;
          float entry = matrix[address];
          
          std::cout << std::setprecision(4);
          std::cout << entry << " ";
        }
        std::cout << "\n";
      }
    };
    
    std::cout << "\n";
    std::cout << "A:\n";
    displayMatrix(A.data());
    
    std::cout << "\n";
    std::cout << "B:\n";
    displayMatrix(B.data());
    
    std::cout << "\n";
    std::cout << "C:\n";
    displayMatrix(C.data());
  }
  
  // Choose an error threshold.
  float errorThreshold = 1e-5;
  if (gemmDesc.memoryPrecisions.value().A == GEMMOperandPrecision::BF16) {
    errorThreshold = 2e-1;
  }
  if (gemmDesc.memoryPrecisions.value().B == GEMMOperandPrecision::BF16) {
    errorThreshold = 2e-1;
  }
  
  // Check the results.
  {
    int64_t errorCount = 0;
    for (int64_t rowID = 0; rowID < problemSize; ++rowID) {
      for (int64_t columnID = 0; columnID < problemSize; ++columnID) {
        float entryB = B[rowID * problemSize + columnID];
        float entryC;
        if (gemmDesc.transposeState.value()[1]) {
          entryC = C[columnID * problemSize + rowID];
        } else {
          entryC = C[rowID * problemSize + columnID];
        }
        
        float actual = entryC;
        float expected = entryB * 2;
        float error = actual - expected;
        if (error < 0) {
          error = -error;
        }
        
        if (error < errorThreshold) {
          // Skip ahead to the next iteration. There is no error message to
          // throw.
          continue;
        }
        if (errorCount > 10) {
          // Don't send too many messages to the console.
          continue;
        }
        errorCount += 1;
        
        std::cout << "C[" << rowID << "][" << columnID << "] | ";
        std::cout << "error: " << error << " | ";
        std::cout << "actual: " << actual << " | ";
        std::cout << "expected: " << expected << " | ";
        std::cout << std::endl;
      }
    }
  }
  
  // Report the performance.
  std::cout << std::endl;
  GEMMShaderCache::fetchKernel(gemmDesc);
  std::cout << maxGFLOPS << " GFLOPS ";
  std::cout << std::endl;
  std::cout << occupancy << " threads/core ";
  std::cout << std::endl;
  
  return 0;
}
