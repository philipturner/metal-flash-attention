//
//  Attention.metal
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

#include <metal_stdlib>
using namespace metal;

// Always reads data like sm(Q^T K)V^T, does not accept already transposed data.

kernel void attention() {
  
}
