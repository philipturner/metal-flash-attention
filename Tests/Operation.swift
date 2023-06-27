//
//  Operation.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Metal

protocol Operation {
  // function for running several trials on a command queue
  
  // function for comparing output to another implementation
  // - might need to be declared per subclass
}

protocol Attention: Operation {
  
}

protocol Convolution: Operation {
  
}

protocol GEMM: Operation {
  
}

protocol Normalization: Operation {
  
}
