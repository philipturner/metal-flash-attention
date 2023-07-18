//
//  main.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/23/23.
//

import AppleGPUInfo
import Foundation
import PythonKit

// Ensure all contexts load correctly.
_ = MetalContext.global
_ = PythonContext.global

//showMaskTest()
//showAttentionTest()

/*
 Failed test: 4x19x1x7x1xf16 (NTNN, 37x37 sparse) - 13.171
  - Sparsity: 24%
  - Mask dims: [4, 1, 19, 1]
 
 Failed test: 8x3x1x5x3xf16 (NTTT, 58x58 sparse) - 11.292
  - Sparsity: 49%
  - Mask dims: [1, 1, 3, 1]
 
 Passed test: 7x134x1x6x7xf16 (NTTN, 16x16 sparse) - 111.609
 
 Passed test: 5x58x1x2x2xf16 (NNTT, 5x5 sparse) - 14.024
 
 Passed test: 8x59x1x1x220xf16 (TTNN, 83x83 sparse) - 186.420
 
 Passed test: 3x59x1x6x24xf16 (TNTT, 3x3 sparse) - 82.680
 
 Passed test: 6x8x1x8x30xf16 (TTNN, 65x65 sparse) - 61.240
 
 Failed test: 2x1x1x8x18xf16 (TNTN, 4x4 sparse) - 10.181
  - Sparsity: 35%
  - Mask dims: [1, 1, 1, 1]
 */

/*
 Passed test: 7x134x1x6x  7xf16 (NTTN, 16x16 sparse) - 111.609
 Passed test: 5x 58x1x2x  2xf16 (NNTT, 5x5 sparse) - 14.024
 Passed test: 8x 85x1x5x  6xf16 (TTNN, 87x87 sparse) - 82.452
 Passed test: 6x 26x1x2x  1xf16 (NNTN, 5x5 sparse) - 6.152
 Passed test: 4x 12x1x8x  7xf16 (TNNN, 1x1 sparse) - 26.201
 Passed test: 6x 60x1x6x  1xf16 (TNTT, 13x13 sparse) - 18.289
 Passed test: 7x190x1x1x  1xf16 (TTTT, 6x6 sparse) - 11.788
 Passed test: 8x  1x1x2x 20xf16 (NTNN, 62x62 sparse) - 3.464
 Passed test: 3x 54x1x5x 28xf16 (NNNT, 10x10 sparse) - 67.635
 Passed test: 6x  7x1x1x 55xf16 (TNNN, 19x19 sparse) - 27.564
 Passed test: 5x145x1x8x  1xf16 (TNTN, 5x5 sparse) - 33.197
 Passed test: 2x 53x1x1x113xf16 (NTNN, 11x11 sparse) - 28.617
 Passed test: 4x 17x1x2x111xf16 (NTTN, 1x1 sparse) - 41.450
 Passed test: 8x243x1x2x  1xf16 (TTTN, 11x11 sparse) - 37.219
 Failed test: 8x  1x1x1x  1xf16 (TNTN, 4x4 sparse) - 1.538
 */

/*
 Passed test: 7x  1x1x7x 1xf16 (TTNN, 99x99 sparse) - 0.000
 Passed test: 4x  4x1x1x68xf16 (NTTT, 19x19 sparse) - 0.000
 Passed test: 2x  1x1x2x22xf16 (NTNT, 11x11 sparse) - 0.000
 Passed test: 8x 11x1x7x 1xf16 (TNTN, 14x14 sparse) - 0.000
 Passed test: 3x136x1x1x 1xf16 (NTNT, 74x74 sparse) - 0.000
 Passed test: 5x  1x1x3x 1xf16 (TTNN, 38x38 sparse) - 0.000
 Passed test: 6x  4x1x7x 1xf16 (NNNN, 2x2 sparse) - 0.000
 */

MFATestCase.runTests(speed: .veryLong)
