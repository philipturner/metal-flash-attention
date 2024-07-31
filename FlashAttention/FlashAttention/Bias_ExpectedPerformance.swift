//
//  Bias_ExpectedPerformance.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/31/24.
//

// MARK: - M1 Max, FP32xFP32->FP32
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   | 1024 threads/core | 4356 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 5675 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 5539 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 4320 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 6475 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 6113 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 6392 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 5690 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 3907 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 3785 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 4231 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 3913 GFLOPS
//  problemSize = 1535 | A   B   |  768 threads/core | 8326 GFLOPS
//  problemSize = 1535 | A   B^T |  768 threads/core | 7905 GFLOPS
//  problemSize = 1535 | A^T B   |  768 threads/core | 8458 GFLOPS
//  problemSize = 1535 | A^T B^T |  768 threads/core | 8390 GFLOPS
//  problemSize = 1536 | A   B   |  832 threads/core | 8424 GFLOPS
//  problemSize = 1536 | A   B^T |  768 threads/core | 8464 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8503 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 8522 GFLOPS
//  problemSize = 1537 | A   B   |  832 threads/core | 7836 GFLOPS
//  problemSize = 1537 | A   B^T |  832 threads/core | 7372 GFLOPS
//  problemSize = 1537 | A^T B   |  832 threads/core | 8097 GFLOPS
//  problemSize = 1537 | A^T B^T |  896 threads/core | 7829 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 5509 GFLOPS
//  problemSize =  511 | A   B^T bias   |  896 threads/core | 4716 GFLOPS
//  problemSize =  511 | A^T B   bias   |  896 threads/core | 4332 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 5426 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 6255 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 4581 GFLOPS
//  problemSize =  512 | A^T B   bias   | 1024 threads/core | 6210 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 5947 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 3957 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 3777 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 4113 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 3941 GFLOPS
//  problemSize = 1535 | A   B   bias   |  768 threads/core | 8329 GFLOPS
//  problemSize = 1535 | A   B^T bias   |  768 threads/core | 7901 GFLOPS
//  problemSize = 1535 | A^T B   bias   |  768 threads/core | 8400 GFLOPS
//  problemSize = 1535 | A^T B^T bias   |  768 threads/core | 8466 GFLOPS
//  problemSize = 1536 | A   B   bias   |  768 threads/core | 8434 GFLOPS
//  problemSize = 1536 | A   B^T bias   |  768 threads/core | 8479 GFLOPS
//  problemSize = 1536 | A^T B   bias   |  832 threads/core | 8490 GFLOPS
//  problemSize = 1536 | A^T B^T bias   |  832 threads/core | 8574 GFLOPS
//  problemSize = 1537 | A   B   bias   |  832 threads/core | 7868 GFLOPS
//  problemSize = 1537 | A   B^T bias   |  832 threads/core | 7384 GFLOPS
//  problemSize = 1537 | A^T B   bias   |  832 threads/core | 8097 GFLOPS
//  problemSize = 1537 | A^T B^T bias   |  896 threads/core | 7877 GFLOPS
//
//  problemSize =  511 | A   B   bias^T | 1024 threads/core | 5056 GFLOPS
//  problemSize =  511 | A   B^T bias^T |  896 threads/core | 4619 GFLOPS
//  problemSize =  511 | A^T B   bias^T | 1024 threads/core | 5413 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 5373 GFLOPS
//  problemSize =  512 | A   B   bias^T | 1024 threads/core | 6304 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 5751 GFLOPS
//  problemSize =  512 | A^T B   bias^T | 1024 threads/core | 4697 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 5646 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 4006 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 3848 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 4106 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 3948 GFLOPS
//  problemSize = 1535 | A   B   bias^T |  768 threads/core | 8369 GFLOPS
//  problemSize = 1535 | A   B^T bias^T |  768 threads/core | 7896 GFLOPS
//  problemSize = 1535 | A^T B   bias^T |  768 threads/core | 8407 GFLOPS
//  problemSize = 1535 | A^T B^T bias^T |  768 threads/core | 8419 GFLOPS
//  problemSize = 1536 | A   B   bias^T |  832 threads/core | 8415 GFLOPS
//  problemSize = 1536 | A   B^T bias^T |  832 threads/core | 8459 GFLOPS
//  problemSize = 1536 | A^T B   bias^T |  832 threads/core | 8522 GFLOPS
//  problemSize = 1536 | A^T B^T bias^T |  832 threads/core | 8536 GFLOPS
//  problemSize = 1537 | A   B   bias^T |  832 threads/core | 7839 GFLOPS
//  problemSize = 1537 | A   B^T bias^T |  832 threads/core | 7381 GFLOPS
//  problemSize = 1537 | A^T B   bias^T |  832 threads/core | 8084 GFLOPS
//  problemSize = 1537 | A^T B^T bias^T |  896 threads/core | 7824 GFLOPS

// MARK: - M1 Max, FP16xFP16->FP16
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   | 1024 threads/core | 6773 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 6620 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 5389 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 6451 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 5543 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 6815 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 7064 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 5261 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 6029 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 5539 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 6076 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 5993 GFLOPS
//  problemSize = 1535 | A   B   | 1024 threads/core | 9433 GFLOPS
//  problemSize = 1535 | A   B^T | 1024 threads/core | 9471 GFLOPS
//  problemSize = 1535 | A^T B   | 1024 threads/core | 9546 GFLOPS
//  problemSize = 1535 | A^T B^T | 1024 threads/core | 9595 GFLOPS
//  problemSize = 1536 | A   B   | 1024 threads/core | 9549 GFLOPS
//  problemSize = 1536 | A   B^T | 1024 threads/core | 9570 GFLOPS
//  problemSize = 1536 | A^T B   | 1024 threads/core | 9626 GFLOPS
//  problemSize = 1536 | A^T B^T | 1024 threads/core | 9654 GFLOPS
//  problemSize = 1537 | A   B   | 1024 threads/core | 8903 GFLOPS
//  problemSize = 1537 | A   B^T | 1024 threads/core | 8816 GFLOPS
//  problemSize = 1537 | A^T B   | 1024 threads/core | 8845 GFLOPS
//  problemSize = 1537 | A^T B^T | 1024 threads/core | 8721 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 6755 GFLOPS
//  problemSize =  511 | A   B^T bias   | 1024 threads/core | 6775 GFLOPS
//  problemSize =  511 | A^T B   bias   | 1024 threads/core | 6684 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 6610 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 5508 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 6681 GFLOPS
//  problemSize =  512 | A^T B   bias   | 1024 threads/core | 7059 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 5383 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 6030 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 4573 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 4588 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 6011 GFLOPS
//  problemSize = 1535 | A   B   bias   | 1024 threads/core | 9420 GFLOPS
//  problemSize = 1535 | A   B^T bias   | 1024 threads/core | 9406 GFLOPS
//  problemSize = 1535 | A^T B   bias   | 1024 threads/core | 9536 GFLOPS
//  problemSize = 1535 | A^T B^T bias   | 1024 threads/core | 9571 GFLOPS
//  problemSize = 1536 | A   B   bias   | 1024 threads/core | 9472 GFLOPS
//  problemSize = 1536 | A   B^T bias   | 1024 threads/core | 9416 GFLOPS
//  problemSize = 1536 | A^T B   bias   | 1024 threads/core | 9586 GFLOPS
//  problemSize = 1536 | A^T B^T bias   | 1024 threads/core | 9616 GFLOPS
//  problemSize = 1537 | A   B   bias   | 1024 threads/core | 8886 GFLOPS
//  problemSize = 1537 | A   B^T bias   | 1024 threads/core | 8829 GFLOPS
//  problemSize = 1537 | A^T B   bias   | 1024 threads/core | 8844 GFLOPS
//  problemSize = 1537 | A^T B^T bias   | 1024 threads/core | 8713 GFLOPS
//
//  problemSize =  511 | A   B   bias^T | 1024 threads/core | 5386 GFLOPS
//  problemSize =  511 | A   B^T bias^T | 1024 threads/core | 6434 GFLOPS
//  problemSize =  511 | A^T B   bias^T | 1024 threads/core | 6851 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 7086 GFLOPS
//  problemSize =  512 | A   B   bias^T | 1024 threads/core | 5388 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 3593 GFLOPS
//  problemSize =  512 | A^T B   bias^T | 1024 threads/core | 7044 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 6978 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 4570 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 5566 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 3811 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 4550 GFLOPS
//  problemSize = 1535 | A   B   bias^T | 1024 threads/core | 9403 GFLOPS
//  problemSize = 1535 | A   B^T bias^T | 1024 threads/core | 9441 GFLOPS
//  problemSize = 1535 | A^T B   bias^T | 1024 threads/core | 9539 GFLOPS
//  problemSize = 1535 | A^T B^T bias^T | 1024 threads/core | 9563 GFLOPS
//  problemSize = 1536 | A   B   bias^T | 1024 threads/core | 9453 GFLOPS
//  problemSize = 1536 | A   B^T bias^T | 1024 threads/core | 9402 GFLOPS
//  problemSize = 1536 | A^T B   bias^T | 1024 threads/core | 9590 GFLOPS
//  problemSize = 1536 | A^T B^T bias^T | 1024 threads/core | 9633 GFLOPS
//  problemSize = 1537 | A   B   bias^T | 1024 threads/core | 8909 GFLOPS
//  problemSize = 1537 | A   B^T bias^T | 1024 threads/core | 8874 GFLOPS
//  problemSize = 1537 | A^T B   bias^T | 1024 threads/core | 8866 GFLOPS
//  problemSize = 1537 | A^T B^T bias^T | 1024 threads/core | 8726 GFLOPS

// MARK: - M1 Max, BF16xBF16->BF16
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   |  896 threads/core | 5432 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 6321 GFLOPS
//  problemSize =  511 | A^T B   |  896 threads/core | 5645 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 7008 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 5145 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 5156 GFLOPS
//  problemSize =  512 | A^T B   |  896 threads/core | 5912 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 5230 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 4901 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 4198 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 5049 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 4580 GFLOPS
//  problemSize = 1535 | A   B   |  768 threads/core | 8595 GFLOPS
//  problemSize = 1535 | A   B^T |  832 threads/core | 8883 GFLOPS
//  problemSize = 1535 | A^T B   |  832 threads/core | 8876 GFLOPS
//  problemSize = 1535 | A^T B^T |  832 threads/core | 9146 GFLOPS
//  problemSize = 1536 | A   B   |  768 threads/core | 8655 GFLOPS
//  problemSize = 1536 | A   B^T |  832 threads/core | 8875 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8949 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 9175 GFLOPS
//  problemSize = 1537 | A   B   |  768 threads/core | 8130 GFLOPS
//  problemSize = 1537 | A   B^T |  832 threads/core | 8379 GFLOPS
//  problemSize = 1537 | A^T B   |  832 threads/core | 8382 GFLOPS
//  problemSize = 1537 | A^T B^T |  832 threads/core | 8663 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 6408 GFLOPS
//  problemSize =  511 | A   B^T bias   |  896 threads/core | 4070 GFLOPS
//  problemSize =  511 | A^T B   bias   |  896 threads/core | 4238 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 6647 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 6684 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 6632 GFLOPS
//  problemSize =  512 | A^T B   bias   |  896 threads/core | 5630 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 6719 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 4603 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 4456 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 4963 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 4626 GFLOPS
//  problemSize = 1535 | A   B   bias   |  832 threads/core | 8577 GFLOPS
//  problemSize = 1535 | A   B^T bias   |  832 threads/core | 8834 GFLOPS
//  problemSize = 1535 | A^T B   bias   |  832 threads/core | 8838 GFLOPS
//  problemSize = 1535 | A^T B^T bias   |  832 threads/core | 9122 GFLOPS
//  problemSize = 1536 | A   B   bias   |  768 threads/core | 8652 GFLOPS
//  problemSize = 1536 | A   B^T bias   |  832 threads/core | 8861 GFLOPS
//  problemSize = 1536 | A^T B   bias   |  832 threads/core | 8905 GFLOPS
//  problemSize = 1536 | A^T B^T bias   |  832 threads/core | 9187 GFLOPS
//  problemSize = 1537 | A   B   bias   |  832 threads/core | 8154 GFLOPS
//  problemSize = 1537 | A   B^T bias   |  832 threads/core | 8292 GFLOPS
//  problemSize = 1537 | A^T B   bias   |  832 threads/core | 8349 GFLOPS
//  problemSize = 1537 | A^T B^T bias   |  832 threads/core | 8662 GFLOPS
//
//  problemSize =  511 | A   B   bias^T |  896 threads/core | 5380 GFLOPS
//  problemSize =  511 | A   B^T bias^T |  896 threads/core | 5354 GFLOPS
//  problemSize =  511 | A^T B   bias^T |  896 threads/core | 5427 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 6619 GFLOPS
//  problemSize =  512 | A   B   bias^T |  896 threads/core | 4313 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 5109 GFLOPS
//  problemSize =  512 | A^T B   bias^T |  896 threads/core | 5793 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 5174 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 4759 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 4170 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 4967 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 4387 GFLOPS
//  problemSize = 1535 | A   B   bias^T |  832 threads/core | 8606 GFLOPS
//  problemSize = 1535 | A   B^T bias^T |  832 threads/core | 8854 GFLOPS
//  problemSize = 1535 | A^T B   bias^T |  832 threads/core | 8890 GFLOPS
//  problemSize = 1535 | A^T B^T bias^T |  832 threads/core | 9145 GFLOPS
//  problemSize = 1536 | A   B   bias^T |  768 threads/core | 8653 GFLOPS
//  problemSize = 1536 | A   B^T bias^T |  832 threads/core | 8879 GFLOPS
//  problemSize = 1536 | A^T B   bias^T |  832 threads/core | 8932 GFLOPS
//  problemSize = 1536 | A^T B^T bias^T |  832 threads/core | 9180 GFLOPS
//  problemSize = 1537 | A   B   bias^T |  832 threads/core | 7932 GFLOPS
//  problemSize = 1537 | A   B^T bias^T |  832 threads/core | 8298 GFLOPS
//  problemSize = 1537 | A^T B   bias^T |  832 threads/core | 8345 GFLOPS
//  problemSize = 1537 | A^T B^T bias^T |  832 threads/core | 8452 GFLOPS

// MARK: - M4, FP32xFP32->FP32
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   | 1024 threads/core | 2731 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 2725 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 2672 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 2659 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 2807 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 2821 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 2754 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 2719 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 2385 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 2385 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 2344 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 2326 GFLOPS
//  problemSize = 1023 | A   B   | 1024 threads/core | 3051 GFLOPS
//  problemSize = 1023 | A   B^T | 1024 threads/core | 2819 GFLOPS
//  problemSize = 1023 | A^T B   | 1024 threads/core | 3016 GFLOPS
//  problemSize = 1023 | A^T B^T | 1024 threads/core | 2970 GFLOPS
//  problemSize = 1024 | A   B   | 1024 threads/core | 3080 GFLOPS
//  problemSize = 1024 | A   B^T | 1024 threads/core | 3050 GFLOPS
//  problemSize = 1024 | A^T B   | 1024 threads/core | 3021 GFLOPS
//  problemSize = 1024 | A^T B^T | 1024 threads/core | 2990 GFLOPS
//  problemSize = 1025 | A   B   | 1024 threads/core | 2876 GFLOPS
//  problemSize = 1025 | A   B^T | 1024 threads/core | 2633 GFLOPS
//  problemSize = 1025 | A^T B   | 1024 threads/core | 2848 GFLOPS
//  problemSize = 1025 | A^T B^T | 1024 threads/core | 2799 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 2710 GFLOPS
//  problemSize =  511 | A   B^T bias   | 1024 threads/core | 2716 GFLOPS
//  problemSize =  511 | A^T B   bias   | 1024 threads/core | 2695 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 2636 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 2746 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 2778 GFLOPS
//  problemSize =  512 | A^T B   bias   | 1024 threads/core | 2706 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 2667 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 2379 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 2377 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 2337 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 2327 GFLOPS
//  problemSize = 1023 | A   B   bias   | 1024 threads/core | 3062 GFLOPS
//  problemSize = 1023 | A   B^T bias   | 1024 threads/core | 2806 GFLOPS
//  problemSize = 1023 | A^T B   bias   | 1024 threads/core | 3015 GFLOPS
//  problemSize = 1023 | A^T B^T bias   | 1024 threads/core | 2963 GFLOPS
//  problemSize = 1024 | A   B   bias   | 1024 threads/core | 3068 GFLOPS
//  problemSize = 1024 | A   B^T bias   | 1024 threads/core | 3048 GFLOPS
//  problemSize = 1024 | A^T B   bias   | 1024 threads/core | 3074 GFLOPS
//  problemSize = 1024 | A^T B^T bias   | 1024 threads/core | 2947 GFLOPS
//  problemSize = 1025 | A   B   bias   | 1024 threads/core | 2880 GFLOPS
//  problemSize = 1025 | A   B^T bias   | 1024 threads/core | 2656 GFLOPS
//  problemSize = 1025 | A^T B   bias   | 1024 threads/core | 2856 GFLOPS
//  problemSize = 1025 | A^T B^T bias   | 1024 threads/core | 2828 GFLOPS
//
//  problemSize =  511 | A   B   bias^T | 1024 threads/core | 2731 GFLOPS
//  problemSize =  511 | A   B^T bias^T | 1024 threads/core | 2721 GFLOPS
//  problemSize =  511 | A^T B   bias^T | 1024 threads/core | 2711 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 2680 GFLOPS
//  problemSize =  512 | A   B   bias^T | 1024 threads/core | 2751 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 2792 GFLOPS
//  problemSize =  512 | A^T B   bias^T | 1024 threads/core | 2725 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 2696 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 2387 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 2379 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 2351 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 2310 GFLOPS
//  problemSize = 1023 | A   B   bias^T | 1024 threads/core | 3082 GFLOPS
//  problemSize = 1023 | A   B^T bias^T | 1024 threads/core | 2807 GFLOPS
//  problemSize = 1023 | A^T B   bias^T | 1024 threads/core | 3034 GFLOPS
//  problemSize = 1023 | A^T B^T bias^T | 1024 threads/core | 2973 GFLOPS
//  problemSize = 1024 | A   B   bias^T | 1024 threads/core | 3093 GFLOPS
//  problemSize = 1024 | A   B^T bias^T | 1024 threads/core | 3060 GFLOPS
//  problemSize = 1024 | A^T B   bias^T | 1024 threads/core | 3035 GFLOPS
//  problemSize = 1024 | A^T B^T bias^T | 1024 threads/core | 2951 GFLOPS
//  problemSize = 1025 | A   B   bias^T | 1024 threads/core | 2900 GFLOPS
//  problemSize = 1025 | A   B^T bias^T | 1024 threads/core | 2662 GFLOPS
//  problemSize = 1025 | A^T B   bias^T | 1024 threads/core | 2860 GFLOPS
//  problemSize = 1025 | A^T B^T bias^T | 1024 threads/core | 2802 GFLOPS

// MARK: - M4, FP16xFP16->FP16
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   | 1024 threads/core | 2985 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 3029 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 2903 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 2934 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 3095 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 3058 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 3003 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 2986 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 2666 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 2600 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 2525 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 2563 GFLOPS
//  problemSize = 1023 | A   B   | 1024 threads/core | 3403 GFLOPS
//  problemSize = 1023 | A   B^T | 1024 threads/core | 3341 GFLOPS
//  problemSize = 1023 | A^T B   | 1024 threads/core | 3376 GFLOPS
//  problemSize = 1023 | A^T B^T | 1024 threads/core | 3332 GFLOPS
//  problemSize = 1024 | A   B   | 1024 threads/core | 3409 GFLOPS
//  problemSize = 1024 | A   B^T | 1024 threads/core | 3395 GFLOPS
//  problemSize = 1024 | A^T B   | 1024 threads/core | 3379 GFLOPS
//  problemSize = 1024 | A^T B^T | 1024 threads/core | 3335 GFLOPS
//  problemSize = 1025 | A   B   | 1024 threads/core | 3220 GFLOPS
//  problemSize = 1025 | A   B^T | 1024 threads/core | 3157 GFLOPS
//  problemSize = 1025 | A^T B   | 1024 threads/core | 3193 GFLOPS
//  problemSize = 1025 | A^T B^T | 1024 threads/core | 3150 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 2985 GFLOPS
//  problemSize =  511 | A   B^T bias   | 1024 threads/core | 2974 GFLOPS
//  problemSize =  511 | A^T B   bias   | 1024 threads/core | 2930 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 2903 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 3060 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 2974 GFLOPS
//  problemSize =  512 | A^T B   bias   | 1024 threads/core | 2972 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 2873 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 2593 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 2596 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 2555 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 2507 GFLOPS
//  problemSize = 1023 | A   B   bias   | 1024 threads/core | 3400 GFLOPS
//  problemSize = 1023 | A   B^T bias   | 1024 threads/core | 3334 GFLOPS
//  problemSize = 1023 | A^T B   bias   | 1024 threads/core | 3351 GFLOPS
//  problemSize = 1023 | A^T B^T bias   | 1024 threads/core | 3315 GFLOPS
//  problemSize = 1024 | A   B   bias   | 1024 threads/core | 3429 GFLOPS
//  problemSize = 1024 | A   B^T bias   | 1024 threads/core | 3397 GFLOPS
//  problemSize = 1024 | A^T B   bias   | 1024 threads/core | 3402 GFLOPS
//  problemSize = 1024 | A^T B^T bias   | 1024 threads/core | 3355 GFLOPS
//  problemSize = 1025 | A   B   bias   | 1024 threads/core | 3214 GFLOPS
//  problemSize = 1025 | A   B^T bias   | 1024 threads/core | 3156 GFLOPS
//  problemSize = 1025 | A^T B   bias   | 1024 threads/core | 3175 GFLOPS
//  problemSize = 1025 | A^T B^T bias   | 1024 threads/core | 3128 GFLOPS
//
//  problemSize =  511 | A   B   bias^T | 1024 threads/core | 2957 GFLOPS
//  problemSize =  511 | A   B^T bias^T | 1024 threads/core | 2983 GFLOPS
//  problemSize =  511 | A^T B   bias^T | 1024 threads/core | 2896 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 2914 GFLOPS
//  problemSize =  512 | A   B   bias^T | 1024 threads/core | 2996 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 2988 GFLOPS
//  problemSize =  512 | A^T B   bias^T | 1024 threads/core | 2924 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 2962 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 2615 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 2606 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 2548 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 2552 GFLOPS
//  problemSize = 1023 | A   B   bias^T | 1024 threads/core | 3398 GFLOPS
//  problemSize = 1023 | A   B^T bias^T | 1024 threads/core | 3344 GFLOPS
//  problemSize = 1023 | A^T B   bias^T | 1024 threads/core | 3362 GFLOPS
//  problemSize = 1023 | A^T B^T bias^T | 1024 threads/core | 3324 GFLOPS
//  problemSize = 1024 | A   B   bias^T | 1024 threads/core | 3426 GFLOPS
//  problemSize = 1024 | A   B^T bias^T | 1024 threads/core | 3394 GFLOPS
//  problemSize = 1024 | A^T B   bias^T | 1024 threads/core | 3404 GFLOPS
//  problemSize = 1024 | A^T B^T bias^T | 1024 threads/core | 3345 GFLOPS
//  problemSize = 1025 | A   B   bias^T | 1024 threads/core | 3217 GFLOPS
//  problemSize = 1025 | A   B^T bias^T | 1024 threads/core | 3157 GFLOPS
//  problemSize = 1025 | A^T B   bias^T | 1024 threads/core | 3177 GFLOPS
//  problemSize = 1025 | A^T B^T bias^T | 1024 threads/core | 3142 GFLOPS

// MARK: - M4, BF16xBF16->BF16
//
// Before Any Modifications
//
//  problemSize =  511 | A   B   | 1024 threads/core | 2918 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 2967 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 2871 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 2891 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 3052 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 3037 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 2935 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 2909 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 2581 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 2557 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 2516 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 2535 GFLOPS
//  problemSize = 1023 | A   B   | 1024 threads/core | 3342 GFLOPS
//  problemSize = 1023 | A   B^T | 1024 threads/core | 3291 GFLOPS
//  problemSize = 1023 | A^T B   | 1024 threads/core | 3321 GFLOPS
//  problemSize = 1023 | A^T B^T | 1024 threads/core | 3271 GFLOPS
//  problemSize = 1024 | A   B   | 1024 threads/core | 3319 GFLOPS
//  problemSize = 1024 | A   B^T | 1024 threads/core | 3332 GFLOPS
//  problemSize = 1024 | A^T B   | 1024 threads/core | 3297 GFLOPS
//  problemSize = 1024 | A^T B^T | 1024 threads/core | 3231 GFLOPS
//  problemSize = 1025 | A   B   | 1024 threads/core | 3166 GFLOPS
//  problemSize = 1025 | A   B^T | 1024 threads/core | 3104 GFLOPS
//  problemSize = 1025 | A^T B   | 1024 threads/core | 3146 GFLOPS
//  problemSize = 1025 | A^T B^T | 1024 threads/core | 3091 GFLOPS
//
// After Adding Bias via Async Copy
//
//  problemSize =  511 | A   B   bias   | 1024 threads/core | 2907 GFLOPS
//  problemSize =  511 | A   B^T bias   | 1024 threads/core | 2891 GFLOPS
//  problemSize =  511 | A^T B   bias   | 1024 threads/core | 2857 GFLOPS
//  problemSize =  511 | A^T B^T bias   | 1024 threads/core | 2846 GFLOPS
//  problemSize =  512 | A   B   bias   | 1024 threads/core | 2926 GFLOPS
//  problemSize =  512 | A   B^T bias   | 1024 threads/core | 3051 GFLOPS
//  problemSize =  512 | A^T B   bias   | 1024 threads/core | 2875 GFLOPS
//  problemSize =  512 | A^T B^T bias   | 1024 threads/core | 2873 GFLOPS
//  problemSize =  513 | A   B   bias   | 1024 threads/core | 2523 GFLOPS
//  problemSize =  513 | A   B^T bias   | 1024 threads/core | 2549 GFLOPS
//  problemSize =  513 | A^T B   bias   | 1024 threads/core | 2487 GFLOPS
//  problemSize =  513 | A^T B^T bias   | 1024 threads/core | 2501 GFLOPS
//  problemSize = 1023 | A   B   bias   | 1024 threads/core | 3343 GFLOPS
//  problemSize = 1023 | A   B^T bias   | 1024 threads/core | 3293 GFLOPS
//  problemSize = 1023 | A^T B   bias   | 1024 threads/core | 3289 GFLOPS
//  problemSize = 1023 | A^T B^T bias   | 1024 threads/core | 3250 GFLOPS
//  problemSize = 1024 | A   B   bias   | 1024 threads/core | 3339 GFLOPS
//  problemSize = 1024 | A   B^T bias   | 1024 threads/core | 3309 GFLOPS
//  problemSize = 1024 | A^T B   bias   | 1024 threads/core | 3308 GFLOPS
//  problemSize = 1024 | A^T B^T bias   | 1024 threads/core | 3247 GFLOPS
//  problemSize = 1025 | A   B   bias   | 1024 threads/core | 3160 GFLOPS
//  problemSize = 1025 | A   B^T bias   | 1024 threads/core | 3093 GFLOPS
//  problemSize = 1025 | A^T B   bias   | 1024 threads/core | 3109 GFLOPS
//  problemSize = 1025 | A^T B^T bias   | 1024 threads/core | 3074 GFLOPS
//
//  problemSize =  511 | A   B   bias^T | 1024 threads/core | 2914 GFLOPS
//  problemSize =  511 | A   B^T bias^T | 1024 threads/core | 2887 GFLOPS
//  problemSize =  511 | A^T B   bias^T | 1024 threads/core | 2873 GFLOPS
//  problemSize =  511 | A^T B^T bias^T | 1024 threads/core | 2840 GFLOPS
//  problemSize =  512 | A   B   bias^T | 1024 threads/core | 2946 GFLOPS
//  problemSize =  512 | A   B^T bias^T | 1024 threads/core | 2984 GFLOPS
//  problemSize =  512 | A^T B   bias^T | 1024 threads/core | 2894 GFLOPS
//  problemSize =  512 | A^T B^T bias^T | 1024 threads/core | 2850 GFLOPS
//  problemSize =  513 | A   B   bias^T | 1024 threads/core | 2555 GFLOPS
//  problemSize =  513 | A   B^T bias^T | 1024 threads/core | 2583 GFLOPS
//  problemSize =  513 | A^T B   bias^T | 1024 threads/core | 2478 GFLOPS
//  problemSize =  513 | A^T B^T bias^T | 1024 threads/core | 2487 GFLOPS
//  problemSize = 1023 | A   B   bias^T | 1024 threads/core | 3333 GFLOPS
//  problemSize = 1023 | A   B^T bias^T | 1024 threads/core | 3277 GFLOPS
//  problemSize = 1023 | A^T B   bias^T | 1024 threads/core | 3322 GFLOPS
//  problemSize = 1023 | A^T B^T bias^T | 1024 threads/core | 3242 GFLOPS
//  problemSize = 1024 | A   B   bias^T | 1024 threads/core | 3356 GFLOPS
//  problemSize = 1024 | A   B^T bias^T | 1024 threads/core | 3314 GFLOPS
//  problemSize = 1024 | A^T B   bias^T | 1024 threads/core | 3298 GFLOPS
//  problemSize = 1024 | A^T B^T bias^T | 1024 threads/core | 3222 GFLOPS
//  problemSize = 1025 | A   B   bias^T | 1024 threads/core | 3153 GFLOPS
//  problemSize = 1025 | A   B^T bias^T | 1024 threads/core | 3097 GFLOPS
//  problemSize = 1025 | A^T B   bias^T | 1024 threads/core | 3143 GFLOPS
//  problemSize = 1025 | A^T B^T bias^T | 1024 threads/core | 3068 GFLOPS
