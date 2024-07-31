//
//  ExpectedPerformance.swift
//  FlashAttention
//
//  Created by Philip Turner on 7/31/24.
//

// MARK: Performance Before Modifications (Laplacian Test)

// M1 Max, FP32xFP32->FP32
//
//  problemSize =  511 | A   B   | 1024 threads/core | 5606 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 5586 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 4253 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 5260 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 6588 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 6143 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 4806 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 5957 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 3984 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 3837 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 4101 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 4015 GFLOPS
//  problemSize = 1535 | A   B   |  768 threads/core | 8349 GFLOPS
//  problemSize = 1535 | A   B^T |  768 threads/core | 7900 GFLOPS
//  problemSize = 1535 | A^T B   |  768 threads/core | 8466 GFLOPS
//  problemSize = 1535 | A^T B^T |  768 threads/core | 8409 GFLOPS
//  problemSize = 1536 | A   B   |  832 threads/core | 8451 GFLOPS
//  problemSize = 1536 | A   B^T |  832 threads/core | 8485 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8535 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 8525 GFLOPS
//  problemSize = 1536 | A   B   |  832 threads/core | 8446 GFLOPS
//  problemSize = 1536 | A   B^T |  832 threads/core | 8485 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8537 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 8535 GFLOPS

// M1 Max, FP16xFP16->FP16
//
//  problemSize =  511 | A   B   | 1024 threads/core | 6807 GFLOPS
//  problemSize =  511 | A   B^T | 1024 threads/core | 6150 GFLOPS
//  problemSize =  511 | A^T B   | 1024 threads/core | 5465 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 5268 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 7197 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 5468 GFLOPS
//  problemSize =  512 | A^T B   | 1024 threads/core | 4323 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 4234 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 6142 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 4261 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 4691 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 4414 GFLOPS
//  problemSize = 1535 | A   B   | 1024 threads/core | 9440 GFLOPS
//  problemSize = 1535 | A   B^T | 1024 threads/core | 9487 GFLOPS
//  problemSize = 1535 | A^T B   | 1024 threads/core | 9560 GFLOPS
//  problemSize = 1535 | A^T B^T | 1024 threads/core | 9592 GFLOPS
//  problemSize = 1536 | A   B   | 1024 threads/core | 9567 GFLOPS
//  problemSize = 1536 | A   B^T | 1024 threads/core | 9608 GFLOPS
//  problemSize = 1536 | A^T B   | 1024 threads/core | 9657 GFLOPS
//  problemSize = 1536 | A^T B^T | 1024 threads/core | 9640 GFLOPS
//  problemSize = 1536 | A   B   | 1024 threads/core | 9571 GFLOPS
//  problemSize = 1536 | A   B^T | 1024 threads/core | 9597 GFLOPS
//  problemSize = 1536 | A^T B   | 1024 threads/core | 9643 GFLOPS
//  problemSize = 1536 | A^T B^T | 1024 threads/core | 9656 GFLOPS

// M1 Max, BF16xBF16->BF16
//
//  problemSize =  511 | A   B   | 1024 threads/core | 6247 GFLOPS
//  problemSize =  511 | A   B^T |  896 threads/core | 5524 GFLOPS
//  problemSize =  511 | A^T B   |  896 threads/core | 4408 GFLOPS
//  problemSize =  511 | A^T B^T | 1024 threads/core | 6754 GFLOPS
//  problemSize =  512 | A   B   | 1024 threads/core | 5112 GFLOPS
//  problemSize =  512 | A   B^T | 1024 threads/core | 6668 GFLOPS
//  problemSize =  512 | A^T B   |  896 threads/core | 5680 GFLOPS
//  problemSize =  512 | A^T B^T | 1024 threads/core | 6871 GFLOPS
//  problemSize =  513 | A   B   | 1024 threads/core | 4872 GFLOPS
//  problemSize =  513 | A   B^T | 1024 threads/core | 4260 GFLOPS
//  problemSize =  513 | A^T B   | 1024 threads/core | 5181 GFLOPS
//  problemSize =  513 | A^T B^T | 1024 threads/core | 4927 GFLOPS
//  problemSize = 1535 | A   B   |  768 threads/core | 8601 GFLOPS
//  problemSize = 1535 | A   B^T |  832 threads/core | 8887 GFLOPS
//  problemSize = 1535 | A^T B   |  832 threads/core | 8888 GFLOPS
//  problemSize = 1535 | A^T B^T |  832 threads/core | 9162 GFLOPS
//  problemSize = 1536 | A   B   |  768 threads/core | 8671 GFLOPS
//  problemSize = 1536 | A   B^T |  832 threads/core | 8869 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8950 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 9207 GFLOPS
//  problemSize = 1536 | A   B   |  768 threads/core | 8669 GFLOPS
//  problemSize = 1536 | A   B^T |  832 threads/core | 8886 GFLOPS
//  problemSize = 1536 | A^T B   |  832 threads/core | 8951 GFLOPS
//  problemSize = 1536 | A^T B^T |  832 threads/core | 9212 GFLOPS

// M4, FP32xFP32->FP32
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

// M4, FP16xFP16->FP16
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

// M4, BF16xBF16->BF16
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
