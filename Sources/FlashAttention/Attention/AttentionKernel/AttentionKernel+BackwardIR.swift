//
//  AttentionKernel+BackwardIR.swift
//  FlashAttention
//
//  Backward attention kernel IR generation.
//
//  backwardQuery:    D = dO·O; for c: S=Q*K^T, P=softmax(S,L), dP=dO*V^T, dS=P*(dP-D), dQ+=dS*K
//  backwardKeyValue: for r: S^T=K*Q^T, P^T=softmax(S^T,L), dV+=P^T*dO, dP^T=V*dO^T, dS^T=P^T*(dP^T-D), dK+=dS^T*Q
//

extension AttentionKernel {

  // MARK: - Backward Query

  func generateBackwardQueryKernel(
    desc: MonolithicDescriptor,
    R: UInt32, C: UInt32, D: UInt32,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    parallelDim: UInt32, traversalDim: UInt32,
    paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    sSramCount: Int,
    scaleFactor: Float, tgFloats: Int,
    leadingDim: (AttentionOperand) -> UInt32,
    leadingBlockDim: (AttentionOperand) -> UInt32,
    memPrec: (AttentionOperand) -> GEMMOperandPrecision,
    regPrec: (AttentionOperand) -> GEMMOperandPrecision,
    memSize: (AttentionOperand) -> UInt32
  ) -> String {
    var ir = ""

    let regS = regPrec(.S)
    let regP = regPrec(.P)
    let regdP = regPrec(.dP)
    let regdS = regPrec(.dS)
    let regQ = regPrec(.Q)
    let regK = regPrec(.K)
    let regV = regPrec(.V)
    let regdO = regPrec(.dO)
    let regdQ = regPrec(.dQ)
    let regO = regPrec(.O)
    let memQ = memPrec(.Q)
    let memK = memPrec(.K)
    let memV = memPrec(.V)
    let memO = memPrec(.O)
    let memdO = memPrec(.dO)
    let memdQ = memPrec(.dQ)
    let memL = memPrec(.L)
    let memD = memPrec(.D)

    let isCachedQ = cached(.Q)
    let isCacheddO = cached(.dO)
    let isCacheddQ = cached(.dQ)

    // dQ accumulator count = paddedD / 8 (always full head dim)
    let dqCount = Int(paddedD / 8)

    // Derivative scale factor: 1/sqrt(D) (no log2(e) factor)
    let derivScale = 1.0 / Float(headDimension).squareRoot()

    // MARK: - Setup

    ir += "  ; === Backward Query setup ===\n"

    // Cache load Q if cached
    if isCachedQ {
      ir += generateCacheLoad(
        operand: .Q, prefix: "cq_",
        parallelDim: parallelDim, D: D, paddedD: paddedD,
        blockP: blockP, blockH: blockH,
        leadingDim: leadingDim(.Q), leadingBlockDim: leadingBlockDim(.Q),
        memPrec: memQ, regPrec: regQ,
        transposed: transposed(.Q)
      )
    }

    // Cache load dO if cached
    if isCacheddO {
      ir += generateCacheLoad(
        operand: .dO, prefix: "cdo_",
        parallelDim: parallelDim, D: D, paddedD: paddedD,
        blockP: blockP, blockH: blockH,
        leadingDim: leadingDim(.dO), leadingBlockDim: leadingBlockDim(.dO),
        memPrec: memdO, regPrec: regdO,
        transposed: transposed(.dO)
      )
    }

    // Initialize dQ accumulators to zero
    for i in 0..<dqCount {
      ir += "  \(irZeroVec64(result: "%dq_init_\(i)", precision: regdQ))\n"
    }

    // Load L scalar for this row
    // L_sram = L[clamped_par_off]
    let elemSizeL = UInt32(memL.size)
    ir += "  ; Load L scalar\n"
    ir += "  %L_off = zext i32 %clamped_par_off to i64\n"
    ir += "  %L_byte = mul i64 %L_off, \(elemSizeL)\n"
    ir += "  %L_ptr = getelementptr i8, i8 addrspace(1)* %L_buf, i64 %L_byte\n"
    if memL == .FP32 {
      ir += "  %L_typed = bitcast i8 addrspace(1)* %L_ptr to float addrspace(1)*\n"
      ir += "  %L_sram = load float, float addrspace(1)* %L_typed\n"
    } else {
      let tL = irTypeName(memL)
      ir += "  %L_typed = bitcast i8 addrspace(1)* %L_ptr to \(tL) addrspace(1)*\n"
      ir += "  %L_raw = load \(tL), \(tL) addrspace(1)* %L_typed\n"
      ir += "  %L_sram = fpext \(tL) %L_raw to float\n"
    }

    // Compute D = sum(dO * O) across the head dimension
    // D_sram = reduce_sum(dO[row,:] * O[row,:])
    ir += generateComputeD(
      prefix: "cd_",
      D: D, paddedD: paddedD, blockH: blockH,
      headLoopFloor: headLoopFloor, headEdge: headEdge,
      parallelDim: parallelDim,
      cachedO: false, cacheddO: isCacheddO,
      leadingDimO: leadingDim(.O), leadingDimdO: leadingDim(.dO),
      memO: memO, regO: regO,
      memdO: memdO, regdO: regdO,
      transposedO: transposed(.O), transposeddO: transposed(.dO)
    )

    // MARK: - Traversal Loop

    ir += """

      ; === Traversal loop ===
      br label %bq_before_loop
    bq_before_loop:
      br label %bq_loop_header

    bq_loop_header:
      %bq_c = phi i32 [0, %bq_before_loop], [%bq_c_next, %bq_loop_latch]

    """

    // Phi nodes for dQ accumulators
    for i in 0..<dqCount {
      ir += "  %dq_phi_\(i) = phi \(irVecType(regdQ)) [%dq_init_\(i), %bq_before_loop], [%dq_acc_\(i), %bq_loop_latch]\n"
    }

    ir += """

      %bq_loop_done = icmp uge i32 %bq_c, \(traversalDim)
      br i1 %bq_loop_done, label %bq_cleanup, label %bq_loop_body

    bq_loop_body:

    """

    // Step 1: S = Q * K^T (outer product)
    for i in 0..<sSramCount {
      ir += "  \(irZeroVec64(result: "%s_init_\(i)", precision: regS))\n"
    }

    ir += generateOuterProduct(
      prefix: "qo_",
      A: .Q, B: .K, C_name: "s",
      sSramCount: sSramCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bq_c",
      regA: regQ, regB: regK, regC: regS,
      memA: memQ, memB: memK,
      leadingDimA: leadingDim(.Q), leadingDimB: leadingDim(.K),
      leadingBlockDimA: leadingBlockDim(.Q), leadingBlockDimB: leadingBlockDim(.K),
      cachedA: isCachedQ, transposedA: transposed(.Q), transposedB: transposed(.K),
      cachePrefix: "cq_"
    )

    // Mask S at traversal edge (same as forward — zeroes S beyond traversalDim)
    ir += generateMaskEdge(
      prefix: "qm_",
      sSramCount: sSramCount,
      blockT: blockT, traversalDim: traversalDim,
      traversalOffset: "%bq_c",
      regS: regS, scaleFactor: scaleFactor
    )

    // Step 2: P = exp2(S * scaleFactor - L)
    // Unlike forward (online softmax), backward uses stored L.
    ir += generateSoftmaxFromL(
      prefix: "qs_",
      sSramCount: sSramCount,
      regS: regS, regP: regP,
      scaleFactor: scaleFactor,
      lScalar: "%L_sram",
      sSource: "qm_s"  // masked S from generateMaskEdge
    )

    // Step 3: dP = dO * V^T (outer product)
    for i in 0..<sSramCount {
      ir += "  \(irZeroVec64(result: "%dp_init_\(i)", precision: regdP))\n"
    }

    ir += generateOuterProduct(
      prefix: "qp_",
      A: .dO, B: .V, C_name: "dp",
      sSramCount: sSramCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bq_c",
      regA: regdO, regB: regV, regC: regdP,
      memA: memdO, memB: memV,
      leadingDimA: leadingDim(.dO), leadingDimB: leadingDim(.V),
      leadingBlockDimA: leadingBlockDim(.dO), leadingBlockDimB: leadingBlockDim(.V),
      cachedA: isCacheddO, transposedA: transposed(.dO), transposedB: transposed(.V),
      cachePrefix: "cdo_"
    )

    // Step 4: dS = P * (dP * derivScale - D_sram)
    ir += generateDerivativeSoftmax(
      prefix: "qd_",
      sSramCount: sSramCount,
      regP: regP, regdP: regdP, regdS: regdS,
      derivScale: derivScale,
      dScalar: "%D_sram",
      pSource: "qs_p",   // from generateSoftmaxFromL with prefix "qs_"
      dpSource: "dp_final"   // from generateOuterProduct with C_name "dp"
    )

    // Step 5: dQ += dS * K (accumulate)
    ir += generateAccumulate(
      prefix: "qa_",
      A: .dS, B: .K, C_name: "dq",
      accCount: dqCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bq_c",
      regA: regdS, regB: regK, regC: regdQ,
      memB: memK,
      leadingDimB: leadingDim(.K),
      leadingBlockDimB: leadingBlockDim(.K),
      transposedB: transposed(.K),
      cachedC: isCacheddQ,
      isFinalScale: false,
      scaleCorrection: "",
      aSourcePrefix: "qd_ds"  // from generateDerivativeSoftmax with prefix "qd_"
    )

    // MARK: - Loop Latch

    ir += "  br label %bq_loop_latch\n"
    ir += """

    bq_loop_latch:

    """

    for i in 0..<dqCount {
      ir += "  %dq_acc_\(i) = phi \(irVecType(regdQ)) [%qa_dq_final_\(i), %qa_after_head]\n"
    }

    ir += """

      %bq_c_next = add i32 %bq_c, \(blockT)
      br label %bq_loop_header

    """

    // MARK: - Cleanup: store dQ, store D

    ir += generateBackwardQueryCleanup(
      prefix: "qc_",
      dqCount: dqCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regdQ: regdQ, memdQ: memdQ,
      memD: memD,
      leadingDimdQ: leadingDim(.dQ),
      leadingBlockDimdQ: leadingBlockDim(.dQ),
      transposeddQ: transposed(.dQ),
      cacheddQ: isCacheddQ,
      derivScale: derivScale
    )

    return ir
  }

  // MARK: - Backward Key-Value

  func generateBackwardKeyValueKernel(
    desc: MonolithicDescriptor,
    R: UInt32, C: UInt32, D: UInt32,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    parallelDim: UInt32, traversalDim: UInt32,
    paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    sSramCount: Int,
    scaleFactor: Float, tgFloats: Int,
    leadingDim: (AttentionOperand) -> UInt32,
    leadingBlockDim: (AttentionOperand) -> UInt32,
    memPrec: (AttentionOperand) -> GEMMOperandPrecision,
    regPrec: (AttentionOperand) -> GEMMOperandPrecision,
    memSize: (AttentionOperand) -> UInt32
  ) -> String {
    var ir = ""

    let regS = regPrec(.S)
    let regP = regPrec(.P)
    let regdP = regPrec(.dP)
    let regdS = regPrec(.dS)
    let regQ = regPrec(.Q)
    let regK = regPrec(.K)
    let regV = regPrec(.V)
    let regdO = regPrec(.dO)
    let regdK = regPrec(.dK)
    let regdV = regPrec(.dV)
    let memQ = memPrec(.Q)
    let memK = memPrec(.K)
    let memV = memPrec(.V)
    let memdO = memPrec(.dO)
    let memdK = memPrec(.dK)
    let memdV = memPrec(.dV)

    let isCachedK = cached(.K)
    let isCachedV = cached(.V)
    let isCacheddK = cached(.dK)
    let isCacheddV = cached(.dV)

    let dkCount = Int(paddedD / 8)
    let dvCount = Int(paddedD / 8)

    let derivScale = 1.0 / Float(headDimension).squareRoot()

    // MARK: - Setup

    ir += "  ; === Backward Key-Value setup ===\n"

    // Cache load K if cached
    if isCachedK {
      ir += generateCacheLoad(
        operand: .K, prefix: "ck_",
        parallelDim: parallelDim, D: D, paddedD: paddedD,
        blockP: blockP, blockH: blockH,
        leadingDim: leadingDim(.K), leadingBlockDim: leadingBlockDim(.K),
        memPrec: memK, regPrec: regK,
        transposed: transposed(.K)
      )
    }

    // Cache load V if cached
    if isCachedV {
      ir += generateCacheLoad(
        operand: .V, prefix: "cv_",
        parallelDim: parallelDim, D: D, paddedD: paddedD,
        blockP: blockP, blockH: blockH,
        leadingDim: leadingDim(.V), leadingBlockDim: leadingBlockDim(.V),
        memPrec: memV, regPrec: regV,
        transposed: transposed(.V)
      )
    }

    // Initialize dK and dV accumulators to zero
    for i in 0..<dkCount {
      ir += "  \(irZeroVec64(result: "%dk_init_\(i)", precision: regdK))\n"
    }
    for i in 0..<dvCount {
      ir += "  \(irZeroVec64(result: "%dv_init_\(i)", precision: regdV))\n"
    }

    // MARK: - Traversal Loop

    ir += """

      ; === Traversal loop ===
      br label %bkv_before_loop
    bkv_before_loop:
      br label %bkv_loop_header

    bkv_loop_header:
      %bkv_r = phi i32 [0, %bkv_before_loop], [%bkv_r_next, %bkv_loop_latch]

    """

    // Phi nodes for dK and dV accumulators
    for i in 0..<dkCount {
      ir += "  %dk_phi_\(i) = phi \(irVecType(regdK)) [%dk_init_\(i), %bkv_before_loop], [%dk_acc_\(i), %bkv_loop_latch]\n"
    }
    for i in 0..<dvCount {
      ir += "  %dv_phi_\(i) = phi \(irVecType(regdV)) [%dv_init_\(i), %bkv_before_loop], [%dv_acc_\(i), %bkv_loop_latch]\n"
    }

    ir += """

      %bkv_loop_done = icmp uge i32 %bkv_r, \(traversalDim)
      br i1 %bkv_loop_done, label %bkv_cleanup, label %bkv_loop_body

    bkv_loop_body:

    """

    // L and D are per-row scalars that vary per-column of S^T (each column
    // = different traversal row). They are loaded per-tile inside
    // generateSoftmaxFromLVector and generateDerivativeSoftmaxVector
    // respectively, not once per traversal block.

    // Step 1: S^T = K * Q^T (outer product)
    for i in 0..<sSramCount {
      ir += "  \(irZeroVec64(result: "%s_init_\(i)", precision: regS))\n"
    }

    ir += generateOuterProduct(
      prefix: "ko_",
      A: .K, B: .Q, C_name: "s",
      sSramCount: sSramCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bkv_r",
      regA: regK, regB: regQ, regC: regS,
      memA: memK, memB: memQ,
      leadingDimA: leadingDim(.K), leadingDimB: leadingDim(.Q),
      leadingBlockDimA: leadingBlockDim(.K), leadingBlockDimB: leadingBlockDim(.Q),
      cachedA: isCachedK, transposedA: transposed(.K), transposedB: transposed(.Q),
      cachePrefix: "ck_"
    )

    // Mask S^T at traversal edge
    ir += generateMaskEdge(
      prefix: "km_",
      sSramCount: sSramCount,
      blockT: blockT, traversalDim: traversalDim,
      traversalOffset: "%bkv_r",
      regS: regS, scaleFactor: scaleFactor
    )

    // Step 2: P^T = exp2(S^T * scaleFactor - L)
    // L varies per-column of S^T (each column = different traversal row),
    // so it must be loaded per-tile from the L buffer.
    ir += generateSoftmaxFromLVector(
      prefix: "ks_",
      sSramCount: sSramCount,
      regS: regS, regP: regP,
      scaleFactor: scaleFactor,
      lBufferName: "%L_buf",
      traversalOffset: "%bkv_r",
      traversalDim: traversalDim,
      memL: memPrec(.L),
      sSource: "km_s"  // masked S from generateMaskEdge
    )

    // Step 3: dV += P^T * dO (accumulate)
    ir += generateAccumulate(
      prefix: "kv_",
      A: .P, B: .dO, C_name: "dv",
      accCount: dvCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bkv_r",
      regA: regP, regB: regdO, regC: regdV,
      memB: memdO,
      leadingDimB: leadingDim(.dO),
      leadingBlockDimB: leadingBlockDim(.dO),
      transposedB: transposed(.dO),
      cachedC: isCacheddV,
      isFinalScale: false,
      scaleCorrection: "",
      aSourcePrefix: "ks_p"  // from generateSoftmaxFromLVector with prefix "ks_"
    )

    // Step 4: dP^T = V * dO^T (outer product)
    for i in 0..<sSramCount {
      ir += "  \(irZeroVec64(result: "%dp_init_\(i)", precision: regdP))\n"
    }

    ir += generateOuterProduct(
      prefix: "kp_",
      A: .V, B: .dO, C_name: "dp",
      sSramCount: sSramCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bkv_r",
      regA: regV, regB: regdO, regC: regdP,
      memA: memV, memB: memdO,
      leadingDimA: leadingDim(.V), leadingDimB: leadingDim(.dO),
      leadingBlockDimA: leadingBlockDim(.V), leadingBlockDimB: leadingBlockDim(.dO),
      cachedA: isCachedV, transposedA: transposed(.V), transposedB: transposed(.dO),
      cachePrefix: "cv_"
    )

    // Step 5: dS^T = P^T * (dP^T * derivScale - D_stored)
    // D_stored is pre-scaled (D_raw * derivScale) from bwd_q kernel.
    // D varies per-column of S^T (each column = different traversal row),
    // so it must be loaded per-tile, not once per traversal block.
    ir += generateDerivativeSoftmaxVector(
      prefix: "kd_",
      sSramCount: sSramCount,
      regP: regP, regdP: regdP, regdS: regdS,
      derivScale: derivScale,
      dBufferName: "%D_buf",
      traversalOffset: "%bkv_r",
      traversalDim: traversalDim,
      memD: memPrec(.D),
      pSource: "ks_p",   // from generateSoftmaxFromLVector with prefix "ks_"
      dpSource: "dp_final"    // from generateOuterProduct with C_name "dp"
    )

    // Step 6: dK += dS^T * Q (accumulate)
    ir += generateAccumulate(
      prefix: "kk_",
      A: .dS, B: .Q, C_name: "dk",
      accCount: dkCount,
      blockP: blockP, blockT: blockT, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim, traversalDim: traversalDim,
      traversalOffset: "%bkv_r",
      regA: regdS, regB: regQ, regC: regdK,
      memB: memQ,
      leadingDimB: leadingDim(.Q),
      leadingBlockDimB: leadingBlockDim(.Q),
      transposedB: transposed(.Q),
      cachedC: isCacheddK,
      isFinalScale: false,
      scaleCorrection: "",
      aSourcePrefix: "kd_ds"  // from generateDerivativeSoftmaxVector with prefix "kd_"
    )

    // MARK: - Loop Latch

    ir += "  br label %bkv_loop_latch\n"
    ir += """

    bkv_loop_latch:

    """

    for i in 0..<dkCount {
      ir += "  %dk_acc_\(i) = phi \(irVecType(regdK)) [%kk_dk_final_\(i), %kk_after_head]\n"
    }
    for i in 0..<dvCount {
      ir += "  %dv_acc_\(i) = phi \(irVecType(regdV)) [%kv_dv_final_\(i), %kk_after_head]\n"
    }

    ir += """

      %bkv_r_next = add i32 %bkv_r, \(blockT)
      br label %bkv_loop_header

    """

    // MARK: - Cleanup: store dK, store dV

    ir += generateBackwardKeyValueCleanup(
      prefix: "kc_",
      dkCount: dkCount, dvCount: dvCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regdK: regdK, memdK: memdK,
      regdV: regdV, memdV: memdV,
      leadingDimdK: leadingDim(.dK),
      leadingBlockDimdK: leadingBlockDim(.dK),
      leadingDimdV: leadingDim(.dV),
      leadingBlockDimdV: leadingBlockDim(.dV),
      transposeddK: transposed(.dK),
      transposeddV: transposed(.dV),
      cacheddK: isCacheddK,
      cacheddV: isCacheddV
    )

    return ir
  }
}
