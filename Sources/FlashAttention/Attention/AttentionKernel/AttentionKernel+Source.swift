//
//  AttentionKernel+Source.swift
//  FlashAttention
//
//  Generates a complete monolithic LLVM IR kernel for attention.
//  Each (R, C, D, kernel_type, precision, transpose_state) config → unique IR.
//  IR → MetalASM.assemble() → metallib → makeLibrary(data:) → pipeline.
//
//  All dimensions baked as literals (no function constants).
//  Kernel runs straight through: setup → loop → cleanup (no state machine).
//  Dynamic traversal loop with phi nodes for live SSA state.
//

extension AttentionKernel {

  /// Problem-specific constants baked directly into IR as literals.
  public struct MonolithicDescriptor {
    public var R: UInt32 = 0          // row sequence length
    public var C: UInt32 = 0          // column sequence length
    /// Leading dimensions derived from transpose state + sequence lengths.
    public var leadingDimensions: [AttentionOperand: UInt32] = [:]

    public init() {}
  }

  /// Generate a complete LLVM IR module for a monolithic attention kernel.
  ///
  /// The returned string can be passed directly to `MetalASM.assemble(ir:)`
  /// to produce a metallib loadable via `MTLDevice.makeLibrary(data:)`.
  public func createSource(descriptor desc: MonolithicDescriptor) -> String {

    let R = desc.R
    let C = desc.C
    let D = UInt32(headDimension)

    // Compute derived dimensions
    let blockP = blockDimensions.parallelization  // parallelization block
    let blockT = blockDimensions.traversal         // traversal block
    let blockH = blockDimensions.head              // head block

    // For the forward kernel:
    //   parallelization = R, traversal = C
    //   Q, O are parallelization operands; K, V are traversal operands
    let parallelDim: UInt32
    let traversalDim: UInt32
    switch type {
    case .forward, .backwardQuery:
      parallelDim = R
      traversalDim = C
    case .backwardKeyValue:
      parallelDim = C
      traversalDim = R
    }

    // Padded head dimension (round up to 8)
    let paddedD = (D + 7) / 8 * 8

    // Padded head edge
    let headEdge: UInt32 = {
      let blockDim = UInt32(blockH)
      let remainder = D % blockDim
      var output = (remainder == 0) ? blockDim : remainder
      output = (output + 7) / 8 * 8
      return output
    }()

    // Leading dimension helpers
    func leadingDim(_ operand: AttentionOperand) -> UInt32 {
      if let ld = desc.leadingDimensions[operand] {
        return ld
      }
      // Default: if transposed, leading dim = sequence length; else = D
      if transposed(operand) {
        switch operand {
        case .Q, .dQ: return R
        case .K, .dK: return C
        case .V, .dV: return C
        case .O, .dO: return R
        default: fatalError("Unrecognized operand.")
        }
      } else {
        return D
      }
    }

    func leadingBlockDim(_ operand: AttentionOperand) -> UInt32 {
      UInt32(leadingBlockDimension(operand))
    }

    // Memory precision helpers
    func memPrec(_ operand: AttentionOperand) -> GEMMOperandPrecision {
      memoryPrecisions[operand]!
    }
    func regPrec(_ operand: AttentionOperand) -> GEMMOperandPrecision {
      registerPrecisions[operand]!
    }
    func memSize(_ operand: AttentionOperand) -> UInt32 {
      UInt32(memPrec(operand).size)
    }

    // S/P accumulator count = traversal_block / 8
    let sSramCount = Int(blockT / 8)

    // O accumulator count when cached = paddedD / 8
    // O accumulator count when not cached = blockH / 8
    let oCachedCount = Int(paddedD / 8)

    // Head loop bounds
    let headLoopEnd = paddedD
    let headLoopFloor = headLoopEnd - headLoopEnd % UInt32(blockH)

    // TG memory size (in floats, for the global buffer)
    let tgFloats = max(8192, (Int(threadgroupMemoryAllocation) + 128 + 3) / 4)

    // Scale factor: log2(e) / sqrt(D)
    let logBase2E: Float = 1.442695041
    let scaleFactor = logBase2E / Float(headDimension).squareRoot()

    // Determine which multiply-accumulate intrinsics we need
    var maDeclarations = Set<String>()

    // For S = Q * K^T (outer product)
    let regS = regPrec(.S)
    let regQ = regPrec(.Q)
    let regK = regPrec(.K)
    maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regQ, B: regK, C: regS))

    // For O += P * V (accumulate)
    let regO = regPrec(.O)
    let regP = regPrec(.P)
    let regV = regPrec(.V)
    maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regP, B: regV, C: regO))

    if type == .backwardQuery {
      // dP = dO * V^T
      let regdP = regPrec(.dP)
      let regdO = regPrec(.dO)
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regdO, B: regV, C: regdP))
      // dQ += dS * K
      let regdS = regPrec(.dS)
      let regdQ = regPrec(.dQ)
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regdS, B: regK, C: regdQ))
    }

    if type == .backwardKeyValue {
      // S^T = K * Q^T
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regK, B: regQ, C: regS))
      // dV += P^T * dO
      let regdO = regPrec(.dO)
      let regdV = regPrec(.dV)
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regP, B: regdO, C: regdV))
      // dP^T = V * dO^T
      let regdP = regPrec(.dP)
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regV, B: regdO, C: regdP))
      // dK += dS^T * Q
      let regdS = regPrec(.dS)
      let regdK = regPrec(.dK)
      maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regdS, B: regQ, C: regdK))
    }

    // MARK: - IR Generation

    var ir = ""

    // Module header
    ir += irModuleHeader(tgBufferFloats: tgFloats) + "\n"
    ir += irIntrinsicDeclarations() + "\n"
    ir += irAttentionIntrinsicDeclarations() + "\n"

    // Multiply-accumulate intrinsic declarations
    for decl in maDeclarations.sorted() {
      ir += "  \(decl)\n"
    }

    // Kernel function signature: 10 device buffers + 1 TG buffer + gid + sidx + lane_id
    ir += """

    define void @attention(
        i8 addrspace(1)* noundef "air-buffer-no-alias" %Q,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %K,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %V,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %O,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %L_buf,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %D_buf,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %dO,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %dV,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %dK,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %dQ,
        i8 addrspace(3)* noundef %tg_base,
        <3 x i32> noundef %gid,
        i16 noundef %sidx_i16,
        i16 noundef %lane_id_i16
    ) local_unnamed_addr #0 {
    entry:
      %sidx = zext i16 %sidx_i16 to i32
      %lane_id = zext i16 %lane_id_i16 to i32
      %gid_x = extractelement <3 x i32> %gid, i64 0

      ; Event alloca for async copy
      %ev = alloca [2 x %event_t addrspace(3)*], align 8
      %ev_i8 = bitcast [2 x %event_t addrspace(3)*]* %ev to i8*
      call void @llvm.lifetime.start.p0i8(i64 16, i8* nonnull %ev_i8) #4

      ; === Morton order computation ===
      %q = lshr i32 %lane_id, 2
      %m_floor = and i32 %q, 16380
      %h = lshr i32 %lane_id, 1
      %m_in_quad = and i32 %h, 3
      %morton_y = or i32 %m_floor, %m_in_quad
      %n_floor = and i32 %h, 4
      %n_in_quad_s = shl i32 %lane_id, 1
      %n_in_quad = and i32 %n_in_quad_s, 2
      %morton_x = or i32 %n_floor, %n_in_quad

      ; parallelization_group_offset = gid.x * blockP
      %par_group_off = mul i32 %gid_x, \(blockP)

      ; Early return if entire group is out of bounds
      %early_oob = icmp uge i32 %par_group_off, \(parallelDim)
      br i1 %early_oob, label %exit, label %valid_group

    valid_group:
      ; Gate async copies to simdgroup 0
      %is_sidx0 = icmp eq i32 %sidx, 0

      ; Compute thread offsets within group
      ; oig_y = sidx * 8 + morton_y (parallelization offset within group)
      %oig_y_base = mul i32 %sidx, 8
      %oig_y = add i32 %oig_y_base, %morton_y

      ; unsafe_par_thread_off = par_group_off + oig_y
      %unsafe_par_off = add i32 %par_group_off, %oig_y
      ; clamped_par_off = min(unsafe_par_off, parallelDim - 1)
      %par_dim_m1 = sub i32 \(parallelDim), 1
      %par_cmp = icmp ult i32 %unsafe_par_off, \(parallelDim)
      %clamped_par_off = select i1 %par_cmp, i32 %unsafe_par_off, i32 %par_dim_m1

    """

    // Dispatch to kernel type
    switch type {
    case .forward:
      ir += generateForwardKernel(
        desc: desc, R: R, C: C, D: D,
        blockP: blockP, blockT: blockT, blockH: blockH,
        parallelDim: parallelDim, traversalDim: traversalDim,
        paddedD: paddedD, headEdge: headEdge,
        headLoopFloor: headLoopFloor,
        oCachedCount: oCachedCount, sSramCount: sSramCount,
        scaleFactor: scaleFactor, tgFloats: tgFloats,
        leadingDim: leadingDim, leadingBlockDim: leadingBlockDim,
        memPrec: memPrec, regPrec: regPrec, memSize: memSize
      )
    case .backwardQuery:
      ir += generateBackwardQueryKernel(
        desc: desc, R: R, C: C, D: D,
        blockP: blockP, blockT: blockT, blockH: blockH,
        parallelDim: parallelDim, traversalDim: traversalDim,
        paddedD: paddedD, headEdge: headEdge,
        headLoopFloor: headLoopFloor,
        sSramCount: sSramCount,
        scaleFactor: scaleFactor, tgFloats: tgFloats,
        leadingDim: leadingDim, leadingBlockDim: leadingBlockDim,
        memPrec: memPrec, regPrec: regPrec, memSize: memSize
      )
    case .backwardKeyValue:
      ir += generateBackwardKeyValueKernel(
        desc: desc, R: R, C: C, D: D,
        blockP: blockP, blockT: blockT, blockH: blockH,
        parallelDim: parallelDim, traversalDim: traversalDim,
        paddedD: paddedD, headEdge: headEdge,
        headLoopFloor: headLoopFloor,
        sSramCount: sSramCount,
        scaleFactor: scaleFactor, tgFloats: tgFloats,
        leadingDim: leadingDim, leadingBlockDim: leadingBlockDim,
        memPrec: memPrec, regPrec: regPrec, memSize: memSize
      )
    }

    // Exit block
    ir += """

    exit:
      call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %ev_i8) #4
      ret void
    }

    """

    // Metadata
    ir += irAttentionKernelMetadata() + "\n"

    return ir
  }
}
