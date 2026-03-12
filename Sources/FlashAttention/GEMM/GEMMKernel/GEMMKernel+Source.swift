//
//  GEMMKernel+Source.swift
//  FlashAttention
//
//  Generates a complete monolithic LLVM IR kernel for GEMM.
//  All K iterations are unrolled inline: async copy → barrier → multiply.
//  C accumulators stay in registers throughout — no save/restore to TG.
//  No dispatch loop, no command protocol, no visible function overhead.
//
//  Assembled in-process via MetalASM: parse → bitcode → metallib.
//

extension GEMMKernel {

  /// All the problem-specific constants that would have been function constants
  /// in the reverse-linking path. Now baked directly into the IR as literals.
  public struct MonolithicDescriptor {
    public var M: UInt32 = 0
    public var N: UInt32 = 0
    public var K: UInt32 = 0
    public var leadingDimensionA: UInt32 = 0
    public var leadingDimensionB: UInt32 = 0
    public var leadingDimensionC: UInt32 = 0
    public var loadPreviousC: Bool = false

    public init() {}
  }

  /// Generate a complete LLVM IR module for a monolithic GEMM kernel.
  ///
  /// The returned string can be passed directly to `MetalASM.assemble(ir:)`
  /// to produce a metallib loadable via `MTLDevice.makeLibrary(data:)`.
  public func createSource(descriptor desc: MonolithicDescriptor) -> String {
    let M = desc.M
    let N = desc.N
    let K = desc.K
    let ldA = desc.leadingDimensionA
    let ldB = desc.leadingDimensionB
    let ldC = desc.leadingDimensionC

    // Derived constants (same logic as createConstants() in Source.swift)
    let M_group = blockDimensions.M
    let N_group = blockDimensions.N
    let K_group = blockDimensions.K
    let regM = registerM
    let regN = registerN

    let M_edge = M - (M % UInt32(M_group))
    let N_edge = N - (N % UInt32(N_group))
    let M_remainder = (M % UInt32(regM) == 0) ? regM : UInt16(M % UInt32(regM))
    let N_remainder = (N % UInt32(regN) == 0) ? regN : UInt16(N % UInt32(regN))

    let M_shift: UInt16 = (M < UInt32(M_group)) ? 0 : regM - M_remainder
    let N_shift: UInt16 = (N < UInt32(N_group)) ? 0 : regN - N_remainder

    let asyncIterationsStart: UInt32 = preferAsyncLoad ? 0 : (K - (K % UInt32(K_group)))
    let numAsyncIterations = (K > asyncIterationsStart)
      ? (K - asyncIterationsStart + UInt32(K_group) - 1) / UInt32(K_group)
      : 0

    let cSramCount = Int(regM / 8) * Int(regN / 8)
    let blockBytesA = Int(blockBytes("A"))

    // Precision types
    let regA = registerPrecisions.A
    let regB = registerPrecisions.B
    let regC = registerPrecisions.C
    let _ = memoryPrecisions.A  // used in sub-generators
    let _ = memoryPrecisions.B
    let _ = memoryPrecisions.C

    // Determine which multiply-accumulate intrinsic variants we need.
    var maDeclarations = Set<String>()
    maDeclarations.insert(irMultiplyAccumulateDeclaration(A: regA, B: regB, C: regC))

    // TG buffer: just enough for A block + B block (no C save area needed!)
    let tgFloats = max(8192, (Int(threadgroupMemoryAllocation) + 128 + 3) / 4)

    // MARK: - IR Generation

    var ir = ""

    // Module header
    ir += irModuleHeader(tgBufferFloats: tgFloats) + "\n"
    ir += irIntrinsicDeclarations() + "\n"

    // Multiply-accumulate intrinsic declarations
    for decl in maDeclarations {
      ir += "  \(decl)\n"
    }

    // Kernel function signature: 3 device buffers + 1 TG buffer + gid + sidx + lane_id
    ir += """

    define void @gemm(
        i8 addrspace(1)* noundef "air-buffer-no-alias" %A,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %B,
        i8 addrspace(1)* noundef "air-buffer-no-alias" %C,
        i8 addrspace(3)* noundef %tg_base,
        <3 x i32> noundef %gid,
        i16 noundef %sidx_i16,
        i16 noundef %lane_id_i16
    ) local_unnamed_addr #0 {
    entry:
      %sidx = zext i16 %sidx_i16 to i32
      %lane_id = zext i16 %lane_id_i16 to i32
      %gid_x = extractelement <3 x i32> %gid, i64 0
      %gid_y = extractelement <3 x i32> %gid, i64 1

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

      ; sid.x = sidx % splits.N, sid.y = sidx / splits.N
      %sid_x = urem i32 %sidx, \(splits.N)
      %sid_y = udiv i32 %sidx, \(splits.N)

      ; offset_in_group = (sid.x * regN + morton_x, sid.y * regM + morton_y)
      %oig_x_base = mul i32 %sid_x, \(regN)
      %oig_x = add i32 %oig_x_base, %morton_x
      %oig_y_base = mul i32 %sid_y, \(regM)
      %oig_y = add i32 %oig_y_base, %morton_y

      ; M_offset and N_offset
      %M_offset_base = mul i32 %gid_y, \(M_group)
      %N_offset_base = mul i32 %gid_x, \(N_group)

      ; Gate async copies to simdgroup 0
      %is_sidx0 = icmp eq i32 %sidx, 0

    """

    // Edge shifting
    if M_shift != 0 {
      ir += """
        ; M edge shift
        %m_edge_cmp = icmp uge i32 %M_offset_base, \(M_edge)
        %m_shift_val = select i1 %m_edge_cmp, i32 \(M_shift), i32 0
        %M_offset = sub i32 %M_offset_base, %m_shift_val

      """
    } else {
      ir += "  %M_offset = add i32 %M_offset_base, 0\n"
    }

    if N_shift != 0 {
      ir += """
        ; N edge shift
        %n_edge_cmp = icmp uge i32 %N_offset_base, \(N_edge)
        %n_shift_val = select i1 %n_edge_cmp, i32 \(N_shift), i32 0
        %N_offset = sub i32 %N_offset_base, %n_shift_val

      """
    } else {
      ir += "  %N_offset = add i32 %N_offset_base, 0\n"
    }

    // Out-of-bounds check
    ir += """
      ; out_of_bounds = (M_offset + sid.y * regM >= M) || (N_offset + sid.x * regN >= N)
      %oob_m_off = add i32 %M_offset, %oig_y_base
      %oob_m = icmp uge i32 %oob_m_off, \(M)
      %oob_n_off = add i32 %N_offset, %oig_x_base
      %oob_n = icmp uge i32 %oob_n_off, \(N)
      %out_of_bounds = or i1 %oob_m, %oob_n

    """

    // Initialize C accumulators to zero
    for i in 0..<cSramCount {
      ir += "  \(irZeroVec64(result: "%c_init_\(i)", precision: regC))\n"
    }

    // === loadPreviousC: async copy C from device → TG, load into accumulators ===
    if desc.loadPreviousC {
      ir += generateInlineLoadC(
        desc: desc, ldC: ldC,
        M_group: M_group, N_group: N_group,
        cSramCount: cSramCount
      )
      // After this block, %c_start_N hold the loaded values (or zero if OOB)
    } else {
      // C starts as zero
      for i in 0..<cSramCount {
        ir += "  %c_start_\(i) = bitcast \(irVecType(regC)) %c_init_\(i) to \(irVecType(regC))\n"
      }
    }

    // === Main K loop: fully unrolled, C stays in registers ===
    //
    // Structure for each async iteration i (0..numAsyncIterations-1):
    //   1. Inline async copy (A+B) for k = asyncIterationsStart + i * K_group
    //   2. Barrier (wait for data)
    //   3. Multiply-accumulate from TG data (K_group steps of 8)
    //
    // Between iterations, C accumulators are SSA-chained:
    //   %c_iter0_N → %c_iter1_N → ... → %c_final_N

    // Track current C accumulator names
    var cNames = (0..<cSramCount).map { "%c_start_\($0)" }

    for iter in 0..<Int(numAsyncIterations) {
      let kStart = asyncIterationsStart + UInt32(iter) * UInt32(K_group)
      let kRemaining = K - kStart
      let kTile = min(UInt32(K_group), kRemaining)
      let kSteps = Int((kTile + 7) / 8)
      let p = "it\(iter)_"

      // 1. Inline async copy for A and B (gated to sidx == 0)
      ir += "  br i1 %is_sidx0, label %\(p)do_copy, label %\(p)skip_copy\n\n"
      ir += "\(p)do_copy:\n"
      ir += generateInlineDualCopy(
        prefix: p,
        kStart: kStart, kTile: kTile,
        ldA: ldA, ldB: ldB,
        M_group: M_group, N_group: N_group, K_group: K_group,
        blockBytesA: blockBytesA,
        M: M, N: N
      )
      ir += "  br label %\(p)after_copy\n\n"
      ir += "\(p)skip_copy:\n"
      ir += "  br label %\(p)after_copy\n\n"
      ir += "\(p)after_copy:\n"

      // 2. Barrier after copy (all threads must wait)
      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

      // 3. Multiply-accumulate from TG
      let newNames = generateInlineTGMultiply(
        prefix: p,
        kSteps: kSteps,
        regM: regM, regN: regN,
        blockBytesA: blockBytesA,
        cSramCount: cSramCount,
        cInputNames: cNames
      )
      ir += newNames.ir

      // If not last iteration, barrier before next copy overwrites TG
      if iter < Int(numAsyncIterations) - 1 {
        ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"
      }

      cNames = newNames.outputNames
    }

    // === Store C to TG and async copy to device ===
    // After all multiply iterations, cNames holds the final accumulators.
    // We use %c_final_N aliases for the store logic.
    for i in 0..<cSramCount {
      ir += "  %c_final_\(i) = bitcast \(irVecType(regC)) \(cNames[i]) to \(irVecType(regC))\n"
    }

    // CRITICAL: Barrier before C store to ensure all simdgroups are done reading
    // A/B from TG (since C store overwrites the A/B area in TG).
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    // Fast path: direct register → device store for non-edge threadgroups.
    // Static condition: M >= M_group && N >= N_group (checked at codegen time).
    // Runtime condition: not an edge threadgroup (M_offset_base < M_edge && N_offset_base < N_edge).
    let canDirectStore = !preferAsyncStore
      && M >= UInt32(M_group)
      && N >= UInt32(N_group)

    if canDirectStore {
      // Runtime edge check
      ir += "  ; Fast path: direct device store for non-edge threadgroups\n"
      ir += "  %fast_not_edge_m = icmp ult i32 %M_offset_base, \(M_edge)\n"
      ir += "  %fast_not_edge_n = icmp ult i32 %N_offset_base, \(N_edge)\n"
      ir += "  %fast_use_direct = and i1 %fast_not_edge_m, %fast_not_edge_n\n"
      ir += "  br i1 %fast_use_direct, label %c_direct_store, label %c_slow_store\n\n"

      // Fast path: direct store
      ir += "c_direct_store:\n"
      ir += generateDirectStoreCToDevice(
        desc: desc, ldC: ldC,
        cSramCount: cSramCount
      )
      ir += "  br label %exit\n\n"

      // Slow path: TG → async copy (edge threadgroups)
      ir += "c_slow_store:\n"
    }

    // Store C registers to TG block at offset 0 (no cmd area needed anymore)
    ir += generateStoreCToTGInline(
      desc: desc, ldC: ldC,
      cSramCount: cSramCount,
      M_group: M_group, N_group: N_group
    )

    // Barrier so all threads finish writing to TG before async store
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    // Inline async store: TG → device (gated to sidx == 0)
    ir += "  br i1 %is_sidx0, label %cs_do_store, label %cs_skip_store\n\n"
    ir += "cs_do_store:\n"
    ir += generateInlineStoreC(
      desc: desc, ldC: ldC,
      M_group: M_group, N_group: N_group
    )
    // generateInlineStoreC ends with br to exit, so no fall-through needed
    ir += "cs_skip_store:\n"
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n"
    ir += "  br label %exit\n\n"

    // Exit
    ir += """

    exit:
      call void @llvm.lifetime.end.p0i8(i64 16, i8* nonnull %ev_i8) #4
      ret void
    }

    """

    // Metadata
    ir += irGEMMKernelMetadata() + "\n"

    return ir
  }

  // MARK: - Inline Async Copy (A+B)

  /// Generate inline dual async copy: A and B from device → TG.
  /// All parameters are computed statically (no TG cmd area).
  private func generateInlineDualCopy(
    prefix p: String,
    kStart: UInt32, kTile: UInt32,
    ldA: UInt32, ldB: UInt32,
    M_group: UInt16, N_group: UInt16, K_group: UInt16,
    blockBytesA: Int,
    M: UInt32, N: UInt32
  ) -> String {
    let memA_size = UInt32(memoryPrecisions.A.size)
    let memB_size = UInt32(memoryPrecisions.B.size)

    var ir = "  ; === Async copy iter \(p) (k=\(kStart)) ===\n"

    // A source offset: apply_offset(A, ldA, (kStart, M_offset), A_trans)
    if transposeState.A {
      ir += "  %\(p)a_row = mul i32 \(kStart), \(ldA)\n"
      ir += "  %\(p)a_off32 = add i32 %\(p)a_row, %M_offset\n"
    } else {
      ir += "  %\(p)a_row = mul i32 %M_offset, \(ldA)\n"
      ir += "  %\(p)a_off32 = add i32 %\(p)a_row, \(kStart)\n"
    }
    ir += "  %\(p)a_off = zext i32 %\(p)a_off32 to i64\n"
    ir += "  %\(p)a_byte = mul i64 %\(p)a_off, \(memA_size)\n"
    ir += "  %\(p)a_src_p = getelementptr i8, i8 addrspace(1)* %A, i64 %\(p)a_byte\n"

    // A tile dimensions (clamped)
    let aMTile = min(UInt32(M_group), M)

    // A dst/src strides
    let aDstStride = UInt32(leadingBlockDimensions.A) * memA_size
    let aSrcStride = ldA * memA_size

    // Tile dims for async copy (src tile = actual data, dst tile = full block for zero-padding)
    let (aSrcW, aSrcH, aDstW, aDstH): (String, String, UInt32, UInt32)
    if transposeState.A {
      aSrcW = "\(aMTile * memA_size)"  // width in bytes
      aSrcH = "\(kTile)"
      aDstW = UInt32(M_group) * memA_size
      aDstH = UInt32(K_group)
    } else {
      aSrcW = "\(kTile * memA_size)"
      aSrcH = "\(aMTile)"
      aDstW = UInt32(K_group) * memA_size
      aDstH = UInt32(M_group)
    }

    // A dst ptr = tg_base + 0 (A block starts at beginning of TG data area)
    ir += "  %\(p)a_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

    // Build tile vectors
    ir += "  %\(p)a_stile_w = insertelement <2 x i64> zeroinitializer, i64 \(aSrcW), i32 0\n"
    ir += "  %\(p)a_stile = insertelement <2 x i64> %\(p)a_stile_w, i64 \(aSrcH), i32 1\n"
    ir += "  %\(p)a_dtile_w = insertelement <2 x i64> zeroinitializer, i64 \(aDstW), i32 0\n"
    ir += "  %\(p)a_dtile = insertelement <2 x i64> %\(p)a_dtile_w, i64 \(aDstH), i32 1\n"

    // Event pointer
    ir += "  %\(p)ev0p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"

    // A async copy call
    ir += """
      %\(p)a_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
        i64 1, i64 1,
        i8 addrspace(3)* %\(p)a_dst_p, i64 \(aDstStride), i64 1, <2 x i64> %\(p)a_dtile,
        i8 addrspace(1)* %\(p)a_src_p, i64 \(aSrcStride), i64 1, <2 x i64> %\(p)a_stile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %\(p)a_ev, %event_t addrspace(3)** %\(p)ev0p

    """

    // B source offset: apply_offset(B, ldB, (N_offset, kStart), B_trans)
    if transposeState.B {
      ir += "  %\(p)b_row = mul i32 %N_offset, \(ldB)\n"
      ir += "  %\(p)b_off32 = add i32 %\(p)b_row, \(kStart)\n"
    } else {
      ir += "  %\(p)b_row = mul i32 \(kStart), \(ldB)\n"
      ir += "  %\(p)b_off32 = add i32 %\(p)b_row, %N_offset\n"
    }
    ir += "  %\(p)b_off = zext i32 %\(p)b_off32 to i64\n"
    ir += "  %\(p)b_byte = mul i64 %\(p)b_off, \(memB_size)\n"
    ir += "  %\(p)b_src_p = getelementptr i8, i8 addrspace(1)* %B, i64 %\(p)b_byte\n"

    let bNTile = min(UInt32(N_group), N)
    let bDstStride = UInt32(leadingBlockDimensions.B) * memB_size
    let bSrcStride = ldB * memB_size

    let (bSrcW, bSrcH, bDstW, bDstH): (String, String, UInt32, UInt32)
    if transposeState.B {
      bSrcW = "\(kTile * memB_size)"
      bSrcH = "\(bNTile)"
      bDstW = UInt32(K_group) * memB_size
      bDstH = UInt32(N_group)
    } else {
      bSrcW = "\(bNTile * memB_size)"
      bSrcH = "\(kTile)"
      bDstW = UInt32(N_group) * memB_size
      bDstH = UInt32(K_group)
    }

    // B dst ptr = tg_base + blockBytesA
    ir += "  %\(p)b_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 \(blockBytesA)\n"

    ir += "  %\(p)b_stile_w = insertelement <2 x i64> zeroinitializer, i64 \(bSrcW), i32 0\n"
    ir += "  %\(p)b_stile = insertelement <2 x i64> %\(p)b_stile_w, i64 \(bSrcH), i32 1\n"
    ir += "  %\(p)b_dtile_w = insertelement <2 x i64> zeroinitializer, i64 \(bDstW), i32 0\n"
    ir += "  %\(p)b_dtile = insertelement <2 x i64> %\(p)b_dtile_w, i64 \(bDstH), i32 1\n"

    ir += "  %\(p)ev1p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 1\n"

    ir += """
      %\(p)b_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
        i64 1, i64 1,
        i8 addrspace(3)* %\(p)b_dst_p, i64 \(bDstStride), i64 1, <2 x i64> %\(p)b_dtile,
        i8 addrspace(1)* %\(p)b_src_p, i64 \(bSrcStride), i64 1, <2 x i64> %\(p)b_stile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %\(p)b_ev, %event_t addrspace(3)** %\(p)ev1p

    """

    // Wait for both copies
    ir += "  call void @air.wait_simdgroup_events(i32 2, %event_t addrspace(3)** %\(p)ev0p)\n"

    return ir
  }

  // MARK: - Inline TG Multiply

  /// Result of inline TG multiply generation.
  struct InlineTGMultiplyResult {
    var ir: String
    var outputNames: [String]  // new C accumulator SSA names
  }

  /// Generate multiply-accumulate from TG data for one K-group iteration.
  /// C accumulators are passed in and out via SSA names (no TG save/restore).
  private func generateInlineTGMultiply(
    prefix p: String,
    kSteps: Int,
    regM: UInt16, regN: UInt16,
    blockBytesA: Int,
    cSramCount: Int,
    cInputNames: [String]
  ) -> InlineTGMultiplyResult {
    let regA = registerPrecisions.A
    let regB = registerPrecisions.B
    let regC = registerPrecisions.C
    let memA = memoryPrecisions.A
    let memB = memoryPrecisions.B
    let ldBlockA = leadingBlockDimensions.A
    let ldBlockB = leadingBlockDimensions.B

    var ir = ""
    ir += "  ; Multiply from TG (\(p))\n"
    ir += "  br i1 %out_of_bounds, label %\(p)skip_mul, label %\(p)do_mul\n\n"
    ir += "\(p)do_mul:\n"

    // TG block pointers (A at offset 0, B at offset blockBytesA)
    ir += "  %\(p)a_blk = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"
    ir += "  %\(p)b_blk = getelementptr i8, i8 addrspace(3)* %tg_base, i64 \(blockBytesA)\n"

    for k in 0..<kSteps {
      let kOff = k * 8

      // Load A tiles
      for m in stride(from: 0, to: Int(regM), by: 8) {
        let aElemType = irTypeName(memA)
        let aElemSize = memA.size

        if transposeState.A {
          for elem in 0..<2 {
            ir += "  %\(p)a_trow_\(k)_\(m)_\(elem) = add i32 %morton_x, \(kOff + elem)\n"
            ir += "  %\(p)a_tcol_\(k)_\(m)_\(elem) = add i32 %oig_y, \(m)\n"
            ir += "  %\(p)a_taddr_\(k)_\(m)_\(elem) = mul i32 %\(p)a_trow_\(k)_\(m)_\(elem), \(ldBlockA)\n"
            ir += "  %\(p)a_taddr2_\(k)_\(m)_\(elem) = add i32 %\(p)a_taddr_\(k)_\(m)_\(elem), %\(p)a_tcol_\(k)_\(m)_\(elem)\n"
            ir += "  %\(p)a_tbyte_\(k)_\(m)_\(elem) = mul i32 %\(p)a_taddr2_\(k)_\(m)_\(elem), \(aElemSize)\n"
            ir += "  %\(p)a_tbyte64_\(k)_\(m)_\(elem) = zext i32 %\(p)a_tbyte_\(k)_\(m)_\(elem) to i64\n"
            ir += "  %\(p)a_tptr_\(k)_\(m)_\(elem) = getelementptr i8, i8 addrspace(3)* %\(p)a_blk, i64 %\(p)a_tbyte64_\(k)_\(m)_\(elem)\n"
            ir += "  %\(p)a_ttyped_\(k)_\(m)_\(elem) = bitcast i8 addrspace(3)* %\(p)a_tptr_\(k)_\(m)_\(elem) to \(aElemType) addrspace(3)*\n"
            ir += "  %\(p)a_tload_\(k)_\(m)_\(elem) = load \(aElemType), \(aElemType) addrspace(3)* %\(p)a_ttyped_\(k)_\(m)_\(elem)\n"
          }
          if memA != regA {
            for elem in 0..<2 {
              ir += "  %\(p)a_text_\(k)_\(m)_\(elem) = fpext \(aElemType) %\(p)a_tload_\(k)_\(m)_\(elem) to \(irTypeName(regA))\n"
            }
            ir += "  %\(p)a_v2_\(k)_\(m) = insertelement <2 x \(irTypeName(regA))> undef, \(irTypeName(regA)) %\(p)a_text_\(k)_\(m)_0, i32 0\n"
            ir += "  %\(p)a_v2b_\(k)_\(m) = insertelement <2 x \(irTypeName(regA))> %\(p)a_v2_\(k)_\(m), \(irTypeName(regA)) %\(p)a_text_\(k)_\(m)_1, i32 1\n"
          } else {
            ir += "  %\(p)a_v2_\(k)_\(m) = insertelement <2 x \(aElemType)> undef, \(aElemType) %\(p)a_tload_\(k)_\(m)_0, i32 0\n"
            ir += "  %\(p)a_v2b_\(k)_\(m) = insertelement <2 x \(aElemType)> %\(p)a_v2_\(k)_\(m), \(aElemType) %\(p)a_tload_\(k)_\(m)_1, i32 1\n"
          }
          ir += irShuffleToVec64(result: "%\(p)a_sram_\(k)_\(m)", src: "%\(p)a_v2b_\(k)_\(m)", type: regA) + "\n"
        } else {
          ir += "  %\(p)a_row_\(k)_\(m) = add i32 %oig_y, \(m)\n"
          ir += "  %\(p)a_col_\(k)_\(m) = add i32 %morton_x, \(kOff)\n"
          ir += "  %\(p)a_addr_\(k)_\(m) = mul i32 %\(p)a_row_\(k)_\(m), \(ldBlockA)\n"
          ir += "  %\(p)a_addr2_\(k)_\(m) = add i32 %\(p)a_addr_\(k)_\(m), %\(p)a_col_\(k)_\(m)\n"
          ir += "  %\(p)a_byte_\(k)_\(m) = mul i32 %\(p)a_addr2_\(k)_\(m), \(aElemSize)\n"
          ir += "  %\(p)a_byte64_\(k)_\(m) = zext i32 %\(p)a_byte_\(k)_\(m) to i64\n"
          ir += "  %\(p)a_ptr_\(k)_\(m) = getelementptr i8, i8 addrspace(3)* %\(p)a_blk, i64 %\(p)a_byte64_\(k)_\(m)\n"
          ir += "  %\(p)a_typed_\(k)_\(m) = bitcast i8 addrspace(3)* %\(p)a_ptr_\(k)_\(m) to <2 x \(aElemType)> addrspace(3)*\n"
          ir += "  %\(p)a_load_\(k)_\(m) = load <2 x \(aElemType)>, <2 x \(aElemType)> addrspace(3)* %\(p)a_typed_\(k)_\(m), align \(aElemSize * 2)\n"

          if memA != regA {
            ir += "  %\(p)a_cvt0_\(k)_\(m) = extractelement <2 x \(aElemType)> %\(p)a_load_\(k)_\(m), i32 0\n"
            ir += "  %\(p)a_cvt1_\(k)_\(m) = extractelement <2 x \(aElemType)> %\(p)a_load_\(k)_\(m), i32 1\n"
            ir += "  %\(p)a_ext0_\(k)_\(m) = fpext \(aElemType) %\(p)a_cvt0_\(k)_\(m) to \(irTypeName(regA))\n"
            ir += "  %\(p)a_ext1_\(k)_\(m) = fpext \(aElemType) %\(p)a_cvt1_\(k)_\(m) to \(irTypeName(regA))\n"
            ir += "  %\(p)a_v2_\(k)_\(m) = insertelement <2 x \(irTypeName(regA))> undef, \(irTypeName(regA)) %\(p)a_ext0_\(k)_\(m), i32 0\n"
            ir += "  %\(p)a_v2b_\(k)_\(m) = insertelement <2 x \(irTypeName(regA))> %\(p)a_v2_\(k)_\(m), \(irTypeName(regA)) %\(p)a_ext1_\(k)_\(m), i32 1\n"
            ir += irShuffleToVec64(result: "%\(p)a_sram_\(k)_\(m)", src: "%\(p)a_v2b_\(k)_\(m)", type: regA) + "\n"
          } else {
            ir += irShuffleToVec64(result: "%\(p)a_sram_\(k)_\(m)", src: "%\(p)a_load_\(k)_\(m)", type: regA) + "\n"
          }
        }
      }

      // Load B tiles
      for n in stride(from: 0, to: Int(regN), by: 8) {
        let bElemType = irTypeName(memB)
        let bElemSize = memB.size

        if transposeState.B {
          for elem in 0..<2 {
            ir += "  %\(p)b_trow_\(k)_\(n)_\(elem) = add i32 %oig_x, \(n + elem)\n"
            ir += "  %\(p)b_tcol_\(k)_\(n)_\(elem) = add i32 %morton_y, \(kOff)\n"
            ir += "  %\(p)b_taddr_\(k)_\(n)_\(elem) = mul i32 %\(p)b_trow_\(k)_\(n)_\(elem), \(ldBlockB)\n"
            ir += "  %\(p)b_taddr2_\(k)_\(n)_\(elem) = add i32 %\(p)b_taddr_\(k)_\(n)_\(elem), %\(p)b_tcol_\(k)_\(n)_\(elem)\n"
            ir += "  %\(p)b_tbyte_\(k)_\(n)_\(elem) = mul i32 %\(p)b_taddr2_\(k)_\(n)_\(elem), \(bElemSize)\n"
            ir += "  %\(p)b_tbyte64_\(k)_\(n)_\(elem) = zext i32 %\(p)b_tbyte_\(k)_\(n)_\(elem) to i64\n"
            ir += "  %\(p)b_tptr_\(k)_\(n)_\(elem) = getelementptr i8, i8 addrspace(3)* %\(p)b_blk, i64 %\(p)b_tbyte64_\(k)_\(n)_\(elem)\n"
            ir += "  %\(p)b_ttyped_\(k)_\(n)_\(elem) = bitcast i8 addrspace(3)* %\(p)b_tptr_\(k)_\(n)_\(elem) to \(bElemType) addrspace(3)*\n"
            ir += "  %\(p)b_tload_\(k)_\(n)_\(elem) = load \(bElemType), \(bElemType) addrspace(3)* %\(p)b_ttyped_\(k)_\(n)_\(elem)\n"
          }
          if memB != regB {
            for elem in 0..<2 {
              ir += "  %\(p)b_text_\(k)_\(n)_\(elem) = fpext \(bElemType) %\(p)b_tload_\(k)_\(n)_\(elem) to \(irTypeName(regB))\n"
            }
            ir += "  %\(p)b_v2_\(k)_\(n) = insertelement <2 x \(irTypeName(regB))> undef, \(irTypeName(regB)) %\(p)b_text_\(k)_\(n)_0, i32 0\n"
            ir += "  %\(p)b_v2b_\(k)_\(n) = insertelement <2 x \(irTypeName(regB))> %\(p)b_v2_\(k)_\(n), \(irTypeName(regB)) %\(p)b_text_\(k)_\(n)_1, i32 1\n"
          } else {
            ir += "  %\(p)b_v2_\(k)_\(n) = insertelement <2 x \(bElemType)> undef, \(bElemType) %\(p)b_tload_\(k)_\(n)_0, i32 0\n"
            ir += "  %\(p)b_v2b_\(k)_\(n) = insertelement <2 x \(bElemType)> %\(p)b_v2_\(k)_\(n), \(bElemType) %\(p)b_tload_\(k)_\(n)_1, i32 1\n"
          }
          ir += irShuffleToVec64(result: "%\(p)b_sram_\(k)_\(n)", src: "%\(p)b_v2b_\(k)_\(n)", type: regB) + "\n"
        } else {
          ir += "  %\(p)b_row_\(k)_\(n) = add i32 %morton_y, \(kOff)\n"
          ir += "  %\(p)b_col_\(k)_\(n) = add i32 %oig_x, \(n)\n"
          ir += "  %\(p)b_addr_\(k)_\(n) = mul i32 %\(p)b_row_\(k)_\(n), \(ldBlockB)\n"
          ir += "  %\(p)b_addr2_\(k)_\(n) = add i32 %\(p)b_addr_\(k)_\(n), %\(p)b_col_\(k)_\(n)\n"
          ir += "  %\(p)b_byte_\(k)_\(n) = mul i32 %\(p)b_addr2_\(k)_\(n), \(bElemSize)\n"
          ir += "  %\(p)b_byte64_\(k)_\(n) = zext i32 %\(p)b_byte_\(k)_\(n) to i64\n"
          ir += "  %\(p)b_ptr_\(k)_\(n) = getelementptr i8, i8 addrspace(3)* %\(p)b_blk, i64 %\(p)b_byte64_\(k)_\(n)\n"
          ir += "  %\(p)b_typed_\(k)_\(n) = bitcast i8 addrspace(3)* %\(p)b_ptr_\(k)_\(n) to <2 x \(bElemType)> addrspace(3)*\n"
          ir += "  %\(p)b_load_\(k)_\(n) = load <2 x \(bElemType)>, <2 x \(bElemType)> addrspace(3)* %\(p)b_typed_\(k)_\(n), align \(bElemSize * 2)\n"

          if memB != regB {
            ir += "  %\(p)b_cvt0_\(k)_\(n) = extractelement <2 x \(bElemType)> %\(p)b_load_\(k)_\(n), i32 0\n"
            ir += "  %\(p)b_cvt1_\(k)_\(n) = extractelement <2 x \(bElemType)> %\(p)b_load_\(k)_\(n), i32 1\n"
            ir += "  %\(p)b_ext0_\(k)_\(n) = fpext \(bElemType) %\(p)b_cvt0_\(k)_\(n) to \(irTypeName(regB))\n"
            ir += "  %\(p)b_ext1_\(k)_\(n) = fpext \(bElemType) %\(p)b_cvt1_\(k)_\(n) to \(irTypeName(regB))\n"
            ir += "  %\(p)b_v2_\(k)_\(n) = insertelement <2 x \(irTypeName(regB))> undef, \(irTypeName(regB)) %\(p)b_ext0_\(k)_\(n), i32 0\n"
            ir += "  %\(p)b_v2b_\(k)_\(n) = insertelement <2 x \(irTypeName(regB))> %\(p)b_v2_\(k)_\(n), \(irTypeName(regB)) %\(p)b_ext1_\(k)_\(n), i32 1\n"
            ir += irShuffleToVec64(result: "%\(p)b_sram_\(k)_\(n)", src: "%\(p)b_v2b_\(k)_\(n)", type: regB) + "\n"
          } else {
            ir += irShuffleToVec64(result: "%\(p)b_sram_\(k)_\(n)", src: "%\(p)b_load_\(k)_\(n)", type: regB) + "\n"
          }
        }
      }

      // Multiply-accumulate: for each (m, n) pair
      for m in stride(from: 0, to: Int(regM), by: 8) {
        for n in stride(from: 0, to: Int(regN), by: 8) {
          let cIdx = (m / 8) * (Int(regN) / 8) + (n / 8)
          let cIn = (k == 0) ? cInputNames[cIdx] : "%\(p)c_ma_\(k-1)_\(m)_\(n)"
          let cOut = "%\(p)c_ma_\(k)_\(m)_\(n)"

          ir += irMultiplyAccumulateCall(
            result: cOut,
            A: ("%\(p)a_sram_\(k)_\(m)", regA),
            B: ("%\(p)b_sram_\(k)_\(n)", regB),
            C: (cIn, regC)
          ) + "\n"
        }
      }
    }

    ir += "  br label %\(p)after_mul\n\n"
    ir += "\(p)skip_mul:\n"
    ir += "  br label %\(p)after_mul\n\n"
    ir += "\(p)after_mul:\n"

    // Phi nodes to merge C values
    var outputNames = [String]()
    for m in stride(from: 0, to: Int(regM), by: 8) {
      for n in stride(from: 0, to: Int(regN), by: 8) {
        let cIdx = (m / 8) * (Int(regN) / 8) + (n / 8)
        let lastK = kSteps - 1
        let outName = "%\(p)c_out_\(cIdx)"
        ir += "  \(outName) = phi \(irVecType(regC)) [%\(p)c_ma_\(lastK)_\(m)_\(n), %\(p)do_mul], [\(cInputNames[cIdx]), %\(p)skip_mul]\n"
        outputNames.append(outName)
      }
    }
    ir += "\n"

    return InlineTGMultiplyResult(ir: ir, outputNames: outputNames)
  }

  // MARK: - Inline C Store

  /// Store C registers to TG block then inline async copy TG → device.
  /// Uses a single OOB branch instead of per-tile branches.
  private func generateStoreCToTGInline(
    desc: MonolithicDescriptor,
    ldC: UInt32,
    cSramCount: Int,
    M_group: UInt16, N_group: UInt16
  ) -> String {
    let memC = memoryPrecisions.C
    let regC = registerPrecisions.C
    let elemSize = memC.size

    var ir = "  ; Store C registers to TG block\n"

    // Single OOB branch for all tile stores
    ir += "  br i1 %out_of_bounds, label %c_skip_all_stores, label %c_do_all_stores\n\n"
    ir += "c_do_all_stores:\n"
    ir += "  %c_store_base = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

    for m in stride(from: 0, to: Int(registerM), by: 8) {
      for n in stride(from: 0, to: Int(registerN), by: 8) {
        let cIdx = (m / 8) * (Int(registerN) / 8) + (n / 8)

        ir += irShuffleFromVec64(result: "%c_st_\(cIdx)", src: "%c_final_\(cIdx)", type: regC) + "\n"

        if regC != memC {
          ir += "  %c_st_e0_\(cIdx) = extractelement <2 x \(irTypeName(regC))> %c_st_\(cIdx), i32 0\n"
          ir += "  %c_st_e1_\(cIdx) = extractelement <2 x \(irTypeName(regC))> %c_st_\(cIdx), i32 1\n"
          ir += "  %c_st_trunc0_\(cIdx) = fptrunc \(irTypeName(regC)) %c_st_e0_\(cIdx) to \(irTypeName(memC))\n"
          ir += "  %c_st_trunc1_\(cIdx) = fptrunc \(irTypeName(regC)) %c_st_e1_\(cIdx) to \(irTypeName(memC))\n"
          ir += "  %c_st_mem_\(cIdx)_a = insertelement <2 x \(irTypeName(memC))> undef, \(irTypeName(memC)) %c_st_trunc0_\(cIdx), i32 0\n"
          ir += "  %c_st_mem_\(cIdx) = insertelement <2 x \(irTypeName(memC))> %c_st_mem_\(cIdx)_a, \(irTypeName(memC)) %c_st_trunc1_\(cIdx), i32 1\n"
        }

        let storeVec = (regC != memC) ? "%c_st_mem_\(cIdx)" : "%c_st_\(cIdx)"
        let storeType = irTypeName(memC)

        ir += "  %c_tg_row_\(cIdx) = add i32 %oig_y, \(m)\n"
        ir += "  %c_tg_addr_\(cIdx) = mul i32 %c_tg_row_\(cIdx), \(leadingBlockDimensions.C)\n"
        ir += "  %c_tg_col_\(cIdx) = add i32 %oig_x, \(n)\n"
        ir += "  %c_tg_addr2_\(cIdx) = add i32 %c_tg_addr_\(cIdx), %c_tg_col_\(cIdx)\n"
        ir += "  %c_tg_byte_\(cIdx) = mul i32 %c_tg_addr2_\(cIdx), \(elemSize)\n"
        ir += "  %c_tg_byte64_\(cIdx) = zext i32 %c_tg_byte_\(cIdx) to i64\n"

        ir += "  %c_tg_ptr_\(cIdx) = getelementptr i8, i8 addrspace(3)* %c_store_base, i64 %c_tg_byte64_\(cIdx)\n"
        ir += "  %c_tg_typed_\(cIdx) = bitcast i8 addrspace(3)* %c_tg_ptr_\(cIdx) to <2 x \(storeType)> addrspace(3)*\n"
        ir += "  store <2 x \(storeType)> \(storeVec), <2 x \(storeType)> addrspace(3)* %c_tg_typed_\(cIdx)\n"
      }
    }

    ir += "  br label %c_skip_all_stores\n\n"
    ir += "c_skip_all_stores:\n"

    return ir
  }

  /// Direct store: C registers → device memory (fast path, no TG).
  /// Each thread stores its <2 x T> directly to device via GEP + store.
  /// Only valid for non-edge threadgroups where all M_group × N_group elements
  /// are in bounds and no edge shift occurred.
  private func generateDirectStoreCToDevice(
    desc: MonolithicDescriptor,
    ldC: UInt32,
    cSramCount: Int
  ) -> String {
    let memC = memoryPrecisions.C
    let regC = registerPrecisions.C
    let elemSize = UInt32(memC.size)

    var ir = "  ; Direct store C: registers → device\n"

    for m in stride(from: 0, to: Int(registerM), by: 8) {
      for n in stride(from: 0, to: Int(registerN), by: 8) {
        let cIdx = (m / 8) * (Int(registerN) / 8) + (n / 8)

        // Unshuffle from <64 x T> to <2 x T>
        ir += irShuffleFromVec64(result: "%cd_v2_\(cIdx)", src: "%c_final_\(cIdx)", type: regC) + "\n"

        // Precision conversion if needed
        if regC != memC {
          ir += "  %cd_e0_\(cIdx) = extractelement <2 x \(irTypeName(regC))> %cd_v2_\(cIdx), i32 0\n"
          ir += "  %cd_e1_\(cIdx) = extractelement <2 x \(irTypeName(regC))> %cd_v2_\(cIdx), i32 1\n"
          ir += "  %cd_trunc0_\(cIdx) = fptrunc \(irTypeName(regC)) %cd_e0_\(cIdx) to \(irTypeName(memC))\n"
          ir += "  %cd_trunc1_\(cIdx) = fptrunc \(irTypeName(regC)) %cd_e1_\(cIdx) to \(irTypeName(memC))\n"
          ir += "  %cd_mem_\(cIdx)_a = insertelement <2 x \(irTypeName(memC))> undef, \(irTypeName(memC)) %cd_trunc0_\(cIdx), i32 0\n"
          ir += "  %cd_mem_\(cIdx) = insertelement <2 x \(irTypeName(memC))> %cd_mem_\(cIdx)_a, \(irTypeName(memC)) %cd_trunc1_\(cIdx), i32 1\n"
        }

        let storeVec = (regC != memC) ? "%cd_mem_\(cIdx)" : "%cd_v2_\(cIdx)"
        let storeType = irTypeName(memC)

        // Device address: C + (M_offset + oig_y + m) * ldC + (N_offset + oig_x + n)
        ir += "  %cd_row_\(cIdx) = add i32 %M_offset, %oig_y\n"
        ir += "  %cd_row2_\(cIdx) = add i32 %cd_row_\(cIdx), \(m)\n"
        ir += "  %cd_col_\(cIdx) = add i32 %N_offset, %oig_x\n"
        ir += "  %cd_col2_\(cIdx) = add i32 %cd_col_\(cIdx), \(n)\n"
        ir += "  %cd_off_\(cIdx) = mul i32 %cd_row2_\(cIdx), \(ldC)\n"
        ir += "  %cd_off2_\(cIdx) = add i32 %cd_off_\(cIdx), %cd_col2_\(cIdx)\n"
        ir += "  %cd_off64_\(cIdx) = zext i32 %cd_off2_\(cIdx) to i64\n"
        ir += "  %cd_byte_\(cIdx) = mul i64 %cd_off64_\(cIdx), \(elemSize)\n"
        ir += "  %cd_ptr_\(cIdx) = getelementptr i8, i8 addrspace(1)* %C, i64 %cd_byte_\(cIdx)\n"
        ir += "  %cd_typed_\(cIdx) = bitcast i8 addrspace(1)* %cd_ptr_\(cIdx) to <2 x \(storeType)> addrspace(1)*\n"
        ir += "  store <2 x \(storeType)> \(storeVec), <2 x \(storeType)> addrspace(1)* %cd_typed_\(cIdx)\n"
      }
    }

    return ir
  }

  /// Inline async store: C from TG → device (no cmd protocol).
  private func generateInlineStoreC(
    desc: MonolithicDescriptor,
    ldC: UInt32,
    M_group: UInt16, N_group: UInt16
  ) -> String {
    let elemSize = UInt32(memoryPrecisions.C.size)

    var ir = "  ; Inline async store C: TG → device\n"

    // Use UNSHIFTED offsets for device address (same as old code)
    ir += "  %c_dev_m_unshift = mul i32 %gid_y, \(M_group)\n"
    ir += "  %c_dev_n_unshift = mul i32 %gid_x, \(N_group)\n"
    ir += "  %c_dev_row = mul i32 %c_dev_m_unshift, \(ldC)\n"
    ir += "  %c_dev_off32 = add i32 %c_dev_row, %c_dev_n_unshift\n"
    ir += "  %c_dev_off = zext i32 %c_dev_off32 to i64\n"
    ir += "  %c_dev_byte = mul i64 %c_dev_off, \(elemSize)\n"

    // Tile dimensions
    ir += "  %c_tile_n = sub i32 \(desc.N), %c_dev_n_unshift\n"
    ir += "  %c_tile_n_cmp = icmp ult i32 %c_tile_n, \(N_group)\n"
    ir += "  %c_tile_w32 = select i1 %c_tile_n_cmp, i32 %c_tile_n, i32 \(N_group)\n"
    ir += "  %c_tile_w_bytes32 = mul i32 %c_tile_w32, \(elemSize)\n"
    ir += "  %c_tile_w_bytes = zext i32 %c_tile_w_bytes32 to i64\n"

    ir += "  %c_tile_m = sub i32 \(desc.M), %c_dev_m_unshift\n"
    ir += "  %c_tile_m_cmp = icmp ult i32 %c_tile_m, \(M_group)\n"
    ir += "  %c_tile_h32 = select i1 %c_tile_m_cmp, i32 %c_tile_m, i32 \(M_group)\n"
    ir += "  %c_tile_h = zext i32 %c_tile_h32 to i64\n"

    // TG source pointer with edge shift
    let M_edge = desc.M - (desc.M % UInt32(M_group))
    let N_edge = desc.N - (desc.N % UInt32(N_group))
    let M_remainder_store = (desc.M % UInt32(registerM) == 0) ? registerM : UInt16(desc.M % UInt32(registerM))
    let N_remainder_store = (desc.N % UInt32(registerN) == 0) ? registerN : UInt16(desc.N % UInt32(registerN))
    let M_shift_val: UInt16 = (desc.M < UInt32(M_group)) ? 0 : registerM - M_remainder_store
    let N_shift_val: UInt16 = (desc.N < UInt32(N_group)) ? 0 : registerN - N_remainder_store

    if M_shift_val != 0 || N_shift_val != 0 {
      ir += "  ; TG block shift for edge correction\n"
      if M_shift_val != 0 {
        ir += "  %cs_m_edge_cmp = icmp uge i32 %c_dev_m_unshift, \(M_edge)\n"
        ir += "  %cs_m_shift = select i1 %cs_m_edge_cmp, i32 \(M_shift_val), i32 0\n"
      } else {
        ir += "  %cs_m_shift = add i32 0, 0\n"
      }
      if N_shift_val != 0 {
        ir += "  %cs_n_edge_cmp = icmp uge i32 %c_dev_n_unshift, \(N_edge)\n"
        ir += "  %cs_n_shift = select i1 %cs_n_edge_cmp, i32 \(N_shift_val), i32 0\n"
      } else {
        ir += "  %cs_n_shift = add i32 0, 0\n"
      }
      ir += "  %cs_tg_shift_row = mul i32 %cs_m_shift, \(leadingBlockDimensions.C)\n"
      ir += "  %cs_tg_shift_off = add i32 %cs_tg_shift_row, %cs_n_shift\n"
      ir += "  %cs_tg_shift_bytes = mul i32 %cs_tg_shift_off, \(elemSize)\n"
      ir += "  %cs_tg_off32 = add i32 0, %cs_tg_shift_bytes\n"
      ir += "  %cs_tg_off = zext i32 %cs_tg_off32 to i64\n"
    } else {
      ir += "  %cs_tg_off = add i64 0, 0\n"
    }

    let dstStride = ldC * elemSize
    let srcStride = UInt32(leadingBlockDimensions.C) * elemSize

    ir += "  %cs_dst_p = getelementptr i8, i8 addrspace(1)* %C, i64 %c_dev_byte\n"
    ir += "  %cs_src_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %cs_tg_off\n"

    ir += "  %cs_tile_w = insertelement <2 x i64> zeroinitializer, i64 %c_tile_w_bytes, i32 0\n"
    ir += "  %cs_tile = insertelement <2 x i64> %cs_tile_w, i64 %c_tile_h, i32 1\n"

    ir += "  %cs_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"

    ir += """
      %cs_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
        i64 1, i64 1,
        i8 addrspace(1)* %cs_dst_p, i64 \(dstStride), i64 1, <2 x i64> %cs_tile,
        i8 addrspace(3)* %cs_src_p, i64 \(srcStride), i64 1, <2 x i64> %cs_tile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %cs_ev, %event_t addrspace(3)** %cs_evp
      call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %cs_evp)
      call void @air.wg.barrier(i32 2, i32 1)
      br label %exit

    """

    return ir
  }

  // MARK: - Inline Load C (for loadPreviousC)

  /// Inline async load of previous C from device → TG, then load into registers.
  private func generateInlineLoadC(
    desc: MonolithicDescriptor,
    ldC: UInt32,
    M_group: UInt16, N_group: UInt16,
    cSramCount: Int
  ) -> String {
    let memC = memoryPrecisions.C
    let regC = registerPrecisions.C
    let elemSize = UInt32(memC.size)

    var ir = "  ; === Load previous C ===\n"

    // Device offset for C
    ir += "  %lpc_dev_row = mul i32 %M_offset, \(ldC)\n"
    ir += "  %lpc_dev_off32 = add i32 %lpc_dev_row, %N_offset\n"
    ir += "  %lpc_dev_off = zext i32 %lpc_dev_off32 to i64\n"
    ir += "  %lpc_dev_byte = mul i64 %lpc_dev_off, \(elemSize)\n"

    // Tile dimensions
    ir += "  %lpc_n_rem = sub i32 \(desc.N), %N_offset\n"
    ir += "  %lpc_n_cmp = icmp ult i32 %lpc_n_rem, \(N_group)\n"
    ir += "  %lpc_tw32 = select i1 %lpc_n_cmp, i32 %lpc_n_rem, i32 \(N_group)\n"
    ir += "  %lpc_tw_bytes32 = mul i32 %lpc_tw32, \(elemSize)\n"
    ir += "  %lpc_tw = zext i32 %lpc_tw_bytes32 to i64\n"

    ir += "  %lpc_m_rem = sub i32 \(desc.M), %M_offset\n"
    ir += "  %lpc_m_cmp = icmp ult i32 %lpc_m_rem, \(M_group)\n"
    ir += "  %lpc_th32 = select i1 %lpc_m_cmp, i32 %lpc_m_rem, i32 \(M_group)\n"
    ir += "  %lpc_th = zext i32 %lpc_th32 to i64\n"

    let dstStride = UInt32(leadingBlockDimensions.C) * elemSize
    let srcStride = ldC * elemSize

    ir += "  %lpc_src_p = getelementptr i8, i8 addrspace(1)* %C, i64 %lpc_dev_byte\n"
    ir += "  %lpc_dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

    ir += "  %lpc_tile_w = insertelement <2 x i64> zeroinitializer, i64 %lpc_tw, i32 0\n"
    ir += "  %lpc_tile = insertelement <2 x i64> %lpc_tile_w, i64 %lpc_th, i32 1\n"

    ir += "  %lpc_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"

    // Gate async copy to sidx == 0
    ir += "  br i1 %is_sidx0, label %lpc_do_copy, label %lpc_skip_copy\n\n"
    ir += "lpc_do_copy:\n"
    ir += """
      %lpc_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
        i64 1, i64 1,
        i8 addrspace(3)* %lpc_dst_p, i64 \(dstStride), i64 1, <2 x i64> %lpc_tile,
        i8 addrspace(1)* %lpc_src_p, i64 \(srcStride), i64 1, <2 x i64> %lpc_tile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %lpc_ev, %event_t addrspace(3)** %lpc_evp
      call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %lpc_evp)
      br label %lpc_after_copy

    """
    ir += "lpc_skip_copy:\n"
    ir += "  br label %lpc_after_copy\n\n"
    ir += "lpc_after_copy:\n"
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    // Load C from TG into accumulators
    ir += "  %lc_base = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

    for m in stride(from: 0, to: Int(registerM), by: 8) {
      for n in stride(from: 0, to: Int(registerN), by: 8) {
        let cIdx = (m / 8) * (Int(registerN) / 8) + (n / 8)

        ir += "  %lc_row_\(cIdx) = add i32 %oig_y, \(m)\n"
        ir += "  %lc_addr_\(cIdx) = mul i32 %lc_row_\(cIdx), \(leadingBlockDimensions.C)\n"
        ir += "  %lc_col_\(cIdx) = add i32 %oig_x, \(n)\n"
        ir += "  %lc_addr2_\(cIdx) = add i32 %lc_addr_\(cIdx), %lc_col_\(cIdx)\n"
        ir += "  %lc_byte_\(cIdx) = mul i32 %lc_addr2_\(cIdx), \(memC.size)\n"
        ir += "  %lc_byte64_\(cIdx) = zext i32 %lc_byte_\(cIdx) to i64\n"

        ir += "  br i1 %out_of_bounds, label %lc_skip_\(cIdx), label %lc_do_\(cIdx)\n\n"

        ir += "lc_do_\(cIdx):\n"
        ir += "  %lc_ptr_\(cIdx) = getelementptr i8, i8 addrspace(3)* %lc_base, i64 %lc_byte64_\(cIdx)\n"
        ir += "  %lc_typed_\(cIdx) = bitcast i8 addrspace(3)* %lc_ptr_\(cIdx) to <2 x \(irTypeName(memC))> addrspace(3)*\n"
        ir += "  %lc_load_\(cIdx) = load <2 x \(irTypeName(memC))>, <2 x \(irTypeName(memC))> addrspace(3)* %lc_typed_\(cIdx), align \(memC.size * 2)\n"

        if memC != regC {
          ir += "  %lc_e0_\(cIdx) = extractelement <2 x \(irTypeName(memC))> %lc_load_\(cIdx), i32 0\n"
          ir += "  %lc_e1_\(cIdx) = extractelement <2 x \(irTypeName(memC))> %lc_load_\(cIdx), i32 1\n"
          ir += "  %lc_ext0_\(cIdx) = fpext \(irTypeName(memC)) %lc_e0_\(cIdx) to \(irTypeName(regC))\n"
          ir += "  %lc_ext1_\(cIdx) = fpext \(irTypeName(memC)) %lc_e1_\(cIdx) to \(irTypeName(regC))\n"
          ir += "  %lc_v2_\(cIdx) = insertelement <2 x \(irTypeName(regC))> undef, \(irTypeName(regC)) %lc_ext0_\(cIdx), i32 0\n"
          ir += "  %lc_v2b_\(cIdx) = insertelement <2 x \(irTypeName(regC))> %lc_v2_\(cIdx), \(irTypeName(regC)) %lc_ext1_\(cIdx), i32 1\n"
          ir += irShuffleToVec64(result: "%lc_loaded_\(cIdx)", src: "%lc_v2b_\(cIdx)", type: regC) + "\n"
        } else {
          ir += irShuffleToVec64(result: "%lc_loaded_\(cIdx)", src: "%lc_load_\(cIdx)", type: regC) + "\n"
        }
        ir += "  br label %lc_skip_\(cIdx)\n\n"

        // The OOB branch comes from the block containing `br i1 %out_of_bounds`:
        // for cIdx==0, that's "lpc_after_copy"; for cIdx>0, it's "lc_skip_(cIdx-1)"
        ir += "lc_skip_\(cIdx):\n"
        let oobSource = (cIdx == 0) ? "lpc_after_copy" : "lc_skip_\(cIdx - 1)"
        ir += "  %c_start_\(cIdx) = phi \(irVecType(regC)) [%lc_loaded_\(cIdx), %lc_do_\(cIdx)], [%c_init_\(cIdx), %\(oobSource)]\n"
      }
    }

    // Barrier before A/B copies reuse TG
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    return ir
  }
}
