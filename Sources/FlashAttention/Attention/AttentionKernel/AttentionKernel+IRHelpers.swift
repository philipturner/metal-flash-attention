//
//  AttentionKernel+IRHelpers.swift
//  FlashAttention
//
//  Shared IR generation helpers for attention monolithic kernels.
//  These generate the building blocks: async copy, outer product,
//  softmax, accumulate, cache load/store.
//

extension AttentionKernel {

  // MARK: - Async Copy Device → TG (2D)

  /// Generate an inline async copy START (no wait). Gated to sidx == 0.
  /// Returns IR that ends after the copy is initiated but NOT waited on.
  /// Use generateAsyncCopyWait to wait + barrier later.
  func generateAsyncCopyStart(
    prefix p: String,
    buffer: String,
    operand: AttentionOperand,
    dOuter: String,
    seqOffset: String,
    seqDim: UInt32,
    blockSeq: UInt16,
    blockHead: UInt16,
    D: UInt32,
    leadingDim: UInt32,
    leadingBlockDim: UInt32,
    memPrec: GEMMOperandPrecision,
    transposed: Bool,
    tgOffset: String = "0",
    eventSlot: Int = 0
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    var ir = ""

    ir += "  ; Async copy START \(operand) device→TG (\(p))\n"
    ir += "  br i1 %is_sidx0, label %\(p)do_copy, label %\(p)skip_copy\n\n"
    ir += "\(p)do_copy:\n"

    // Source offset calculation
    if transposed {
      ir += "  %\(p)src_row = mul i32 \(dOuter), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(seqOffset)\n"
    } else {
      ir += "  %\(p)src_row = mul i32 \(seqOffset), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(dOuter)\n"
    }
    ir += "  %\(p)src_off = zext i32 %\(p)src_off32 to i64\n"
    ir += "  %\(p)src_byte = mul i64 %\(p)src_off, \(elemSize)\n"
    ir += "  %\(p)src_p = getelementptr i8, i8 addrspace(1)* \(buffer), i64 %\(p)src_byte\n"

    ir += "  %\(p)d_rem_32 = sub i32 \(D), \(dOuter)\n"
    ir += "  %\(p)d_cmp = icmp ult i32 %\(p)d_rem_32, \(blockHead)\n"
    ir += "  %\(p)d_src = select i1 %\(p)d_cmp, i32 %\(p)d_rem_32, i32 \(blockHead)\n"
    // Guard: if seqOffset >= seqDim, zero-height copy (no-op).
    ir += "  %\(p)seq_oob = icmp uge i32 \(seqOffset), \(seqDim)\n"
    ir += "  %\(p)seq_rem_raw = sub i32 \(seqDim), \(seqOffset)\n"
    ir += "  %\(p)seq_rem_32 = select i1 %\(p)seq_oob, i32 0, i32 %\(p)seq_rem_raw\n"
    ir += "  %\(p)seq_cmp = icmp ult i32 %\(p)seq_rem_32, \(blockSeq)\n"
    ir += "  %\(p)seq_src = select i1 %\(p)seq_cmp, i32 %\(p)seq_rem_32, i32 \(blockSeq)\n"

    let dstStride = leadingBlockDim * elemSize
    let srcStride = leadingDim * elemSize

    let (srcW, srcH, dstW, dstH): (String, String, UInt32, UInt32)
    if transposed {
      srcW = "%\(p)seq_src_bytes"
      srcH = "%\(p)d_src_ext"
      ir += "  %\(p)seq_src_bytes32 = mul i32 %\(p)seq_src, \(elemSize)\n"
      ir += "  %\(p)seq_src_bytes = zext i32 %\(p)seq_src_bytes32 to i64\n"
      ir += "  %\(p)d_src_ext = zext i32 %\(p)d_src to i64\n"
      dstW = UInt32(blockSeq) * elemSize
      dstH = UInt32(blockHead)
    } else {
      srcW = "%\(p)d_src_bytes"
      srcH = "%\(p)seq_src_ext"
      ir += "  %\(p)d_src_bytes32 = mul i32 %\(p)d_src, \(elemSize)\n"
      ir += "  %\(p)d_src_bytes = zext i32 %\(p)d_src_bytes32 to i64\n"
      ir += "  %\(p)seq_src_ext = zext i32 %\(p)seq_src to i64\n"
      dstW = UInt32(blockHead) * elemSize
      dstH = UInt32(blockSeq)
    }

    // TG destination pointer
    ir += "  %\(p)dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 \(tgOffset)\n"

    // Tile vectors
    ir += "  %\(p)stile_w = insertelement <2 x i64> zeroinitializer, i64 \(srcW), i32 0\n"
    ir += "  %\(p)stile = insertelement <2 x i64> %\(p)stile_w, i64 \(srcH), i32 1\n"
    ir += "  %\(p)dtile_w = insertelement <2 x i64> zeroinitializer, i64 \(dstW), i32 0\n"
    ir += "  %\(p)dtile = insertelement <2 x i64> %\(p)dtile_w, i64 \(dstH), i32 1\n"

    // Event pointer (use specified slot)
    ir += "  %\(p)evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 \(eventSlot)\n"

    // Async copy call — NO wait
    ir += """
      %\(p)ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
        i64 1, i64 1,
        i8 addrspace(3)* %\(p)dst_p, i64 \(dstStride), i64 1, <2 x i64> %\(p)dtile,
        i8 addrspace(1)* %\(p)src_p, i64 \(srcStride), i64 1, <2 x i64> %\(p)stile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %\(p)ev, %event_t addrspace(3)** %\(p)evp

    """

    ir += "  br label %\(p)after_copy\n\n"
    ir += "\(p)skip_copy:\n"
    ir += "  br label %\(p)after_copy\n\n"
    ir += "\(p)after_copy:\n"
    // Barrier so all threads sync after sidx0 initiated the copy
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    return ir
  }

  /// Wait for a previously started async copy + barrier for TG visibility.
  func generateAsyncCopyWait(
    prefix p: String,
    eventSlot: Int = 0
  ) -> String {
    var ir = ""
    ir += "  ; Wait for async copy (\(p))\n"
    ir += "  br i1 %is_sidx0, label %\(p)do_wait, label %\(p)skip_wait\n\n"
    ir += "\(p)do_wait:\n"
    ir += "  %\(p)evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 \(eventSlot)\n"
    ir += "  call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %\(p)evp)\n"
    ir += "  br label %\(p)after_wait\n\n"
    ir += "\(p)skip_wait:\n"
    ir += "  br label %\(p)after_wait\n\n"
    ir += "\(p)after_wait:\n"
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"
    return ir
  }

  /// Generate an inline async copy from device to threadgroup.
  /// Gated to sidx == 0. Returns IR that ends after barrier.
  func generateAsyncCopyDeviceToTG(
    prefix p: String,
    buffer: String,           // SSA name of device buffer (e.g. "%K")
    operand: AttentionOperand,
    dOuter: String,           // SSA name or literal for d_outer offset
    seqOffset: String,        // SSA name or literal for sequence offset
    seqDim: UInt32,           // total sequence dimension
    blockSeq: UInt16,         // block size along sequence dim
    blockHead: UInt16,        // block size along head dim (register size for this iteration)
    D: UInt32,                // total head dimension
    leadingDim: UInt32,
    leadingBlockDim: UInt32,
    memPrec: GEMMOperandPrecision,
    transposed: Bool,
    tgOffset: String = "0"    // byte offset into TG buffer
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    var ir = ""

    ir += "  ; Async copy \(operand) device→TG (\(p))\n"
    ir += "  br i1 %is_sidx0, label %\(p)do_copy, label %\(p)skip_copy\n\n"
    ir += "\(p)do_copy:\n"

    // Source offset calculation
    if transposed {
      // offset = (dOuter) * leadingDim + seqOffset
      ir += "  %\(p)src_row = mul i32 \(dOuter), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(seqOffset)\n"
    } else {
      // offset = seqOffset * leadingDim + dOuter
      ir += "  %\(p)src_row = mul i32 \(seqOffset), \(leadingDim)\n"
      ir += "  %\(p)src_off32 = add i32 %\(p)src_row, \(dOuter)\n"
    }
    ir += "  %\(p)src_off = zext i32 %\(p)src_off32 to i64\n"
    ir += "  %\(p)src_byte = mul i64 %\(p)src_off, \(elemSize)\n"
    ir += "  %\(p)src_p = getelementptr i8, i8 addrspace(1)* \(buffer), i64 %\(p)src_byte\n"

    // D source tile = min(blockHead, D - dOuter)
    // Seq source tile = min(blockSeq, seqDim - seqOffset)
    ir += "  %\(p)d_rem_32 = sub i32 \(D), \(dOuter)\n"
    ir += "  %\(p)d_cmp = icmp ult i32 %\(p)d_rem_32, \(blockHead)\n"
    ir += "  %\(p)d_src = select i1 %\(p)d_cmp, i32 %\(p)d_rem_32, i32 \(blockHead)\n"
    ir += "  %\(p)seq_rem_32 = sub i32 \(seqDim), \(seqOffset)\n"
    ir += "  %\(p)seq_cmp = icmp ult i32 %\(p)seq_rem_32, \(blockSeq)\n"
    ir += "  %\(p)seq_src = select i1 %\(p)seq_cmp, i32 %\(p)seq_rem_32, i32 \(blockSeq)\n"

    let dstStride = leadingBlockDim * elemSize
    let srcStride = leadingDim * elemSize

    // Build tile vectors. The tile is (width_bytes, height).
    // For non-transposed: width = D elements, height = seq elements
    // For transposed: width = seq elements, height = D elements
    let (srcW, srcH, dstW, dstH): (String, String, UInt32, UInt32)
    if transposed {
      srcW = "%\(p)seq_src_bytes"
      srcH = "%\(p)d_src_ext"
      ir += "  %\(p)seq_src_bytes32 = mul i32 %\(p)seq_src, \(elemSize)\n"
      ir += "  %\(p)seq_src_bytes = zext i32 %\(p)seq_src_bytes32 to i64\n"
      ir += "  %\(p)d_src_ext = zext i32 %\(p)d_src to i64\n"
      dstW = UInt32(blockSeq) * elemSize
      dstH = UInt32(blockHead)
    } else {
      srcW = "%\(p)d_src_bytes"
      srcH = "%\(p)seq_src_ext"
      ir += "  %\(p)d_src_bytes32 = mul i32 %\(p)d_src, \(elemSize)\n"
      ir += "  %\(p)d_src_bytes = zext i32 %\(p)d_src_bytes32 to i64\n"
      ir += "  %\(p)seq_src_ext = zext i32 %\(p)seq_src to i64\n"
      dstW = UInt32(blockHead) * elemSize
      dstH = UInt32(blockSeq)
    }

    // TG destination pointer
    ir += "  %\(p)dst_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 \(tgOffset)\n"

    // Tile vectors
    ir += "  %\(p)stile_w = insertelement <2 x i64> zeroinitializer, i64 \(srcW), i32 0\n"
    ir += "  %\(p)stile = insertelement <2 x i64> %\(p)stile_w, i64 \(srcH), i32 1\n"
    ir += "  %\(p)dtile_w = insertelement <2 x i64> zeroinitializer, i64 \(dstW), i32 0\n"
    ir += "  %\(p)dtile = insertelement <2 x i64> %\(p)dtile_w, i64 \(dstH), i32 1\n"

    // Event pointer
    ir += "  %\(p)evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"

    // Async copy call
    ir += """
      %\(p)ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
        i64 1, i64 1,
        i8 addrspace(3)* %\(p)dst_p, i64 \(dstStride), i64 1, <2 x i64> %\(p)dtile,
        i8 addrspace(1)* %\(p)src_p, i64 \(srcStride), i64 1, <2 x i64> %\(p)stile,
        <2 x i64> zeroinitializer, i32 0
      )
      store %event_t addrspace(3)* %\(p)ev, %event_t addrspace(3)** %\(p)evp
      call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %\(p)evp)

    """

    ir += "  br label %\(p)after_copy\n\n"
    ir += "\(p)skip_copy:\n"
    ir += "  br label %\(p)after_copy\n\n"
    ir += "\(p)after_copy:\n"
    ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

    return ir
  }

  // MARK: - Load from TG into Register (SIMD matrix)

  /// Generate IR to load a <2 x T> from threadgroup memory, convert precision,
  /// and expand to <64 x T> via insertelement.
  func generateTGLoad(
    prefix p: String,
    tgOffset: String,         // byte offset into TG (literal or SSA)
    rowOffset: String,        // row offset (SSA)
    colOffset: String,        // column offset (SSA)
    leadingBlockDim: UInt32,
    memPrec: GEMMOperandPrecision,
    regPrec: GEMMOperandPrecision,
    transposed: Bool
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    let memType = irTypeName(memPrec)
    let regType = irTypeName(regPrec)
    var ir = ""

    // Address: tg_base + tgOffset + (row * leadingBlockDim + col) * elemSize
    // For non-transposed: row = seqOffset, col = headOffset (standard matrix layout)
    // The caller sets rowOffset/colOffset appropriately for the transpose state.
    let tgOffsetI64 = (tgOffset == "0") ? nil : tgOffset
    if transposed {
      // Element address: each thread reads 2 elements (via morton_x offsets)
      // For transposed, row=morton_x+d, col=oig_y or similar
      for elem in 0..<2 {
        ir += "  %\(p)r_\(elem) = add i32 \(rowOffset), \(elem)\n"
        ir += "  %\(p)addr_\(elem) = mul i32 %\(p)r_\(elem), \(leadingBlockDim)\n"
        ir += "  %\(p)addr2_\(elem) = add i32 %\(p)addr_\(elem), \(colOffset)\n"
        ir += "  %\(p)byte_\(elem) = mul i32 %\(p)addr2_\(elem), \(elemSize)\n"
        ir += "  %\(p)byte64_\(elem) = zext i32 %\(p)byte_\(elem) to i64\n"
        if let off = tgOffsetI64 {
          ir += "  %\(p)byte64o_\(elem) = add i64 %\(p)byte64_\(elem), \(off)\n"
          ir += "  %\(p)ptr_\(elem) = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(p)byte64o_\(elem)\n"
        } else {
          ir += "  %\(p)ptr_\(elem) = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(p)byte64_\(elem)\n"
        }
        ir += "  %\(p)typed_\(elem) = bitcast i8 addrspace(3)* %\(p)ptr_\(elem) to \(memType) addrspace(3)*\n"
        ir += "  %\(p)load_\(elem) = load \(memType), \(memType) addrspace(3)* %\(p)typed_\(elem)\n"
      }
      if memPrec != regPrec {
        for elem in 0..<2 {
          ir += "  %\(p)ext_\(elem) = fpext \(memType) %\(p)load_\(elem) to \(regType)\n"
        }
        ir += "  %\(p)v2_a = insertelement <2 x \(regType)> undef, \(regType) %\(p)ext_0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(regType)> %\(p)v2_a, \(regType) %\(p)ext_1, i32 1\n"
      } else {
        ir += "  %\(p)v2_a = insertelement <2 x \(memType)> undef, \(memType) %\(p)load_0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(memType)> %\(p)v2_a, \(memType) %\(p)load_1, i32 1\n"
      }
    } else {
      // Non-transposed: load contiguous <2 x T>
      ir += "  %\(p)addr = mul i32 \(rowOffset), \(leadingBlockDim)\n"
      ir += "  %\(p)addr2 = add i32 %\(p)addr, \(colOffset)\n"
      ir += "  %\(p)byte = mul i32 %\(p)addr2, \(elemSize)\n"
      ir += "  %\(p)byte64 = zext i32 %\(p)byte to i64\n"
      if let off = tgOffsetI64 {
        ir += "  %\(p)byte64o = add i64 %\(p)byte64, \(off)\n"
        ir += "  %\(p)ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(p)byte64o\n"
      } else {
        ir += "  %\(p)ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(p)byte64\n"
      }
      ir += "  %\(p)typed = bitcast i8 addrspace(3)* %\(p)ptr to <2 x \(memType)> addrspace(3)*\n"
      ir += "  %\(p)load = load <2 x \(memType)>, <2 x \(memType)> addrspace(3)* %\(p)typed, align \(elemSize * 2)\n"

      if memPrec != regPrec {
        ir += "  %\(p)e0 = extractelement <2 x \(memType)> %\(p)load, i32 0\n"
        ir += "  %\(p)e1 = extractelement <2 x \(memType)> %\(p)load, i32 1\n"
        ir += "  %\(p)ext0 = fpext \(memType) %\(p)e0 to \(regType)\n"
        ir += "  %\(p)ext1 = fpext \(memType) %\(p)e1 to \(regType)\n"
        ir += "  %\(p)v2_a = insertelement <2 x \(regType)> undef, \(regType) %\(p)ext0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(regType)> %\(p)v2_a, \(regType) %\(p)ext1, i32 1\n"
      } else {
        ir += "  %\(p)v2 = bitcast <2 x \(memType)> %\(p)load to <2 x \(memType)>\n"
      }
    }

    // Expand <2 x T> to <64 x T>
    ir += irShuffleToVec64(result: "%\(p)sram", src: "%\(p)v2", type: regPrec) + "\n"

    return ir
  }

  // MARK: - Load from Device into Register (SIMD matrix)

  /// Generate IR to load a <2 x T> directly from device memory (addrspace 1),
  /// convert precision, and expand to <64 x T> via insertelement.
  /// bufferName: SSA name of the device pointer (e.g., "%Q")
  /// seqOffset: SSA name or literal for the sequence offset of this thread
  /// headOffset: SSA name or literal for the head offset of this thread
  /// leadingDim: the leading dimension of the buffer
  func generateDeviceLoad(
    prefix p: String,
    bufferName: String,
    seqOffset: String,         // row in non-transposed (e.g., "%par_group_off + oig_y")
    headOffset: String,        // col in non-transposed (e.g., "d_outer + morton_x * 8")
    leadingDim: UInt32,
    memPrec: GEMMOperandPrecision,
    regPrec: GEMMOperandPrecision,
    transposed: Bool
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    let memType = irTypeName(memPrec)
    let regType = irTypeName(regPrec)
    var ir = ""

    if transposed {
      // Transposed: each thread reads 2 elements at (row+0, col) and (row+1, col)
      for elem in 0..<2 {
        ir += "  %\(p)r_\(elem) = add i32 \(headOffset), \(elem)\n"
        ir += "  %\(p)addr_\(elem) = mul i32 %\(p)r_\(elem), \(leadingDim)\n"
        ir += "  %\(p)addr2_\(elem) = add i32 %\(p)addr_\(elem), \(seqOffset)\n"
        ir += "  %\(p)byte_\(elem) = mul i32 %\(p)addr2_\(elem), \(elemSize)\n"
        ir += "  %\(p)byte64_\(elem) = zext i32 %\(p)byte_\(elem) to i64\n"
        ir += "  %\(p)ptr_\(elem) = getelementptr i8, i8 addrspace(1)* \(bufferName), i64 %\(p)byte64_\(elem)\n"
        ir += "  %\(p)typed_\(elem) = bitcast i8 addrspace(1)* %\(p)ptr_\(elem) to \(memType) addrspace(1)*\n"
        ir += "  %\(p)load_\(elem) = load \(memType), \(memType) addrspace(1)* %\(p)typed_\(elem)\n"
      }
      if memPrec != regPrec {
        for elem in 0..<2 {
          ir += "  %\(p)ext_\(elem) = fpext \(memType) %\(p)load_\(elem) to \(regType)\n"
        }
        ir += "  %\(p)v2_a = insertelement <2 x \(regType)> undef, \(regType) %\(p)ext_0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(regType)> %\(p)v2_a, \(regType) %\(p)ext_1, i32 1\n"
      } else {
        ir += "  %\(p)v2_a = insertelement <2 x \(memType)> undef, \(memType) %\(p)load_0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(memType)> %\(p)v2_a, \(memType) %\(p)load_1, i32 1\n"
      }
    } else {
      // Non-transposed: load contiguous <2 x T> from row-major layout
      ir += "  %\(p)addr = mul i32 \(seqOffset), \(leadingDim)\n"
      ir += "  %\(p)addr2 = add i32 %\(p)addr, \(headOffset)\n"
      ir += "  %\(p)byte = mul i32 %\(p)addr2, \(elemSize)\n"
      ir += "  %\(p)byte64 = zext i32 %\(p)byte to i64\n"
      ir += "  %\(p)ptr = getelementptr i8, i8 addrspace(1)* \(bufferName), i64 %\(p)byte64\n"
      ir += "  %\(p)typed = bitcast i8 addrspace(1)* %\(p)ptr to <2 x \(memType)> addrspace(1)*\n"
      ir += "  %\(p)load = load <2 x \(memType)>, <2 x \(memType)> addrspace(1)* %\(p)typed, align \(elemSize * 2)\n"

      if memPrec != regPrec {
        ir += "  %\(p)e0 = extractelement <2 x \(memType)> %\(p)load, i32 0\n"
        ir += "  %\(p)e1 = extractelement <2 x \(memType)> %\(p)load, i32 1\n"
        ir += "  %\(p)ext0 = fpext \(memType) %\(p)e0 to \(regType)\n"
        ir += "  %\(p)ext1 = fpext \(memType) %\(p)e1 to \(regType)\n"
        ir += "  %\(p)v2_a = insertelement <2 x \(regType)> undef, \(regType) %\(p)ext0, i32 0\n"
        ir += "  %\(p)v2 = insertelement <2 x \(regType)> %\(p)v2_a, \(regType) %\(p)ext1, i32 1\n"
      } else {
        ir += "  %\(p)v2 = bitcast <2 x \(memType)> %\(p)load to <2 x \(memType)>\n"
      }
    }

    // Expand <2 x T> to <64 x T>
    ir += irShuffleToVec64(result: "%\(p)sram", src: "%\(p)v2", type: regPrec) + "\n"

    return ir
  }

  // MARK: - Outer Product (S = A * B^T)

  /// Generate the outer product: for each d_outer block, async copy B to TG,
  /// barrier, load A from device or cache, load B tiles from TG interleaved
  /// with matmul.
  ///
  /// For B (RHS): always uses TG via async copy.
  /// For A (LHS): uses cached registers if cachedA, otherwise loads directly
  /// from device memory (matching the reference's pattern).
  ///
  /// Result: S accumulators are in %{C_name}_final_0 .. %{C_name}_final_{count-1}
  func generateOuterProduct(
    prefix p: String,
    A: AttentionOperand, B: AttentionOperand, C_name: String,
    sSramCount: Int,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32, traversalDim: UInt32,
    traversalOffset: String,
    regA: GEMMOperandPrecision, regB: GEMMOperandPrecision,
    regC: GEMMOperandPrecision,
    memA: GEMMOperandPrecision, memB: GEMMOperandPrecision,
    leadingDimA: UInt32, leadingDimB: UInt32,
    leadingBlockDimA: UInt32, leadingBlockDimB: UInt32,
    cachedA: Bool,
    transposedA: Bool, transposedB: Bool,
    cachePrefix: String = "cq_",  // prefix for cached A registers (e.g., "cq_", "ck_", "cv_", "cdo_")
    tgOffset: String = "0",             // byte offset into TG for B (K) slot
    skipFirstIterCopy: Bool = false     // if true, skip first d_outer iteration's async copy (TG already loaded)
  ) -> String {
    var ir = ""
    ir += "  ; === Outer Product \(A) * \(B)^T → \(C_name) ===\n"

    // Current accumulator names (chained through d_outer iterations)
    var cNames = (0..<sSramCount).map { "%\(C_name)_init_\($0)" }

    // Head masking for B (K/V) TG loads: when D < paddedD, TG columns beyond D
    // contain garbage (async_copy doesn't zero-fill). If a thread's head position
    // >= D, the loaded register must be zeroed to prevent NaN poisoning the matmul.
    // The head position for B's TG load depends on transpose state:
    //   non-transposed B: col = morton_y + kOff  (head dim along TG columns)
    //   transposed B:     row = morton_y + kOff  (head dim along TG rows)
    // In both cases, morton_y + kOff is the head offset to check against D.
    let needsHeadMaskB = D < paddedD

    // We unroll the head dimension loop at codegen time (same as Metal source)
    // Head loop: d_outer from 0 to headLoopFloor by blockH, then edge
    func emitHeadIteration(dOuterVal: UInt32, regSize: UInt32, iterIdx: Int) {
      let ip = "\(p)\(iterIdx)_"
      let dOuterStr = "\(dOuterVal)"
      let kSteps = Int(regSize / 8)

      if !cachedA {
        // A (Q) not cached: load Q directly from device, K via TG.
        // Matches reference pattern: async_copy K→TG, barrier, then
        // for each d { load Q from device, load K from TG, matmul }.
        // This avoids the extra async copy + barrier for Q entirely.

        // Async copy B (K) to TG (skip first iter if already loaded externally)
        if !(skipFirstIterCopy && iterIdx == 0) {
          ir += generateAsyncCopyDeviceToTG(
            prefix: "\(ip)b",
            buffer: "%\(B)",
            operand: B,
            dOuter: dOuterStr,
            seqOffset: traversalOffset,
            seqDim: traversalDim,
            blockSeq: blockT,
            blockHead: UInt16(regSize),
            D: D,
            leadingDim: leadingDimB,
            leadingBlockDim: leadingBlockDimB,
            memPrec: memB,
            transposed: transposedB,
            tgOffset: tgOffset
          )
        }

        // Head masking for A (Q) when not cached — same NaN prevention
        // as cached Q, but applied per d-step in the uncached path.
        let needsHeadMaskA = D < paddedD

        // Interleaved: for each d step, load Q from device, then
        // for each traversal tile, load K from TG + matmul.
        for k in 0..<kSteps {
          let kOff = k * 8
          let headOffsetA = Int(dOuterVal) + kOff

          // Load A (Q) directly from device memory for this d step
          let aPrefix = "\(ip)a\(k)_"
          let aNeedsMask = needsHeadMaskA && headOffsetA + 8 > Int(D)

          // Check at codegen time: is this entire k-step beyond D?
          if needsHeadMaskA && headOffsetA >= D {
            ir += "  %\(aPrefix)sram = bitcast \(irVecType(regA)) zeroinitializer to \(irVecType(regA))\n"
          } else {
            let aLoadPrefix = aNeedsMask ? "\(aPrefix)r_" : aPrefix
            // Q address: Q[seq, head] where seq = parallelization thread offset,
            // head = d_outer + kOff + morton_x (each thread's 2 elements)
            if transposedA {
              // Q stored as Q^T[head, seq]: addr = (head) * leadingDimA + (seq)
              ir += "  %\(aLoadPrefix)seq = add i32 %clamped_par_off, 0\n"
              ir += "  %\(aLoadPrefix)head = add i32 %morton_x, \(headOffsetA)\n"
              ir += generateDeviceLoad(
                prefix: aLoadPrefix,
                bufferName: "%\(A)",
                seqOffset: "%\(aLoadPrefix)seq",
                headOffset: "%\(aLoadPrefix)head",
                leadingDim: leadingDimA,
                memPrec: memA,
                regPrec: regA,
                transposed: true
              )
            } else {
              // Q stored as Q[seq, head]: addr = (seq) * leadingDimA + (head)
              ir += "  %\(aLoadPrefix)seq = add i32 %clamped_par_off, 0\n"
              ir += "  %\(aLoadPrefix)head = add i32 %morton_x, \(headOffsetA)\n"
              ir += generateDeviceLoad(
                prefix: aLoadPrefix,
                bufferName: "%\(A)",
                seqOffset: "%\(aLoadPrefix)seq",
                headOffset: "%\(aLoadPrefix)head",
                leadingDim: leadingDimA,
                memPrec: memA,
                regPrec: regA,
                transposed: false
              )
            }

            // Head masking: zero when morton_x + headOffset >= D
            // morton_x ∈ {0,2,4,6} already gives the head column position
            if aNeedsMask {
              ir += "  %\(aPrefix)hpos = add i32 %morton_x, \(headOffsetA)\n"
              ir += "  %\(aPrefix)oob = icmp uge i32 %\(aPrefix)hpos, \(D)\n"
              ir += "  %\(aPrefix)sram = select i1 %\(aPrefix)oob, \(irVecType(regA)) zeroinitializer, \(irVecType(regA)) %\(aLoadPrefix)sram\n"
            }
          }

          // Load B (K) tiles from TG and multiply immediately
          for t in 0..<sSramCount {
            let tOff = t * 8
            let bPrefix = "\(ip)b\(k)x\(t)_"
            let headOffsetB = Int(dOuterVal) + kOff
            let bNeedsMask = needsHeadMaskB && headOffsetB + 8 > Int(D)

            // Check at codegen time: is this entire k-step beyond D?
            if needsHeadMaskB && headOffsetB >= D {
              // Entire B register is beyond D — zero it
              ir += "  %\(bPrefix)sram = bitcast \(irVecType(regB)) zeroinitializer to \(irVecType(regB))\n"
            } else {
              // Use a raw prefix when head masking is needed, so TG load
              // produces %{raw}sram and we can select into %{b}sram
              let loadPrefix = bNeedsMask ? "\(bPrefix)r_" : bPrefix
              if transposedB {
                ir += "  %\(loadPrefix)row = add i32 %morton_y, \(kOff)\n"
                ir += "  %\(loadPrefix)col = add i32 %morton_x, \(tOff)\n"
              } else {
                ir += "  %\(loadPrefix)row = add i32 %morton_x, \(tOff)\n"
                ir += "  %\(loadPrefix)col = add i32 %morton_y, \(kOff)\n"
              }
              ir += generateTGLoad(
                prefix: loadPrefix,
                tgOffset: tgOffset,
                rowOffset: "%\(loadPrefix)row",
                colOffset: "%\(loadPrefix)col",
                leadingBlockDim: leadingBlockDimB,
                memPrec: memB,
                regPrec: regB,
                transposed: !transposedB
              )

              // Head masking: zero the register if this thread's head position >= D
              if bNeedsMask {
                ir += "  %\(bPrefix)hpos = add i32 %morton_y, \(headOffsetB)\n"
                ir += "  %\(bPrefix)oob = icmp uge i32 %\(bPrefix)hpos, \(D)\n"
                ir += "  %\(bPrefix)sram = select i1 %\(bPrefix)oob, \(irVecType(regB)) zeroinitializer, \(irVecType(regB)) %\(loadPrefix)sram\n"
              }
            }

            let cIn = (k == 0) ? cNames[t] : "%\(ip)c\(k-1)x\(t)"
            let cOut = "%\(ip)c\(k)x\(t)"
            ir += irMultiplyAccumulateCall(
              result: cOut,
              A: ("%\(aPrefix)sram", regA),
              B: ("%\(bPrefix)sram", regB),
              C: (cIn, regC)
            ) + "\n"
          }
        }

      } else {
        // A is cached: only B needs TG. Copy B → load B → use cached A → matmul.
        if !(skipFirstIterCopy && iterIdx == 0) {
          ir += generateAsyncCopyDeviceToTG(
            prefix: "\(ip)b",
            buffer: "%\(B)",
            operand: B,
            dOuter: dOuterStr,
            seqOffset: traversalOffset,
            seqDim: traversalDim,
            blockSeq: blockT,
            blockHead: UInt16(regSize),
            D: D,
            leadingDim: leadingDimB,
            leadingBlockDim: leadingBlockDimB,
            memPrec: memB,
            transposed: transposedB,
            tgOffset: tgOffset
          )
        }

        for k in 0..<kSteps {
          let kOff = k * 8

          // Use cached A register
          let aPrefix = "\(ip)a\(k)_"
          let cachedIdx = (Int(dOuterVal) + kOff) / 8
          ir += "  %\(aPrefix)sram = bitcast \(irVecType(regA)) %\(cachePrefix)sram_\(cachedIdx) to \(irVecType(regA))\n"

          // Load B tile(s) and multiply-accumulate
          for t in 0..<sSramCount {
            let tOff = t * 8
            let bPrefix = "\(ip)b\(k)x\(t)_"
            let headOffsetB = Int(dOuterVal) + kOff
            let bNeedsMask = needsHeadMaskB && headOffsetB + 8 > Int(D)

            if needsHeadMaskB && headOffsetB >= D {
              ir += "  %\(bPrefix)sram = bitcast \(irVecType(regB)) zeroinitializer to \(irVecType(regB))\n"
            } else {
              let loadPrefix = bNeedsMask ? "\(bPrefix)r_" : bPrefix
              if transposedB {
                ir += "  %\(loadPrefix)row = add i32 %morton_y, \(kOff)\n"
                ir += "  %\(loadPrefix)col = add i32 %morton_x, \(tOff)\n"
              } else {
                ir += "  %\(loadPrefix)row = add i32 %morton_x, \(tOff)\n"
                ir += "  %\(loadPrefix)col = add i32 %morton_y, \(kOff)\n"
              }
              ir += generateTGLoad(
                prefix: loadPrefix,
                tgOffset: tgOffset,
                rowOffset: "%\(loadPrefix)row",
                colOffset: "%\(loadPrefix)col",
                leadingBlockDim: leadingBlockDimB,
                memPrec: memB,
                regPrec: regB,
                transposed: !transposedB
              )

              if bNeedsMask {
                ir += "  %\(bPrefix)hpos = add i32 %morton_y, \(headOffsetB)\n"
                ir += "  %\(bPrefix)oob = icmp uge i32 %\(bPrefix)hpos, \(D)\n"
                ir += "  %\(bPrefix)sram = select i1 %\(bPrefix)oob, \(irVecType(regB)) zeroinitializer, \(irVecType(regB)) %\(loadPrefix)sram\n"
              }
            }

            let cIn = (k == 0) ? cNames[t] : "%\(ip)c\(k-1)x\(t)"
            let cOut = "%\(ip)c\(k)x\(t)"
            ir += irMultiplyAccumulateCall(
              result: cOut,
              A: ("%\(aPrefix)sram", regA),
              B: ("%\(bPrefix)sram", regB),
              C: (cIn, regC)
            ) + "\n"
          }
        }
      }

      // Barrier before next head iteration overwrites TG
      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

      // Update accumulator names
      let lastK = kSteps - 1
      cNames = (0..<sSramCount).map { "%\(ip)c\(lastK)x\($0)" }
    }

    // Emit head iterations
    var iterIdx = 0
    var dOuter: UInt32 = 0
    while dOuter < headLoopFloor {
      emitHeadIteration(dOuterVal: dOuter, regSize: UInt32(blockH), iterIdx: iterIdx)
      dOuter += UInt32(blockH)
      iterIdx += 1
    }
    // Edge iteration if needed
    if headLoopFloor < paddedD {
      emitHeadIteration(dOuterVal: headLoopFloor, regSize: headEdge, iterIdx: iterIdx)
    }

    // Final names
    for i in 0..<sSramCount {
      ir += "  %\(C_name)_final_\(i) = bitcast \(irVecType(regC)) \(cNames[i]) to \(irVecType(regC))\n"
    }

    return ir
  }

  // MARK: - Mask Attention Matrix Edge

  func generateMaskEdge(
    prefix p: String,
    sSramCount: Int,
    blockT: UInt16, traversalDim: UInt32,
    traversalOffset: String,
    regS: GEMMOperandPrecision,
    scaleFactor: Float
  ) -> String {
    let logBase2E: Float = 1.442695041
    let t = irTypeName(regS)
    var ir = ""

    ir += "  ; === Mask attention matrix edge ===\n"

    // remainder = traversalDim % blockT
    let remainder = traversalDim % UInt32(blockT)
    if remainder == 0 {
      // No masking needed when traversal is perfectly divisible
      // But we still need to handle the case at runtime
      ir += "  ; No edge masking needed (traversalDim divisible by blockT)\n"
      for i in 0..<sSramCount {
        ir += "  %\(p)s_\(i) = bitcast \(irVecType(regS)) %s_final_\(i) to \(irVecType(regS))\n"
      }
      return ir
    }

    // Check if this is an edge iteration
    let blockEnd = "%\(p)block_end"
    ir += "  \(blockEnd) = add i32 \(traversalOffset), \(blockT)\n"
    ir += "  %\(p)is_edge = icmp ugt i32 \(blockEnd), \(traversalDim)\n"
    ir += "  br i1 %\(p)is_edge, label %\(p)do_mask, label %\(p)skip_mask\n\n"

    ir += "\(p)do_mask:\n"
    // mask_value = (0.875 / logBase2E) * -FLT_MAX
    let maskValue: Float = (0.875 / logBase2E) * -Float.greatestFiniteMagnitude
    ir += "  %\(p)mask_val_f32 = bitcast i32 \(maskValue.bitPattern) to float\n"
    let maskValName: String
    if regS == .FP32 {
      maskValName = "%\(p)mask_val_f32"
    } else {
      maskValName = "%\(p)mask_val_cvt"
      ir += "  \(maskValName) = fptrunc float %\(p)mask_val_f32 to \(irTypeName(regS))\n"
    }

    // Compute which elements to mask
    // remainder_rt = traversalDim - traversalOffset (runtime remainder)
    ir += "  %\(p)rem_rt = sub i32 \(traversalDim), \(traversalOffset)\n"
    // remainderFloor = rem_rt - (rem_rt % 8)
    ir += "  %\(p)rem_mod8 = and i32 %\(p)rem_rt, 7\n"
    ir += "  %\(p)rem_floor = sub i32 %\(p)rem_rt, %\(p)rem_mod8\n"

    // For the block at remainderFloor/8, mask elements where morton_x + index >= remainder - remainderFloor
    // For blocks after remainderFloor, mask all elements
    for i in 0..<sSramCount {
      let blockStart = i * 8
      // If blockStart >= rem_rt, mask everything
      ir += "  %\(p)bs_\(i) = icmp uge i32 \(blockStart), %\(p)rem_rt\n"

      // Extract the 2 thread elements
      ir += "  %\(p)e0_\(i) = extractelement \(irVecType(regS)) %s_final_\(i), i32 0\n"
      ir += "  %\(p)e1_\(i) = extractelement \(irVecType(regS)) %s_final_\(i), i32 1\n"

      // For the edge block (blockStart == remainderFloor):
      // mask element if morton_x + index >= (rem_rt - remainderFloor)
      ir += "  %\(p)is_edge_blk_\(i) = icmp eq i32 \(blockStart), %\(p)rem_floor\n"

      // element 0: mask if morton_x + 0 >= rem_mod8
      ir += "  %\(p)e0_oob_\(i) = icmp uge i32 %morton_x, %\(p)rem_mod8\n"
      ir += "  %\(p)e0_mask_\(i) = and i1 %\(p)is_edge_blk_\(i), %\(p)e0_oob_\(i)\n"
      // element 1: mask if morton_x + 1 >= rem_mod8
      ir += "  %\(p)mx_p1_\(i) = add i32 %morton_x, 1\n"
      ir += "  %\(p)e1_oob_\(i) = icmp uge i32 %\(p)mx_p1_\(i), %\(p)rem_mod8\n"
      ir += "  %\(p)e1_mask_\(i) = and i1 %\(p)is_edge_blk_\(i), %\(p)e1_oob_\(i)\n"

      // Combine: mask if full block OOB or individual element OOB
      ir += "  %\(p)m0_\(i) = or i1 %\(p)bs_\(i), %\(p)e0_mask_\(i)\n"
      ir += "  %\(p)m1_\(i) = or i1 %\(p)bs_\(i), %\(p)e1_mask_\(i)\n"

      // Select masked values
      let maskCast = maskValName
      ir += "  %\(p)me0_\(i) = select i1 %\(p)m0_\(i), \(t) \(maskCast), \(t) %\(p)e0_\(i)\n"
      ir += "  %\(p)me1_\(i) = select i1 %\(p)m1_\(i), \(t) \(maskCast), \(t) %\(p)e1_\(i)\n"

      // Reconstruct <64 x T>
      ir += "  %\(p)sv0_\(i) = insertelement \(irVecType(regS)) %s_final_\(i), \(t) %\(p)me0_\(i), i32 0\n"
      ir += "  %\(p)masked_\(i) = insertelement \(irVecType(regS)) %\(p)sv0_\(i), \(t) %\(p)me1_\(i), i32 1\n"
    }

    ir += "  br label %\(p)after_mask\n\n"
    ir += "\(p)skip_mask:\n"
    ir += "  br label %\(p)after_mask\n\n"
    ir += "\(p)after_mask:\n"

    // Phi nodes
    for i in 0..<sSramCount {
      ir += "  %\(p)s_\(i) = phi \(irVecType(regS)) [%\(p)masked_\(i), %\(p)do_mask], [%s_final_\(i), %\(p)skip_mask]\n"
    }
    ir += "\n"

    return ir
  }

  // MARK: - Online Softmax: Reduce Maximum

  func generateReduceMax(
    prefix p: String,
    sSramCount: Int,
    regS: GEMMOperandPrecision,
    scaleFactor: Float
  ) -> String {
    let t = irTypeName(regS)
    var ir = ""

    ir += "  ; === Reduce max ===\n"

    // Extract element pairs and compute max
    for i in 0..<sSramCount {
      ir += "  %\(p)e0_\(i) = extractelement \(irVecType(regS)) %mask_s_\(i), i32 0\n"
      ir += "  %\(p)e1_\(i) = extractelement \(irVecType(regS)) %mask_s_\(i), i32 1\n"
      if i == 0 {
        ir += "  %\(p)max_\(i) = fcmp fast ogt \(t) %\(p)e0_\(i), %\(p)e1_\(i)\n"
        ir += "  %\(p)m_\(i) = select i1 %\(p)max_\(i), \(t) %\(p)e0_\(i), \(t) %\(p)e1_\(i)\n"
      } else {
        let prev = "%\(p)m_\(i-1)"
        ir += "  %\(p)cmp0_\(i) = fcmp fast ogt \(t) %\(p)e0_\(i), \(prev)\n"
        ir += "  %\(p)sel0_\(i) = select i1 %\(p)cmp0_\(i), \(t) %\(p)e0_\(i), \(t) \(prev)\n"
        ir += "  %\(p)cmp1_\(i) = fcmp fast ogt \(t) %\(p)e1_\(i), %\(p)sel0_\(i)\n"
        ir += "  %\(p)m_\(i) = select i1 %\(p)cmp1_\(i), \(t) %\(p)e1_\(i), \(t) %\(p)sel0_\(i)\n"
      }
    }

    let lastM = "%\(p)m_\(sSramCount - 1)"

    // SIMD reduction: shuffle_xor with masks 1 and 8
    // m_new = max across thread elements
    if t == "float" {
      ir += "  %\(p)mf = bitcast float \(lastM) to float\n"
    } else {
      ir += "  %\(p)mf = fpext \(t) \(lastM) to float\n"
    }
    ir += irShuffleXorCall(result: "%\(p)shuf1", value: "%\(p)mf", mask: 1) + "\n"
    ir += "  %\(p)cmp_s1 = fcmp fast ogt float %\(p)mf, %\(p)shuf1\n"
    ir += "  %\(p)max_s1 = select i1 %\(p)cmp_s1, float %\(p)mf, float %\(p)shuf1\n"
    ir += irShuffleXorCall(result: "%\(p)shuf8", value: "%\(p)max_s1", mask: 8) + "\n"
    ir += "  %\(p)cmp_s8 = fcmp fast ogt float %\(p)max_s1, %\(p)shuf8\n"
    ir += "  %\(p)m_new = select i1 %\(p)cmp_s8, float %\(p)max_s1, float %\(p)shuf8\n"

    // Scale by scaleFactor
    let scaleDoubleHex = String(Double(scaleFactor).bitPattern, radix: 16, uppercase: true)
    ir += "  %\(p)m_new_scaled = fmul fast float %\(p)m_new, 0x\(scaleDoubleHex)\n"

    return ir
  }

  // MARK: - Online Softmax: Correct O

  func generateCorrectO(
    prefix p: String,
    oCachedCount: Int,
    regO: GEMMOperandPrecision
  ) -> String {
    var ir = ""
    ir += "  ; === Correct O ===\n"

    // correction = (m_new > m) ? exp2(m - m_new) : 1.0
    ir += "  %\(p)m_gt = fcmp fast ogt float %rmax_m_new_scaled, %m_phi\n"
    ir += "  %\(p)m_diff = fsub fast float %m_phi, %rmax_m_new_scaled\n"
    ir += irExp2Call(result: "%\(p)exp_diff", value: "%\(p)m_diff") + "\n"
    ir += "  %\(p)correction = select i1 %\(p)m_gt, float %\(p)exp_diff, float 1.0\n"

    // m = max(m, m_new)
    // (already computed as rmax_m_new_scaled, but update only if new > old)
    ir += "  %\(p)m_upd = select i1 %\(p)m_gt, float %rmax_m_new_scaled, float %m_phi\n"

    return ir
  }

  // MARK: - Online Softmax: Compute P = exp2(S * scale - m)

  func generateComputeP(
    prefix p: String,
    sSramCount: Int,
    regS: GEMMOperandPrecision,
    regP: GEMMOperandPrecision,
    scaleFactor: Float
  ) -> String {
    let tS = irTypeName(regS)
    let tP = irTypeName(regP)
    var ir = ""

    ir += "  ; === Compute P = exp2(S * scale - m) ===\n"

    for i in 0..<sSramCount {
      // Extract S elements
      ir += "  %\(p)s0_\(i) = extractelement \(irVecType(regS)) %mask_s_\(i), i32 0\n"
      ir += "  %\(p)s1_\(i) = extractelement \(irVecType(regS)) %mask_s_\(i), i32 1\n"

      // Convert to float for exp2
      let s0f = (regS == .FP32) ? "%\(p)s0_\(i)" : "%\(p)s0f_\(i)"
      let s1f = (regS == .FP32) ? "%\(p)s1_\(i)" : "%\(p)s1f_\(i)"
      if regS != .FP32 {
        ir += "  \(s0f) = fpext \(tS) %\(p)s0_\(i) to float\n"
        ir += "  \(s1f) = fpext \(tS) %\(p)s1_\(i) to float\n"
      }

      // P_i = exp2(S_i * scaleFactor - m)
      let scaleHex = "0x\(String(Double(scaleFactor).bitPattern, radix: 16, uppercase: true))"
      ir += "  %\(p)scaled0_\(i) = fmul fast float \(s0f), \(scaleHex)\n"
      ir += "  %\(p)shifted0_\(i) = fsub fast float %\(p)scaled0_\(i), %corr_m_upd\n"
      ir += irExp2Call(result: "%\(p)p0f_\(i)", value: "%\(p)shifted0_\(i)") + "\n"

      ir += "  %\(p)scaled1_\(i) = fmul fast float \(s1f), \(scaleHex)\n"
      ir += "  %\(p)shifted1_\(i) = fsub fast float %\(p)scaled1_\(i), %corr_m_upd\n"
      ir += irExp2Call(result: "%\(p)p1f_\(i)", value: "%\(p)shifted1_\(i)") + "\n"

      // Convert back to P precision and construct <64 x T>
      let p0 = (regP == .FP32) ? "%\(p)p0f_\(i)" : "%\(p)p0_\(i)"
      let p1 = (regP == .FP32) ? "%\(p)p1f_\(i)" : "%\(p)p1_\(i)"
      if regP != .FP32 {
        ir += "  \(p0) = fptrunc float %\(p)p0f_\(i) to \(tP)\n"
        ir += "  \(p1) = fptrunc float %\(p)p1f_\(i) to \(tP)\n"
      }

      ir += "  %\(p)pv0_\(i) = insertelement \(irVecType(regP)) undef, \(tP) \(p0), i32 0\n"
      ir += "  %\(p)p_\(i) = insertelement \(irVecType(regP)) %\(p)pv0_\(i), \(tP) \(p1), i32 1\n"
    }

    return ir
  }

  // MARK: - Online Softmax: Reduce Sum

  func generateReduceSum(
    prefix p: String,
    sSramCount: Int,
    regP: GEMMOperandPrecision
  ) -> String {
    let tP = irTypeName(regP)
    var ir = ""

    ir += "  ; === Reduce sum ===\n"

    // Sum P elements
    for i in 0..<sSramCount {
      ir += "  %\(p)e0_\(i) = extractelement \(irVecType(regP)) %sp_p_\(i), i32 0\n"
      ir += "  %\(p)e1_\(i) = extractelement \(irVecType(regP)) %sp_p_\(i), i32 1\n"

      let e0f = (regP == .FP32) ? "%\(p)e0_\(i)" : "%\(p)e0f_\(i)"
      let e1f = (regP == .FP32) ? "%\(p)e1_\(i)" : "%\(p)e1f_\(i)"
      if regP != .FP32 {
        ir += "  \(e0f) = fpext \(tP) %\(p)e0_\(i) to float\n"
        ir += "  \(e1f) = fpext \(tP) %\(p)e1_\(i) to float\n"
      }

      if i == 0 {
        ir += "  %\(p)sum_\(i) = fadd fast float \(e0f), \(e1f)\n"
      } else {
        ir += "  %\(p)add0_\(i) = fadd fast float %\(p)sum_\(i-1), \(e0f)\n"
        ir += "  %\(p)sum_\(i) = fadd fast float %\(p)add0_\(i), \(e1f)\n"
      }
    }

    let lastSum = "%\(p)sum_\(sSramCount - 1)"

    // SIMD reduction
    ir += irShuffleXorCall(result: "%\(p)shuf1", value: lastSum, mask: 1) + "\n"
    ir += "  %\(p)sum_s1 = fadd fast float \(lastSum), %\(p)shuf1\n"
    ir += irShuffleXorCall(result: "%\(p)shuf8", value: "%\(p)sum_s1", mask: 8) + "\n"
    ir += "  %\(p)l_new_part = fadd fast float %\(p)sum_s1, %\(p)shuf8\n"

    // l = l * correction + l_new
    ir += "  %\(p)l_corrected = fmul fast float %l_phi, %corr_correction\n"
    ir += "  %\(p)l_new = fadd fast float %\(p)l_corrected, %\(p)l_new_part\n"

    return ir
  }

  // MARK: - Accumulate O += P * V

  func generateAccumulate(
    prefix p: String,
    A: AttentionOperand, B: AttentionOperand, C_name: String,
    accCount: Int,
    blockP: UInt16, blockT: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32, traversalDim: UInt32,
    traversalOffset: String,
    regA: GEMMOperandPrecision, regB: GEMMOperandPrecision,
    regC: GEMMOperandPrecision,
    memB: GEMMOperandPrecision,
    leadingDimB: UInt32,
    leadingBlockDimB: UInt32,
    transposedB: Bool,
    cachedC: Bool,
    isFinalScale: Bool,
    scaleCorrection: String,  // SSA name of correction factor (or "" if none)
    aSourcePrefix: String = "sp_p",       // SSA prefix for A registers (e.g., "sp_p" → %sp_p_0)
    tgOffset: String = "0",              // byte offset into TG for V slot
    skipFirstIterCopy: Bool = false      // if true, skip first d_outer iteration's async copy
  ) -> String {
    var ir = ""
    ir += "  ; === Accumulate \(C_name) += P * V ===\n"

    // Note: head masking in the accumulate uses %morton_x directly.
    // morton_x ∈ {0,2,4,6} already gives the head column position.

    // Scale existing accumulators by correction
    if !scaleCorrection.isEmpty {
      for i in 0..<accCount {
        ir += "  %\(p)scale_e0_\(i) = extractelement \(irVecType(regC)) %\(C_name)_phi_\(i), i32 0\n"
        ir += "  %\(p)scale_e1_\(i) = extractelement \(irVecType(regC)) %\(C_name)_phi_\(i), i32 1\n"
        ir += "  %\(p)scaled_e0_\(i) = fmul fast float %\(p)scale_e0_\(i), %\(scaleCorrection)\n"
        ir += "  %\(p)scaled_e1_\(i) = fmul fast float %\(p)scale_e1_\(i), %\(scaleCorrection)\n"
        ir += "  %\(p)sv0_\(i) = insertelement \(irVecType(regC)) %\(C_name)_phi_\(i), float %\(p)scaled_e0_\(i), i32 0\n"
        ir += "  %\(p)corrected_\(i) = insertelement \(irVecType(regC)) %\(p)sv0_\(i), float %\(p)scaled_e1_\(i), i32 1\n"
      }
    } else {
      for i in 0..<accCount {
        ir += "  %\(p)corrected_\(i) = bitcast \(irVecType(regC)) %\(C_name)_phi_\(i) to \(irVecType(regC))\n"
      }
    }

    // For each d_outer block of the head dimension:
    //   1. Async copy V to TG
    //   2. Barrier
    //   3. Matmul: O_block += P * V_block
    var cNames = (0..<accCount).map { "%\(p)corrected_\($0)" }

    func emitHeadIteration(dOuterVal: UInt32, regSize: UInt32, iterIdx: Int) {
      let ip = "\(p)\(iterIdx)_"

      // Async copy V to TG (skip first iter if already loaded externally)
      if !(skipFirstIterCopy && iterIdx == 0) {
        ir += generateAsyncCopyDeviceToTG(
          prefix: "\(ip)v",
          buffer: "%\(B)",
          operand: B,
          dOuter: "\(dOuterVal)",
          seqOffset: traversalOffset,
          seqDim: traversalDim,
          blockSeq: blockT,
          blockHead: UInt16(regSize),
          D: D,
          leadingDim: leadingDimB,
          leadingBlockDim: leadingBlockDimB,
          memPrec: memB,
          transposed: transposedB,
          tgOffset: tgOffset
        )
      }

      // Inner multiply: for each k step in traversal
      let kSteps = Int(blockT / 8)

      // Multiply P * V for each head block and traversal step
      let needsHeadMaskB = D < paddedD
      let dSteps = Int(regSize / 8)
      for k in 0..<kSteps {
        let kOff = k * 8

        for d in 0..<dSteps {
          let dOff = d * 8
          // Always index by absolute position in the head dimension.
          // Even when O is "uncached" in the original Metal source sense,
          // our IR uses oCachedCount SSA accumulators for all head positions.
          let accIdx = Int(dOuterVal) / 8 + d

          let absHeadOff = Int(dOuterVal) + dOff
          let bNeedsMask = needsHeadMaskB && absHeadOff + 8 > Int(D)

          // Load V tile from TG for this (k, d) pair.
          // V = B operand: B[k,j] = V[k,j] where k=traversal, j=head.
          let vPrefix = "\(ip)v\(k)x\(d)_"

          if needsHeadMaskB && absHeadOff >= Int(D) {
            // Entire B register is beyond D — zero it
            ir += "  %\(vPrefix)sram = bitcast \(irVecType(regB)) zeroinitializer to \(irVecType(regB))\n"
          } else {
            let loadPrefix = bNeedsMask ? "\(vPrefix)r_" : vPrefix
            if transposedB {
              ir += "  %\(loadPrefix)row = add i32 %morton_x, \(dOff)\n"
              ir += "  %\(loadPrefix)col = add i32 %morton_y, \(kOff)\n"
            } else {
              ir += "  %\(loadPrefix)row = add i32 %morton_y, \(kOff)\n"
              ir += "  %\(loadPrefix)col = add i32 %morton_x, \(dOff)\n"
            }
            ir += generateTGLoad(
              prefix: loadPrefix,
              tgOffset: tgOffset,
              rowOffset: "%\(loadPrefix)row",
              colOffset: "%\(loadPrefix)col",
              leadingBlockDim: leadingBlockDimB,
              memPrec: memB,
              regPrec: regB,
              transposed: transposedB
            )

            // Head masking: zero the register if this thread's head position >= D
            if bNeedsMask {
              // morton_x * 2 gives the thread's head position within the 8-wide block
              ir += "  %\(vPrefix)hpos = add i32 %morton_x, \(absHeadOff)\n"
              ir += "  %\(vPrefix)oob = icmp uge i32 %\(vPrefix)hpos, \(D)\n"
              ir += "  %\(vPrefix)sram = select i1 %\(vPrefix)oob, \(irVecType(regB)) zeroinitializer, \(irVecType(regB)) %\(loadPrefix)sram\n"
            }
          }

          // Load P tile: P_sram[k] (traversal step k)
          let pName = "%\(aSourcePrefix)_\(kOff / 8)"

          // Each d-accumulator chains independently across k-steps:
          //   k=0: cNames[accIdx] → c_k0_d{d}
          //   k>0: c_k{k-1}_d{d} → c_k{k}_d{d}
          let cInActual: String
          if k == 0 {
            cInActual = cNames[accIdx]
          } else {
            cInActual = "%\(ip)c\(k-1)d\(d)"
          }
          let cOut = "%\(ip)c\(k)d\(d)"

          ir += irMultiplyAccumulateCall(
            result: cOut,
            A: (pName, regA),
            B: ("%\(vPrefix)sram", regB),
            C: (cInActual, regC)
          ) + "\n"
        }
      }

      // Barrier
      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

      // Update accumulator names (always absolute head position)
      let lastK = kSteps - 1
      for d in 0..<dSteps {
        let accIdx = Int(dOuterVal) / 8 + d
        cNames[accIdx] = "%\(ip)c\(lastK)d\(d)"
      }
    }

    var iterIdx = 0
    var dOuter: UInt32 = 0
    while dOuter < headLoopFloor {
      emitHeadIteration(dOuterVal: dOuter, regSize: UInt32(blockH), iterIdx: iterIdx)
      dOuter += UInt32(blockH)
      iterIdx += 1
    }
    if headLoopFloor < paddedD {
      emitHeadIteration(dOuterVal: headLoopFloor, regSize: headEdge, iterIdx: iterIdx)
    }

    // Branch to after-head label (LLVM IR requires explicit terminators)
    ir += "  br label %\(p)after_head\n"
    ir += "\(p)after_head:\n"

    // Final names
    for i in 0..<accCount {
      ir += "  %\(p)\(C_name)_final_\(i) = bitcast \(irVecType(regC)) \(cNames[i]) to \(irVecType(regC))\n"
    }

    return ir
  }

  // MARK: - Cache Load (Q, dO, etc.)

  func generateCacheLoad(
    operand: AttentionOperand,
    prefix p: String,
    parallelDim: UInt32,
    D: UInt32,
    paddedD: UInt32,
    blockP: UInt16, blockH: UInt16,
    leadingDim: UInt32, leadingBlockDim: UInt32,
    memPrec: GEMMOperandPrecision, regPrec: GEMMOperandPrecision,
    transposed: Bool
  ) -> String {
    var ir = ""
    ir += "  ; === Cache load \(operand) directly from device ===\n"

    // Load directly from device memory into registers — no TG, no async copy,
    // no barriers. Matches reference pattern (preferAsyncCache=false on M1).
    //
    // Head dimension masking: each thread's morton_x determines which head
    // column pair it loads. When morton_x * 2 + headOffset >= D, the load
    // would access beyond the buffer. We must zero the register in that case.
    // For non-transposed, the <2 x T> load reads columns [headOffset, headOffset+1].
    // For transposed, the 2 scalar loads read rows [headOffset, headOffset+1].
    // In both cases, headOffset = morton_x * 2 + dOuter + kOff (morton_x
    // contributes ×2 because each thread holds 2 adjacent elements).

    // Head masking: morton_x ∈ {0,2,4,6} directly gives the head column position.
    let needsHeadMask = D < paddedD

    var dOuter: UInt32 = 0
    var iterIdx = 0
    while dOuter < paddedD {
      let regSize = min(UInt32(blockH), paddedD - dOuter)
      let ip = "\(p)\(iterIdx)_"

      let kSteps = Int(regSize / 8)
      for k in 0..<kSteps {
        let kOff = k * 8
        let regIdx = Int(dOuter) / 8 + k
        let lp = "\(ip)k\(k)_"

        // Check at codegen time if this entire k-step is beyond D
        let headStart = Int(dOuter) + kOff
        if headStart >= D {
          // Entire register is beyond D — just zero it
          ir += "  %\(p)sram_\(regIdx) = bitcast \(irVecType(regPrec)) zeroinitializer to \(irVecType(regPrec))\n"
          continue
        }

        if transposed {
          ir += "  %\(lp)seq = add i32 %clamped_par_off, 0\n"
          ir += "  %\(lp)head = add i32 %morton_x, \(headStart)\n"
          ir += generateDeviceLoad(
            prefix: lp,
            bufferName: "%\(operand)",
            seqOffset: "%\(lp)seq",
            headOffset: "%\(lp)head",
            leadingDim: leadingDim,
            memPrec: memPrec,
            regPrec: regPrec,
            transposed: true
          )
        } else {
          ir += "  %\(lp)seq = add i32 %clamped_par_off, 0\n"
          ir += "  %\(lp)head = add i32 %morton_x, \(headStart)\n"
          ir += generateDeviceLoad(
            prefix: lp,
            bufferName: "%\(operand)",
            seqOffset: "%\(lp)seq",
            headOffset: "%\(lp)head",
            leadingDim: leadingDim,
            memPrec: memPrec,
            regPrec: regPrec,
            transposed: false
          )
        }

        // Head masking: zero the register if this thread's head position >= D.
        // headPos = morton_x + headStart. morton_x ∈ {0,2,4,6}.
        if needsHeadMask && headStart + 8 > D {
          // Some threads in this k-step may be out of bounds
          ir += "  %\(lp)hpos = add i32 %morton_x, \(headStart)\n"
          ir += "  %\(lp)oob = icmp uge i32 %\(lp)hpos, \(D)\n"
          ir += "  %\(p)sram_\(regIdx) = select i1 %\(lp)oob, \(irVecType(regPrec)) zeroinitializer, \(irVecType(regPrec)) %\(lp)sram\n"
        } else {
          ir += "  %\(p)sram_\(regIdx) = bitcast \(irVecType(regPrec)) %\(lp)sram to \(irVecType(regPrec))\n"
        }
      }

      dOuter += UInt32(blockH)
      iterIdx += 1
    }

    return ir
  }

  // MARK: - Forward Cleanup

  func generateForwardCleanup(
    prefix p: String,
    oCachedCount: Int,
    blockP: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32,
    regO: GEMMOperandPrecision, memO: GEMMOperandPrecision,
    memL: GEMMOperandPrecision,
    leadingDimO: UInt32, leadingBlockDimO: UInt32,
    transposedO: Bool,
    cachedO: Bool
  ) -> String {
    let elemSizeO = UInt32(memO.size)
    let elemSizeL = UInt32(memL.size)
    let tO = irTypeName(regO)
    var ir = ""

    ir += """

    cleanup:
      ; === Forward cleanup: O /= l, store O, store L ===

    """

    // O /= l: scale each O accumulator element by 1/l
    ir += "  %\(p)inv_l = fdiv fast float 1.0, %l_phi\n"

    for i in 0..<oCachedCount {
      ir += "  %\(p)oe0_\(i) = extractelement \(irVecType(regO)) %o_phi_\(i), i32 0\n"
      ir += "  %\(p)oe1_\(i) = extractelement \(irVecType(regO)) %o_phi_\(i), i32 1\n"
      ir += "  %\(p)os0_\(i) = fmul fast float %\(p)oe0_\(i), %\(p)inv_l\n"
      ir += "  %\(p)os1_\(i) = fmul fast float %\(p)oe1_\(i), %\(p)inv_l\n"
      ir += "  %\(p)ov0_\(i) = insertelement \(irVecType(regO)) %o_phi_\(i), float %\(p)os0_\(i), i32 0\n"
      ir += "  %\(p)o_scaled_\(i) = insertelement \(irVecType(regO)) %\(p)ov0_\(i), float %\(p)os1_\(i), i32 1\n"
    }

    // Store O: cache store (registers → TG → device) for each d_outer block
    // Use async copy for edge safety
    var dOuter: UInt32 = 0
    var iterIdx = 0
    while dOuter < paddedD {
      let regSize = min(UInt32(blockH), paddedD - dOuter)
      let ip = "\(p)\(iterIdx)_"
      let kSteps = Int(regSize / 8)

      // Store registers to TG
      ir += "  ; Store O block d_outer=\(dOuter) to TG\n"
      for k in 0..<kSteps {
        let kOff = k * 8
        let regIdx = Int(dOuter) / 8 + k
        let sp = "\(ip)k\(k)_"

        // Unshuffle from <64 x T> to <2 x T>
        ir += irShuffleFromVec64(
          result: "%\(sp)v2", src: "%\(p)o_scaled_\(regIdx)", type: regO
        ) + "\n"

        // Convert precision if needed
        let storeVec: String
        let storeType: String
        if regO != memO {
          ir += "  %\(sp)se0 = extractelement <2 x \(tO)> %\(sp)v2, i32 0\n"
          ir += "  %\(sp)se1 = extractelement <2 x \(tO)> %\(sp)v2, i32 1\n"
          ir += "  %\(sp)st0 = fptrunc \(tO) %\(sp)se0 to \(irTypeName(memO))\n"
          ir += "  %\(sp)st1 = fptrunc \(tO) %\(sp)se1 to \(irTypeName(memO))\n"
          ir += "  %\(sp)sv0 = insertelement <2 x \(irTypeName(memO))> undef, \(irTypeName(memO)) %\(sp)st0, i32 0\n"
          ir += "  %\(sp)svec = insertelement <2 x \(irTypeName(memO))> %\(sp)sv0, \(irTypeName(memO)) %\(sp)st1, i32 1\n"
          storeVec = "%\(sp)svec"
          storeType = irTypeName(memO)
        } else {
          storeVec = "%\(sp)v2"
          storeType = tO
        }

        // Guard store: only if thread is in bounds
        ir += "  %\(sp)in_bounds = icmp ult i32 %unsafe_par_off, \(parallelDim)\n"
        ir += "  br i1 %\(sp)in_bounds, label %\(sp)do_store, label %\(sp)skip_store\n\n"
        ir += "\(sp)do_store:\n"

        if transposedO {
          // Transposed: elements 0,1 are at head positions morton_x+kOff and
          // morton_x+kOff+1 for the same seq row (oig_y). These map to
          // TG[head0, oig_y] and TG[head1, oig_y] which are non-contiguous.
          // Must use two scalar stores.
          let se0: String
          let se1: String
          if storeVec.hasSuffix("svec") {
            se0 = "%\(sp)st0"
            se1 = "%\(sp)st1"
          } else {
            // storeVec is %{sp}v2, extract elements
            ir += "  %\(sp)se0 = extractelement <2 x \(storeType)> \(storeVec), i32 0\n"
            ir += "  %\(sp)se1 = extractelement <2 x \(storeType)> \(storeVec), i32 1\n"
            se0 = "%\(sp)se0"
            se1 = "%\(sp)se1"
          }
          // Element 0 → TG[morton_x + kOff, oig_y]
          ir += "  %\(sp)tg_row0 = add i32 %morton_x, \(kOff)\n"
          ir += "  %\(sp)tg_addr0 = mul i32 %\(sp)tg_row0, \(leadingBlockDimO)\n"
          ir += "  %\(sp)tg_addr0b = add i32 %\(sp)tg_addr0, %oig_y\n"
          ir += "  %\(sp)tg_byte0 = mul i32 %\(sp)tg_addr0b, \(elemSizeO)\n"
          ir += "  %\(sp)tg_byte0_64 = zext i32 %\(sp)tg_byte0 to i64\n"
          ir += "  %\(sp)tg_ptr0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte0_64\n"
          ir += "  %\(sp)tg_typed0 = bitcast i8 addrspace(3)* %\(sp)tg_ptr0 to \(storeType) addrspace(3)*\n"
          ir += "  store \(storeType) \(se0), \(storeType) addrspace(3)* %\(sp)tg_typed0\n"
          // Element 1 → TG[morton_x + kOff + 1, oig_y]
          ir += "  %\(sp)tg_row1 = add i32 %morton_x, \(kOff + 1)\n"
          ir += "  %\(sp)tg_addr1 = mul i32 %\(sp)tg_row1, \(leadingBlockDimO)\n"
          ir += "  %\(sp)tg_addr1b = add i32 %\(sp)tg_addr1, %oig_y\n"
          ir += "  %\(sp)tg_byte1 = mul i32 %\(sp)tg_addr1b, \(elemSizeO)\n"
          ir += "  %\(sp)tg_byte1_64 = zext i32 %\(sp)tg_byte1 to i64\n"
          ir += "  %\(sp)tg_ptr1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte1_64\n"
          ir += "  %\(sp)tg_typed1 = bitcast i8 addrspace(3)* %\(sp)tg_ptr1 to \(storeType) addrspace(3)*\n"
          ir += "  store \(storeType) \(se1), \(storeType) addrspace(3)* %\(sp)tg_typed1\n"
        } else {
          // Non-transposed: elements 0,1 are at head positions morton_x+kOff
          // and morton_x+kOff+1, which are contiguous in memory. Use <2 x T> store.
          ir += "  %\(sp)tg_row = add i32 %oig_y, 0\n"
          ir += "  %\(sp)tg_addr = mul i32 %\(sp)tg_row, \(leadingBlockDimO)\n"
          ir += "  %\(sp)tg_col = add i32 %morton_x, \(kOff)\n"
          ir += "  %\(sp)tg_addr2 = add i32 %\(sp)tg_addr, %\(sp)tg_col\n"
          ir += "  %\(sp)tg_byte = mul i32 %\(sp)tg_addr2, \(elemSizeO)\n"
          ir += "  %\(sp)tg_byte64 = zext i32 %\(sp)tg_byte to i64\n"
          ir += "  %\(sp)tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte64\n"
          ir += "  %\(sp)tg_typed = bitcast i8 addrspace(3)* %\(sp)tg_ptr to <2 x \(storeType)> addrspace(3)*\n"
          ir += "  store <2 x \(storeType)> \(storeVec), <2 x \(storeType)> addrspace(3)* %\(sp)tg_typed\n"
        }

        ir += "  br label %\(sp)skip_store\n\n"
        ir += "\(sp)skip_store:\n"
      }

      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n"

      // Async copy TG → device (gated to sidx==0)
      let cp = "\(ip)cp_"
      ir += "  br i1 %is_sidx0, label %\(cp)do, label %\(cp)skip\n\n"
      ir += "\(cp)do:\n"

      // Device offset: par_group_off * leadingDimO + dOuter (non-transposed)
      if transposedO {
        ir += "  %\(cp)dev_row = mul i32 \(dOuter), \(leadingDimO)\n"
        ir += "  %\(cp)dev_off32 = add i32 %\(cp)dev_row, %par_group_off\n"
      } else {
        ir += "  %\(cp)dev_row = mul i32 %par_group_off, \(leadingDimO)\n"
        ir += "  %\(cp)dev_off32 = add i32 %\(cp)dev_row, \(dOuter)\n"
      }
      ir += "  %\(cp)dev_off = zext i32 %\(cp)dev_off32 to i64\n"
      ir += "  %\(cp)dev_byte = mul i64 %\(cp)dev_off, \(elemSizeO)\n"
      ir += "  %\(cp)dst_p = getelementptr i8, i8 addrspace(1)* %O, i64 %\(cp)dev_byte\n"
      ir += "  %\(cp)src_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

      // Tile: min(D - dOuter, blockH) wide, min(parallelDim - par_group_off, blockP) high
      let dTile = min(UInt32(blockH), D - min(dOuter, D))
      ir += "  %\(cp)seq_rem = sub i32 \(parallelDim), %par_group_off\n"
      ir += "  %\(cp)seq_cmp = icmp ult i32 %\(cp)seq_rem, \(blockP)\n"
      ir += "  %\(cp)seq_tile32 = select i1 %\(cp)seq_cmp, i32 %\(cp)seq_rem, i32 \(blockP)\n"
      ir += "  %\(cp)seq_tile = zext i32 %\(cp)seq_tile32 to i64\n"

      let dstStride = leadingDimO * elemSizeO
      let srcStride = leadingBlockDimO * elemSizeO

      let (tileW, tileH): (String, String)
      if transposedO {
        // Width = seq elements, Height = D elements
        ir += "  %\(cp)w_bytes32 = mul i32 %\(cp)seq_tile32, \(elemSizeO)\n"
        ir += "  %\(cp)w_bytes = zext i32 %\(cp)w_bytes32 to i64\n"
        tileW = "%\(cp)w_bytes"
        tileH = "\(dTile)"
      } else {
        // Width = D elements, Height = seq elements
        tileW = "\(dTile * elemSizeO)"
        tileH = "%\(cp)seq_tile"
      }

      ir += "  %\(cp)tile_w = insertelement <2 x i64> zeroinitializer, i64 \(tileW), i32 0\n"
      ir += "  %\(cp)tile = insertelement <2 x i64> %\(cp)tile_w, i64 \(tileH), i32 1\n"

      ir += "  %\(cp)evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"
      ir += """
        %\(cp)ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
          i64 1, i64 1,
          i8 addrspace(1)* %\(cp)dst_p, i64 \(dstStride), i64 1, <2 x i64> %\(cp)tile,
          i8 addrspace(3)* %\(cp)src_p, i64 \(srcStride), i64 1, <2 x i64> %\(cp)tile,
          <2 x i64> zeroinitializer, i32 0
        )
        store %event_t addrspace(3)* %\(cp)ev, %event_t addrspace(3)** %\(cp)evp
        call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %\(cp)evp)

      """

      ir += "  br label %\(cp)skip\n\n"
      ir += "\(cp)skip:\n"
      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

      dOuter += UInt32(blockH)
      iterIdx += 1
    }

    // Store L = m + log2(l) (per-thread scalar, one per parallelization element)
    ir += "  ; Store L\n"
    ir += "  %\(p)L_in_bounds = icmp ult i32 %unsafe_par_off, \(parallelDim)\n"
    ir += "  br i1 %\(p)L_in_bounds, label %\(p)store_L, label %\(p)skip_L\n\n"

    ir += "\(p)store_L:\n"
    ir += irLog2Call(result: "%\(p)log2_l", value: "%l_phi") + "\n"
    ir += "  %\(p)L_val = fadd fast float %m_phi, %\(p)log2_l\n"

    // Address: L_buf + clamped_par_off * elemSizeL
    ir += "  %\(p)L_off = zext i32 %clamped_par_off to i64\n"
    ir += "  %\(p)L_byte = mul i64 %\(p)L_off, \(elemSizeL)\n"
    ir += "  %\(p)L_ptr = getelementptr i8, i8 addrspace(1)* %L_buf, i64 %\(p)L_byte\n"
    if memL == .FP32 {
      ir += "  %\(p)L_typed = bitcast i8 addrspace(1)* %\(p)L_ptr to float addrspace(1)*\n"
      ir += "  store float %\(p)L_val, float addrspace(1)* %\(p)L_typed\n"
    } else {
      // Truncate to memL precision before storing
      let tL = irTypeName(memL)
      ir += "  %\(p)L_trunc = fptrunc float %\(p)L_val to \(tL)\n"
      ir += "  %\(p)L_typed = bitcast i8 addrspace(1)* %\(p)L_ptr to \(tL) addrspace(1)*\n"
      ir += "  store \(tL) %\(p)L_trunc, \(tL) addrspace(1)* %\(p)L_typed\n"
    }
    ir += "  br label %\(p)skip_L\n\n"

    ir += "\(p)skip_L:\n"
    ir += "  br label %exit\n"

    return ir
  }

  // MARK: - Softmax from stored L (backward)

  /// P = exp2(S * scaleFactor - L) where L is a per-row scalar (same for all traversal positions).
  /// Used in backwardQuery where L was stored during the forward pass.
  func generateSoftmaxFromL(
    prefix p: String,
    sSramCount: Int,
    regS: GEMMOperandPrecision, regP: GEMMOperandPrecision,
    scaleFactor: Float,
    lScalar: String,  // SSA name of the L scalar (float)
    sSource: String = "s_final"  // SSA prefix for S vectors (e.g., "s_final" → %s_final_0)
  ) -> String {
    let tS = irTypeName(regS)
    let tP = irTypeName(regP)
    var ir = ""

    ir += "  ; === Softmax from L: P = exp2(S * scale - L) ===\n"

    let scaleHex = "0x\(String(Double(scaleFactor).bitPattern, radix: 16, uppercase: true))"

    for i in 0..<sSramCount {
      ir += "  %\(p)s0_\(i) = extractelement \(irVecType(regS)) %\(sSource)_\(i), i32 0\n"
      ir += "  %\(p)s1_\(i) = extractelement \(irVecType(regS)) %\(sSource)_\(i), i32 1\n"

      let s0f = (regS == .FP32) ? "%\(p)s0_\(i)" : "%\(p)s0f_\(i)"
      let s1f = (regS == .FP32) ? "%\(p)s1_\(i)" : "%\(p)s1f_\(i)"
      if regS != .FP32 {
        ir += "  \(s0f) = fpext \(tS) %\(p)s0_\(i) to float\n"
        ir += "  \(s1f) = fpext \(tS) %\(p)s1_\(i) to float\n"
      }

      ir += "  %\(p)sc0_\(i) = fmul fast float \(s0f), \(scaleHex)\n"
      ir += "  %\(p)sh0_\(i) = fsub fast float %\(p)sc0_\(i), \(lScalar)\n"
      ir += irExp2Call(result: "%\(p)p0f_\(i)", value: "%\(p)sh0_\(i)") + "\n"

      ir += "  %\(p)sc1_\(i) = fmul fast float \(s1f), \(scaleHex)\n"
      ir += "  %\(p)sh1_\(i) = fsub fast float %\(p)sc1_\(i), \(lScalar)\n"
      ir += irExp2Call(result: "%\(p)p1f_\(i)", value: "%\(p)sh1_\(i)") + "\n"

      let p0 = (regP == .FP32) ? "%\(p)p0f_\(i)" : "%\(p)p0_\(i)"
      let p1 = (regP == .FP32) ? "%\(p)p1f_\(i)" : "%\(p)p1_\(i)"
      if regP != .FP32 {
        ir += "  \(p0) = fptrunc float %\(p)p0f_\(i) to \(tP)\n"
        ir += "  \(p1) = fptrunc float %\(p)p1f_\(i) to \(tP)\n"
      }

      ir += "  %\(p)pv0_\(i) = insertelement \(irVecType(regP)) undef, \(tP) \(p0), i32 0\n"
      ir += "  %\(p)p_\(i) = insertelement \(irVecType(regP)) %\(p)pv0_\(i), \(tP) \(p1), i32 1\n"
    }

    return ir
  }

  /// P = exp2(S * scaleFactor - L) where L is a per-column <2 x float> vector.
  /// Used in backwardKeyValue where L comes from the traversal dimension.
  /// The L vector has element 0 for morton_x*2 and element 1 for morton_x*2+1.
  /// P = exp2(S * scaleFactor - L) where L varies per-column of S^T.
  /// Used in backwardKeyValue where each sSram tile covers different traversal rows,
  /// so L must be loaded per-tile from the L buffer.
  func generateSoftmaxFromLVector(
    prefix p: String,
    sSramCount: Int,
    regS: GEMMOperandPrecision, regP: GEMMOperandPrecision,
    scaleFactor: Float,
    lBufferName: String,       // L buffer pointer (e.g., "%L_buf")
    traversalOffset: String,   // loop variable for traversal offset (e.g., "%bkv_r")
    traversalDim: UInt32,      // total traversal dimension (R)
    memL: GEMMOperandPrecision,
    sSource: String = "s_final"  // SSA prefix for S vectors
  ) -> String {
    let tS = irTypeName(regS)
    let tP = irTypeName(regP)
    let elemSizeL = UInt32(memL.size)
    let memTypeL = irTypeName(memL)
    var ir = ""

    ir += "  ; === Softmax from L vector: P = exp2(S * scale - L) ===\n"

    let scaleHex = "0x\(String(Double(scaleFactor).bitPattern, radix: 16, uppercase: true))"

    for i in 0..<sSramCount {
      // Load L for this tile's 2 traversal positions per thread
      // Tile i covers columns i*8..(i+1)*8-1
      // Thread sees columns: traversalOffset + i*8 + morton_x + {0,1}
      // (morton_x is already {0,2,4,6}, NOT {0,1,2,3})
      let tileOff = i * 8
      ir += "  %\(p)tbase_\(i) = add i32 \(traversalOffset), \(tileOff)\n"
      for elem in 0..<2 {
        ir += "  %\(p)mxe_\(i)_\(elem) = add i32 %morton_x, \(elem)\n"
        ir += "  %\(p)lidx_\(i)_\(elem) = add i32 %\(p)tbase_\(i), %\(p)mxe_\(i)_\(elem)\n"
        // Clamp to bounds
        ir += "  %\(p)lcmp_\(i)_\(elem) = icmp ult i32 %\(p)lidx_\(i)_\(elem), \(traversalDim)\n"
        ir += "  %\(p)lsafe_\(i)_\(elem) = select i1 %\(p)lcmp_\(i)_\(elem), i32 %\(p)lidx_\(i)_\(elem), i32 \(traversalDim - 1)\n"
        ir += "  %\(p)loff_\(i)_\(elem) = zext i32 %\(p)lsafe_\(i)_\(elem) to i64\n"
        ir += "  %\(p)lbyte_\(i)_\(elem) = mul i64 %\(p)loff_\(i)_\(elem), \(elemSizeL)\n"
        ir += "  %\(p)lptr_\(i)_\(elem) = getelementptr i8, i8 addrspace(1)* \(lBufferName), i64 %\(p)lbyte_\(i)_\(elem)\n"
        ir += "  %\(p)ltyped_\(i)_\(elem) = bitcast i8 addrspace(1)* %\(p)lptr_\(i)_\(elem) to \(memTypeL) addrspace(1)*\n"
        ir += "  %\(p)lraw_\(i)_\(elem) = load \(memTypeL), \(memTypeL) addrspace(1)* %\(p)ltyped_\(i)_\(elem)\n"
        if memL != .FP32 {
          ir += "  %\(p)L\(elem)_\(i) = fpext \(memTypeL) %\(p)lraw_\(i)_\(elem) to float\n"
        } else {
          ir += "  %\(p)L\(elem)_\(i) = bitcast float %\(p)lraw_\(i)_\(elem) to float\n"
        }
      }

      ir += "  %\(p)s0_\(i) = extractelement \(irVecType(regS)) %\(sSource)_\(i), i32 0\n"
      ir += "  %\(p)s1_\(i) = extractelement \(irVecType(regS)) %\(sSource)_\(i), i32 1\n"

      let s0f = (regS == .FP32) ? "%\(p)s0_\(i)" : "%\(p)s0f_\(i)"
      let s1f = (regS == .FP32) ? "%\(p)s1_\(i)" : "%\(p)s1f_\(i)"
      if regS != .FP32 {
        ir += "  \(s0f) = fpext \(tS) %\(p)s0_\(i) to float\n"
        ir += "  \(s1f) = fpext \(tS) %\(p)s1_\(i) to float\n"
      }

      // P = exp2(S * scale - L)
      // Each tile loads its own L values for the traversal positions it covers
      ir += "  %\(p)sc0_\(i) = fmul fast float \(s0f), \(scaleHex)\n"
      ir += "  %\(p)sh0_\(i) = fsub fast float %\(p)sc0_\(i), %\(p)L0_\(i)\n"
      ir += irExp2Call(result: "%\(p)p0f_\(i)", value: "%\(p)sh0_\(i)") + "\n"

      ir += "  %\(p)sc1_\(i) = fmul fast float \(s1f), \(scaleHex)\n"
      ir += "  %\(p)sh1_\(i) = fsub fast float %\(p)sc1_\(i), %\(p)L1_\(i)\n"
      ir += irExp2Call(result: "%\(p)p1f_\(i)", value: "%\(p)sh1_\(i)") + "\n"

      let p0 = (regP == .FP32) ? "%\(p)p0f_\(i)" : "%\(p)p0_\(i)"
      let p1 = (regP == .FP32) ? "%\(p)p1f_\(i)" : "%\(p)p1_\(i)"
      if regP != .FP32 {
        ir += "  \(p0) = fptrunc float %\(p)p0f_\(i) to \(tP)\n"
        ir += "  \(p1) = fptrunc float %\(p)p1f_\(i) to \(tP)\n"
      }

      ir += "  %\(p)pv0_\(i) = insertelement \(irVecType(regP)) undef, \(tP) \(p0), i32 0\n"
      ir += "  %\(p)p_\(i) = insertelement \(irVecType(regP)) %\(p)pv0_\(i), \(tP) \(p1), i32 1\n"
    }

    return ir
  }

  // MARK: - Derivative Softmax (backward)

  /// dS = P * (dP - D_scalar) * derivScale
  /// Used in backwardQuery where D is a per-row scalar (same for all traversal positions).
  func generateDerivativeSoftmax(
    prefix p: String,
    sSramCount: Int,
    regP: GEMMOperandPrecision, regdP: GEMMOperandPrecision,
    regdS: GEMMOperandPrecision,
    derivScale: Float,
    dScalar: String,  // SSA name of D scalar (float)
    pSource: String,  // SSA prefix for P vectors (e.g., "bq_sf_p" → %bq_sf_p_0)
    dpSource: String  // SSA prefix for dP vectors (e.g., "dp_final" → %dp_final_0)
  ) -> String {
    let tP = irTypeName(regP)
    let tdP = irTypeName(regdP)
    let tdS = irTypeName(regdS)
    var ir = ""

    ir += "  ; === Derivative softmax: dS = P * (dP - D) * scale ===\n"

    let scaleHex = "0x\(String(Double(derivScale).bitPattern, radix: 16, uppercase: true))"

    for i in 0..<sSramCount {
      // Extract P elements
      ir += "  %\(p)p0_\(i) = extractelement \(irVecType(regP)) %\(pSource)_\(i), i32 0\n"
      ir += "  %\(p)p1_\(i) = extractelement \(irVecType(regP)) %\(pSource)_\(i), i32 1\n"

      // Extract dP elements
      ir += "  %\(p)dp0_\(i) = extractelement \(irVecType(regdP)) %\(dpSource)_\(i), i32 0\n"
      ir += "  %\(p)dp1_\(i) = extractelement \(irVecType(regdP)) %\(dpSource)_\(i), i32 1\n"

      // Convert to float
      let p0f = (regP == .FP32) ? "%\(p)p0_\(i)" : "%\(p)p0f_\(i)"
      let p1f = (regP == .FP32) ? "%\(p)p1_\(i)" : "%\(p)p1f_\(i)"
      let dp0f = (regdP == .FP32) ? "%\(p)dp0_\(i)" : "%\(p)dp0f_\(i)"
      let dp1f = (regdP == .FP32) ? "%\(p)dp1_\(i)" : "%\(p)dp1f_\(i)"
      if regP != .FP32 {
        ir += "  \(p0f) = fpext \(tP) %\(p)p0_\(i) to float\n"
        ir += "  \(p1f) = fpext \(tP) %\(p)p1_\(i) to float\n"
      }
      if regdP != .FP32 {
        ir += "  \(dp0f) = fpext \(tdP) %\(p)dp0_\(i) to float\n"
        ir += "  \(dp1f) = fpext \(tdP) %\(p)dp1_\(i) to float\n"
      }

      // dS = P * (dP - D) * derivScale
      ir += "  %\(p)sub0_\(i) = fsub fast float \(dp0f), \(dScalar)\n"
      ir += "  %\(p)pds0_\(i) = fmul fast float \(p0f), %\(p)sub0_\(i)\n"
      ir += "  %\(p)ds0f_\(i) = fmul fast float %\(p)pds0_\(i), \(scaleHex)\n"

      ir += "  %\(p)sub1_\(i) = fsub fast float \(dp1f), \(dScalar)\n"
      ir += "  %\(p)pds1_\(i) = fmul fast float \(p1f), %\(p)sub1_\(i)\n"
      ir += "  %\(p)ds1f_\(i) = fmul fast float %\(p)pds1_\(i), \(scaleHex)\n"

      // Convert to dS precision and construct <64 x T>
      let ds0 = (regdS == .FP32) ? "%\(p)ds0f_\(i)" : "%\(p)ds0_\(i)"
      let ds1 = (regdS == .FP32) ? "%\(p)ds1f_\(i)" : "%\(p)ds1_\(i)"
      if regdS != .FP32 {
        ir += "  \(ds0) = fptrunc float %\(p)ds0f_\(i) to \(tdS)\n"
        ir += "  \(ds1) = fptrunc float %\(p)ds1f_\(i) to \(tdS)\n"
      }

      ir += "  %\(p)dsv0_\(i) = insertelement \(irVecType(regdS)) undef, \(tdS) \(ds0), i32 0\n"
      ir += "  %\(p)ds_\(i) = insertelement \(irVecType(regdS)) %\(p)dsv0_\(i), \(tdS) \(ds1), i32 1\n"
    }

    return ir
  }

  /// dS = P * (dP * derivScale - D_stored)
  /// Used in backwardKeyValue where D varies per-column of S^T.
  /// D_stored is already pre-scaled (D_raw * derivScale) from the bwd_q kernel.
  /// Each sSram tile i covers traversal columns i*8..(i+1)*8-1, so each tile
  /// needs different D values loaded from the D buffer.
  func generateDerivativeSoftmaxVector(
    prefix p: String,
    sSramCount: Int,
    regP: GEMMOperandPrecision, regdP: GEMMOperandPrecision,
    regdS: GEMMOperandPrecision,
    derivScale: Float,
    dBufferName: String,       // D buffer pointer (e.g., "%D_buf")
    traversalOffset: String,   // loop variable for traversal offset (e.g., "%bkv_r")
    traversalDim: UInt32,      // total traversal dimension (R)
    memD: GEMMOperandPrecision,
    pSource: String,  // SSA prefix for P vectors (e.g., "bkv_sf_p" → %bkv_sf_p_0)
    dpSource: String  // SSA prefix for dP vectors (e.g., "dp_final" → %dp_final_0)
  ) -> String {
    let tP = irTypeName(regP)
    let tdP = irTypeName(regdP)
    let tdS = irTypeName(regdS)
    let elemSizeD = UInt32(memD.size)
    let memTypeD = irTypeName(memD)
    var ir = ""

    ir += "  ; === Derivative softmax vector: dS = P * (dP * scale - D_stored) ===\n"

    let scaleHex = "0x\(String(Double(derivScale).bitPattern, radix: 16, uppercase: true))"

    for i in 0..<sSramCount {
      // Load D for this tile's 2 traversal positions per thread
      // Tile i covers columns i*8..(i+1)*8-1
      // Thread sees columns: traversalOffset + i*8 + morton_x + {0,1}
      // (morton_x is already {0,2,4,6}, NOT {0,1,2,3})
      let tileOff = i * 8
      ir += "  %\(p)tbase_\(i) = add i32 \(traversalOffset), \(tileOff)\n"
      for elem in 0..<2 {
        ir += "  %\(p)mxe_\(i)_\(elem) = add i32 %morton_x, \(elem)\n"
        ir += "  %\(p)didx_\(i)_\(elem) = add i32 %\(p)tbase_\(i), %\(p)mxe_\(i)_\(elem)\n"
        // Clamp to bounds
        ir += "  %\(p)dcmp_\(i)_\(elem) = icmp ult i32 %\(p)didx_\(i)_\(elem), \(traversalDim)\n"
        ir += "  %\(p)dsafe_\(i)_\(elem) = select i1 %\(p)dcmp_\(i)_\(elem), i32 %\(p)didx_\(i)_\(elem), i32 \(traversalDim - 1)\n"
        ir += "  %\(p)doff_\(i)_\(elem) = zext i32 %\(p)dsafe_\(i)_\(elem) to i64\n"
        ir += "  %\(p)dbyte_\(i)_\(elem) = mul i64 %\(p)doff_\(i)_\(elem), \(elemSizeD)\n"
        ir += "  %\(p)dptr_\(i)_\(elem) = getelementptr i8, i8 addrspace(1)* \(dBufferName), i64 %\(p)dbyte_\(i)_\(elem)\n"
        ir += "  %\(p)dtyped_\(i)_\(elem) = bitcast i8 addrspace(1)* %\(p)dptr_\(i)_\(elem) to \(memTypeD) addrspace(1)*\n"
        ir += "  %\(p)draw_\(i)_\(elem) = load \(memTypeD), \(memTypeD) addrspace(1)* %\(p)dtyped_\(i)_\(elem)\n"
        if memD != .FP32 {
          ir += "  %\(p)D\(elem)_\(i) = fpext \(memTypeD) %\(p)draw_\(i)_\(elem) to float\n"
        } else {
          ir += "  %\(p)D\(elem)_\(i) = bitcast float %\(p)draw_\(i)_\(elem) to float\n"
        }
      }

      ir += "  %\(p)p0_\(i) = extractelement \(irVecType(regP)) %\(pSource)_\(i), i32 0\n"
      ir += "  %\(p)p1_\(i) = extractelement \(irVecType(regP)) %\(pSource)_\(i), i32 1\n"

      ir += "  %\(p)dp0_\(i) = extractelement \(irVecType(regdP)) %\(dpSource)_\(i), i32 0\n"
      ir += "  %\(p)dp1_\(i) = extractelement \(irVecType(regdP)) %\(dpSource)_\(i), i32 1\n"

      let p0f = (regP == .FP32) ? "%\(p)p0_\(i)" : "%\(p)p0f_\(i)"
      let p1f = (regP == .FP32) ? "%\(p)p1_\(i)" : "%\(p)p1f_\(i)"
      let dp0f = (regdP == .FP32) ? "%\(p)dp0_\(i)" : "%\(p)dp0f_\(i)"
      let dp1f = (regdP == .FP32) ? "%\(p)dp1_\(i)" : "%\(p)dp1f_\(i)"
      if regP != .FP32 {
        ir += "  \(p0f) = fpext \(tP) %\(p)p0_\(i) to float\n"
        ir += "  \(p1f) = fpext \(tP) %\(p)p1_\(i) to float\n"
      }
      if regdP != .FP32 {
        ir += "  \(dp0f) = fpext \(tdP) %\(p)dp0_\(i) to float\n"
        ir += "  \(dp1f) = fpext \(tdP) %\(p)dp1_\(i) to float\n"
      }

      // dS = P * (dP * scale - D_stored)
      // D_stored is already D_raw * derivScale, so scale dP to match
      ir += "  %\(p)sdp0_\(i) = fmul fast float \(dp0f), \(scaleHex)\n"
      ir += "  %\(p)sub0_\(i) = fsub fast float %\(p)sdp0_\(i), %\(p)D0_\(i)\n"
      ir += "  %\(p)ds0f_\(i) = fmul fast float \(p0f), %\(p)sub0_\(i)\n"

      ir += "  %\(p)sdp1_\(i) = fmul fast float \(dp1f), \(scaleHex)\n"
      ir += "  %\(p)sub1_\(i) = fsub fast float %\(p)sdp1_\(i), %\(p)D1_\(i)\n"
      ir += "  %\(p)ds1f_\(i) = fmul fast float \(p1f), %\(p)sub1_\(i)\n"

      let ds0 = (regdS == .FP32) ? "%\(p)ds0f_\(i)" : "%\(p)ds0_\(i)"
      let ds1 = (regdS == .FP32) ? "%\(p)ds1f_\(i)" : "%\(p)ds1_\(i)"
      if regdS != .FP32 {
        ir += "  \(ds0) = fptrunc float %\(p)ds0f_\(i) to \(tdS)\n"
        ir += "  \(ds1) = fptrunc float %\(p)ds1f_\(i) to \(tdS)\n"
      }

      ir += "  %\(p)dsv0_\(i) = insertelement \(irVecType(regdS)) undef, \(tdS) \(ds0), i32 0\n"
      ir += "  %\(p)ds_\(i) = insertelement \(irVecType(regdS)) %\(p)dsv0_\(i), \(tdS) \(ds1), i32 1\n"
    }

    return ir
  }

  // MARK: - Compute D = sum(dO * O)

  /// Compute D = reduce_sum(dO[row,:] * O[row,:])
  /// Both dO and O are loaded from device (or dO from cache).
  /// Result: %D_sram (float scalar per thread, reduced across head dim).
  func generateComputeD(
    prefix p: String,
    D: UInt32, paddedD: UInt32, blockH: UInt16,
    headLoopFloor: UInt32, headEdge: UInt32,
    parallelDim: UInt32,
    cachedO: Bool, cacheddO: Bool,
    leadingDimO: UInt32, leadingDimdO: UInt32,
    memO: GEMMOperandPrecision, regO: GEMMOperandPrecision,
    memdO: GEMMOperandPrecision, regdO: GEMMOperandPrecision,
    transposedO: Bool, transposeddO: Bool
  ) -> String {
    var ir = ""
    ir += "  ; === Compute D = sum(dO * O) ===\n"

    // Accumulate dO * O elementwise across the head dimension
    ir += "  ; D_acc = sum over d of (dO[row,d] * O[row,d])\n"

    var accName = "0.0"  // running float accumulator for element 0+1

    var dOuter: UInt32 = 0
    var iterIdx = 0
    while dOuter < paddedD {
      let regSize = min(UInt32(blockH), paddedD - dOuter)
      let kSteps = Int(regSize / 8)

      for k in 0..<kSteps {
        let kOff = k * 8
        let headStart = Int(dOuter) + kOff
        let ip = "\(p)\(iterIdx)k\(k)_"

        // Skip if beyond D
        if headStart >= D {
          continue
        }

        // Load O from device
        ir += "  %\(ip)o_seq = add i32 %clamped_par_off, 0\n"
        ir += "  %\(ip)o_head = add i32 %morton_x, \(headStart)\n"
        ir += generateDeviceLoad(
          prefix: "\(ip)o_",
          bufferName: "%O",
          seqOffset: "%\(ip)o_seq",
          headOffset: "%\(ip)o_head",
          leadingDim: leadingDimO,
          memPrec: memO,
          regPrec: regO,
          transposed: transposedO
        )

        // Load dO from cache or device
        if cacheddO {
          let cacheIdx = Int(dOuter) / 8 + k
          ir += "  %\(ip)do_sram = bitcast \(irVecType(regdO)) %cdo_sram_\(cacheIdx) to \(irVecType(regdO))\n"
        } else {
          ir += "  %\(ip)do_seq = add i32 %clamped_par_off, 0\n"
          ir += "  %\(ip)do_head = add i32 %morton_x, \(headStart)\n"
          ir += generateDeviceLoad(
            prefix: "\(ip)do_",
            bufferName: "%dO",
            seqOffset: "%\(ip)do_seq",
            headOffset: "%\(ip)do_head",
            leadingDim: leadingDimdO,
            memPrec: memdO,
            regPrec: regdO,
            transposed: transposeddO
          )
        }

        // Extract elements and multiply
        ir += "  %\(ip)o0 = extractelement \(irVecType(regO)) %\(ip)o_sram, i32 0\n"
        ir += "  %\(ip)o1 = extractelement \(irVecType(regO)) %\(ip)o_sram, i32 1\n"
        ir += "  %\(ip)do0 = extractelement \(irVecType(regdO)) %\(ip)do_sram, i32 0\n"
        ir += "  %\(ip)do1 = extractelement \(irVecType(regdO)) %\(ip)do_sram, i32 1\n"

        // Convert to float if needed
        let o0f = (regO == .FP32) ? "%\(ip)o0" : "%\(ip)o0f"
        let o1f = (regO == .FP32) ? "%\(ip)o1" : "%\(ip)o1f"
        let do0f = (regdO == .FP32) ? "%\(ip)do0" : "%\(ip)do0f"
        let do1f = (regdO == .FP32) ? "%\(ip)do1" : "%\(ip)do1f"
        if regO != .FP32 {
          ir += "  \(o0f) = fpext \(irTypeName(regO)) %\(ip)o0 to float\n"
          ir += "  \(o1f) = fpext \(irTypeName(regO)) %\(ip)o1 to float\n"
        }
        if regdO != .FP32 {
          ir += "  \(do0f) = fpext \(irTypeName(regdO)) %\(ip)do0 to float\n"
          ir += "  \(do1f) = fpext \(irTypeName(regdO)) %\(ip)do1 to float\n"
        }

        // Multiply and accumulate
        ir += "  %\(ip)prod0_raw = fmul fast float \(do0f), \(o0f)\n"
        ir += "  %\(ip)prod1_raw = fmul fast float \(do1f), \(o1f)\n"

        // Head masking: zero products for threads whose head position >= D
        // Device load gives positions [morton_x + headStart, morton_x + headStart + 1]
        // morton_x is always even (0, 2, 4, 6, ...), so element 0 is at
        // morton_x + headStart, element 1 is at morton_x + headStart + 1.
        if headStart + 8 > Int(D) {
          ir += "  %\(ip)hpos0 = add i32 %morton_x, \(headStart)\n"
          ir += "  %\(ip)hpos1 = add i32 %morton_x, \(headStart + 1)\n"
          ir += "  %\(ip)oob0 = icmp uge i32 %\(ip)hpos0, \(D)\n"
          ir += "  %\(ip)oob1 = icmp uge i32 %\(ip)hpos1, \(D)\n"
          ir += "  %\(ip)prod0 = select i1 %\(ip)oob0, float 0.0, float %\(ip)prod0_raw\n"
          ir += "  %\(ip)prod1 = select i1 %\(ip)oob1, float 0.0, float %\(ip)prod1_raw\n"
        } else {
          ir += "  %\(ip)prod0 = bitcast float %\(ip)prod0_raw to float\n"
          ir += "  %\(ip)prod1 = bitcast float %\(ip)prod1_raw to float\n"
        }

        ir += "  %\(ip)sum01 = fadd fast float %\(ip)prod0, %\(ip)prod1\n"
        ir += "  %\(ip)acc = fadd fast float \(accName), %\(ip)sum01\n"
        accName = "%\(ip)acc"
      }

      dOuter += UInt32(blockH)
      iterIdx += 1
    }

    // If no iterations happened (D=0?), set to 0
    if accName == "0.0" {
      ir += "  %\(p)final_acc = bitcast float 0.0 to float\n"
      accName = "%\(p)final_acc"
    }

    // SIMD reduction: shuffle_xor with masks 1 and 8
    ir += irShuffleXorCall(result: "%\(p)shuf1", value: accName, mask: 1) + "\n"
    ir += "  %\(p)sum_s1 = fadd fast float \(accName), %\(p)shuf1\n"
    ir += irShuffleXorCall(result: "%\(p)shuf8", value: "%\(p)sum_s1", mask: 8) + "\n"
    ir += "  %\(p)sum_s8 = fadd fast float %\(p)sum_s1, %\(p)shuf8\n"

    // D_sram = raw sum(dO * O) — no scaling here.
    // The derivative softmax formula dS = P * (dP * derivScale - D) handles
    // the scale factor on dP, not on D.
    ir += "  %D_sram = bitcast float %\(p)sum_s8 to float\n"

    return ir
  }

  // MARK: - Load Traversal Scalar (L or D for bkv)

  /// Load a per-row scalar from device memory for each thread's traversal position.
  /// In bkv, the traversal dimension is R, so we load L[r + morton_x*2] and L[r + morton_x*2+1].
  /// Result: a <2 x float> with one L/D value per thread element.
  func generateLoadTraversalScalar(
    prefix p: String,
    bufferName: String,
    resultName: String,
    traversalOffset: String,
    traversalDim: UInt32,
    blockT: UInt16,
    memPrec: GEMMOperandPrecision
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    let memType = irTypeName(memPrec)
    var ir = ""

    ir += "  ; === Load traversal scalar from \(bufferName) ===\n"

    // Each thread loads 2 scalar values for its morton_x positions.
    // Thread position in traversal = traversalOffset + morton_x + {0, 1}
    // (morton_x is already {0,2,4,6}, NOT {0,1,2,3})
    for elem in 0..<2 {
      ir += "  %\(p)mx_\(elem) = add i32 %morton_x, \(elem)\n"
      ir += "  %\(p)idx_\(elem) = add i32 \(traversalOffset), %\(p)mx_\(elem)\n"

      // Clamp to bounds
      ir += "  %\(p)cmp_\(elem) = icmp ult i32 %\(p)idx_\(elem), \(traversalDim)\n"
      ir += "  %\(p)dim_m1_\(elem) = sub i32 \(traversalDim), 1\n"
      ir += "  %\(p)safe_\(elem) = select i1 %\(p)cmp_\(elem), i32 %\(p)idx_\(elem), i32 %\(p)dim_m1_\(elem)\n"

      ir += "  %\(p)off64_\(elem) = zext i32 %\(p)safe_\(elem) to i64\n"
      ir += "  %\(p)byte_\(elem) = mul i64 %\(p)off64_\(elem), \(elemSize)\n"
      ir += "  %\(p)ptr_\(elem) = getelementptr i8, i8 addrspace(1)* \(bufferName), i64 %\(p)byte_\(elem)\n"
      ir += "  %\(p)typed_\(elem) = bitcast i8 addrspace(1)* %\(p)ptr_\(elem) to \(memType) addrspace(1)*\n"
      ir += "  %\(p)raw_\(elem) = load \(memType), \(memType) addrspace(1)* %\(p)typed_\(elem)\n"

      if memPrec != .FP32 {
        ir += "  %\(p)val_\(elem) = fpext \(memType) %\(p)raw_\(elem) to float\n"
      } else {
        ir += "  %\(p)val_\(elem) = bitcast float %\(p)raw_\(elem) to float\n"
      }
    }

    // Construct <2 x float>
    ir += "  %\(p)v0 = insertelement <2 x float> undef, float %\(p)val_0, i32 0\n"
    ir += "  \(resultName) = insertelement <2 x float> %\(p)v0, float %\(p)val_1, i32 1\n"

    return ir
  }

  // MARK: - Cache Store (registers → TG → device)

  /// Store cached registers to device memory via TG + async copy.
  /// Same pattern as generateForwardCleanup O store, but parameterized.
  func generateCacheStore(
    operand: AttentionOperand,
    prefix p: String,
    regCount: Int,
    blockP: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32,
    regPrec: GEMMOperandPrecision, memPrec: GEMMOperandPrecision,
    leadingDim: UInt32, leadingBlockDim: UInt32,
    transposed: Bool,
    phiPrefix: String  // prefix for the phi name of the accumulators (e.g., "dq_phi")
  ) -> String {
    let elemSize = UInt32(memPrec.size)
    let tReg = irTypeName(regPrec)
    var ir = ""

    ir += "  ; === Cache store \(operand) ===\n"

    var dOuter: UInt32 = 0
    var iterIdx = 0
    while dOuter < paddedD {
      let regSize = min(UInt32(blockH), paddedD - dOuter)
      let ip = "\(p)\(iterIdx)_"
      let kSteps = Int(regSize / 8)

      // Store registers to TG
      for k in 0..<kSteps {
        let kOff = k * 8
        let regIdx = Int(dOuter) / 8 + k
        let sp = "\(ip)k\(k)_"

        // Unshuffle from <64 x T> to <2 x T>
        ir += irShuffleFromVec64(
          result: "%\(sp)v2", src: "%\(phiPrefix)_\(regIdx)", type: regPrec
        ) + "\n"

        // Convert precision if needed
        let storeVec: String
        let storeType: String
        if regPrec != memPrec {
          let memT = irTypeName(memPrec)
          ir += "  %\(sp)se0 = extractelement <2 x \(tReg)> %\(sp)v2, i32 0\n"
          ir += "  %\(sp)se1 = extractelement <2 x \(tReg)> %\(sp)v2, i32 1\n"
          ir += "  %\(sp)st0 = fptrunc \(tReg) %\(sp)se0 to \(memT)\n"
          ir += "  %\(sp)st1 = fptrunc \(tReg) %\(sp)se1 to \(memT)\n"
          ir += "  %\(sp)sv0 = insertelement <2 x \(memT)> undef, \(memT) %\(sp)st0, i32 0\n"
          ir += "  %\(sp)svec = insertelement <2 x \(memT)> %\(sp)sv0, \(memT) %\(sp)st1, i32 1\n"
          storeVec = "%\(sp)svec"
          storeType = memT
        } else {
          storeVec = "%\(sp)v2"
          storeType = tReg
        }

        ir += "  %\(sp)in_bounds = icmp ult i32 %unsafe_par_off, \(parallelDim)\n"
        ir += "  br i1 %\(sp)in_bounds, label %\(sp)do_store, label %\(sp)skip_store\n\n"
        ir += "\(sp)do_store:\n"

        if transposed {
          // Transposed: elements 0,1 are at head positions morton_x+kOff and
          // morton_x+kOff+1 for the same seq row (oig_y). In TG[head, seq],
          // these map to non-contiguous addresses. Use two scalar stores.
          let se0: String
          let se1: String
          if storeVec.hasSuffix("svec") {
            se0 = "%\(sp)st0"
            se1 = "%\(sp)st1"
          } else {
            ir += "  %\(sp)se0 = extractelement <2 x \(storeType)> \(storeVec), i32 0\n"
            ir += "  %\(sp)se1 = extractelement <2 x \(storeType)> \(storeVec), i32 1\n"
            se0 = "%\(sp)se0"
            se1 = "%\(sp)se1"
          }
          ir += "  %\(sp)tg_row0 = add i32 %morton_x, \(kOff)\n"
          ir += "  %\(sp)tg_addr0 = mul i32 %\(sp)tg_row0, \(leadingBlockDim)\n"
          ir += "  %\(sp)tg_addr0b = add i32 %\(sp)tg_addr0, %oig_y\n"
          ir += "  %\(sp)tg_byte0 = mul i32 %\(sp)tg_addr0b, \(elemSize)\n"
          ir += "  %\(sp)tg_byte0_64 = zext i32 %\(sp)tg_byte0 to i64\n"
          ir += "  %\(sp)tg_ptr0 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte0_64\n"
          ir += "  %\(sp)tg_typed0 = bitcast i8 addrspace(3)* %\(sp)tg_ptr0 to \(storeType) addrspace(3)*\n"
          ir += "  store \(storeType) \(se0), \(storeType) addrspace(3)* %\(sp)tg_typed0\n"
          ir += "  %\(sp)tg_row1 = add i32 %morton_x, \(kOff + 1)\n"
          ir += "  %\(sp)tg_addr1 = mul i32 %\(sp)tg_row1, \(leadingBlockDim)\n"
          ir += "  %\(sp)tg_addr1b = add i32 %\(sp)tg_addr1, %oig_y\n"
          ir += "  %\(sp)tg_byte1 = mul i32 %\(sp)tg_addr1b, \(elemSize)\n"
          ir += "  %\(sp)tg_byte1_64 = zext i32 %\(sp)tg_byte1 to i64\n"
          ir += "  %\(sp)tg_ptr1 = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte1_64\n"
          ir += "  %\(sp)tg_typed1 = bitcast i8 addrspace(3)* %\(sp)tg_ptr1 to \(storeType) addrspace(3)*\n"
          ir += "  store \(storeType) \(se1), \(storeType) addrspace(3)* %\(sp)tg_typed1\n"
        } else {
          // Non-transposed: elements are at contiguous head positions. Use <2 x T> store.
          ir += "  %\(sp)tg_row = add i32 %oig_y, 0\n"
          ir += "  %\(sp)tg_addr = mul i32 %\(sp)tg_row, \(leadingBlockDim)\n"
          ir += "  %\(sp)tg_col = add i32 %morton_x, \(kOff)\n"
          ir += "  %\(sp)tg_addr2 = add i32 %\(sp)tg_addr, %\(sp)tg_col\n"
          ir += "  %\(sp)tg_byte = mul i32 %\(sp)tg_addr2, \(elemSize)\n"
          ir += "  %\(sp)tg_byte64 = zext i32 %\(sp)tg_byte to i64\n"
          ir += "  %\(sp)tg_ptr = getelementptr i8, i8 addrspace(3)* %tg_base, i64 %\(sp)tg_byte64\n"
          ir += "  %\(sp)tg_typed = bitcast i8 addrspace(3)* %\(sp)tg_ptr to <2 x \(storeType)> addrspace(3)*\n"
          ir += "  store <2 x \(storeType)> \(storeVec), <2 x \(storeType)> addrspace(3)* %\(sp)tg_typed\n"
        }

        ir += "  br label %\(sp)skip_store\n\n"
        ir += "\(sp)skip_store:\n"
      }

      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n"

      // Async copy TG → device
      let cp = "\(ip)cp_"
      ir += "  br i1 %is_sidx0, label %\(cp)do, label %\(cp)skip\n\n"
      ir += "\(cp)do:\n"

      if transposed {
        ir += "  %\(cp)dev_row = mul i32 \(dOuter), \(leadingDim)\n"
        ir += "  %\(cp)dev_off32 = add i32 %\(cp)dev_row, %par_group_off\n"
      } else {
        ir += "  %\(cp)dev_row = mul i32 %par_group_off, \(leadingDim)\n"
        ir += "  %\(cp)dev_off32 = add i32 %\(cp)dev_row, \(dOuter)\n"
      }
      ir += "  %\(cp)dev_off = zext i32 %\(cp)dev_off32 to i64\n"
      ir += "  %\(cp)dev_byte = mul i64 %\(cp)dev_off, \(elemSize)\n"
      ir += "  %\(cp)dst_p = getelementptr i8, i8 addrspace(1)* %\(operand), i64 %\(cp)dev_byte\n"
      ir += "  %\(cp)src_p = getelementptr i8, i8 addrspace(3)* %tg_base, i64 0\n"

      let dTile = min(UInt32(blockH), D - min(dOuter, D))
      ir += "  %\(cp)seq_rem = sub i32 \(parallelDim), %par_group_off\n"
      ir += "  %\(cp)seq_cmp = icmp ult i32 %\(cp)seq_rem, \(blockP)\n"
      ir += "  %\(cp)seq_tile32 = select i1 %\(cp)seq_cmp, i32 %\(cp)seq_rem, i32 \(blockP)\n"
      ir += "  %\(cp)seq_tile = zext i32 %\(cp)seq_tile32 to i64\n"

      let dstStride = leadingDim * elemSize
      let srcStride = leadingBlockDim * elemSize

      let (tileW, tileH): (String, String)
      if transposed {
        ir += "  %\(cp)w_bytes32 = mul i32 %\(cp)seq_tile32, \(elemSize)\n"
        ir += "  %\(cp)w_bytes = zext i32 %\(cp)w_bytes32 to i64\n"
        tileW = "%\(cp)w_bytes"
        tileH = "\(dTile)"
      } else {
        tileW = "\(dTile * elemSize)"
        tileH = "%\(cp)seq_tile"
      }

      ir += "  %\(cp)tile_w = insertelement <2 x i64> zeroinitializer, i64 \(tileW), i32 0\n"
      ir += "  %\(cp)tile = insertelement <2 x i64> %\(cp)tile_w, i64 \(tileH), i32 1\n"

      ir += "  %\(cp)evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* %ev, i64 0, i64 0\n"
      ir += """
        %\(cp)ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
          i64 1, i64 1,
          i8 addrspace(1)* %\(cp)dst_p, i64 \(dstStride), i64 1, <2 x i64> %\(cp)tile,
          i8 addrspace(3)* %\(cp)src_p, i64 \(srcStride), i64 1, <2 x i64> %\(cp)tile,
          <2 x i64> zeroinitializer, i32 0
        )
        store %event_t addrspace(3)* %\(cp)ev, %event_t addrspace(3)** %\(cp)evp
        call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %\(cp)evp)

      """

      ir += "  br label %\(cp)skip\n\n"
      ir += "\(cp)skip:\n"
      ir += "  call void @air.wg.barrier(i32 2, i32 1)\n\n"

      dOuter += UInt32(blockH)
      iterIdx += 1
    }

    return ir
  }

  // MARK: - Backward Query Cleanup

  func generateBackwardQueryCleanup(
    prefix p: String,
    dqCount: Int,
    blockP: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32,
    regdQ: GEMMOperandPrecision, memdQ: GEMMOperandPrecision,
    memD: GEMMOperandPrecision,
    leadingDimdQ: UInt32, leadingBlockDimdQ: UInt32,
    transposeddQ: Bool,
    cacheddQ: Bool,
    derivScale: Float
  ) -> String {
    let elemSizeD = UInt32(memD.size)
    var ir = ""

    ir += """

    bq_cleanup:
      ; === Backward Query cleanup: store dQ, store D ===

    """

    // Store dQ via cache store
    ir += generateCacheStore(
      operand: .dQ,
      prefix: "\(p)q",
      regCount: dqCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regPrec: regdQ, memPrec: memdQ,
      leadingDim: leadingDimdQ, leadingBlockDim: leadingBlockDimdQ,
      transposed: transposeddQ,
      phiPrefix: "dq_phi"
    )

    // Store D scalar to device (scaled by derivScale for compatibility with
    // the reference Metal source, which stores D * derivScale and uses
    // dS = P * (dP * scale - D_stored). Our kernel uses D_sram raw internally
    // with dS = P * (dP - D_sram) * scale, but the test expects scaled D.)
    let dScaleHex = "0x\(String(Double(derivScale).bitPattern, radix: 16, uppercase: true))"
    ir += "  ; Store D (scaled by derivScale)\n"
    ir += "  %\(p)D_scaled = fmul fast float %D_sram, \(dScaleHex)\n"
    ir += "  %\(p)D_in_bounds = icmp ult i32 %unsafe_par_off, \(parallelDim)\n"
    ir += "  br i1 %\(p)D_in_bounds, label %\(p)store_D, label %\(p)skip_D\n\n"

    ir += "\(p)store_D:\n"
    ir += "  %\(p)D_off = zext i32 %clamped_par_off to i64\n"
    ir += "  %\(p)D_byte = mul i64 %\(p)D_off, \(elemSizeD)\n"
    ir += "  %\(p)D_ptr = getelementptr i8, i8 addrspace(1)* %D_buf, i64 %\(p)D_byte\n"
    if memD == .FP32 {
      ir += "  %\(p)D_typed = bitcast i8 addrspace(1)* %\(p)D_ptr to float addrspace(1)*\n"
      ir += "  store float %\(p)D_scaled, float addrspace(1)* %\(p)D_typed\n"
    } else {
      let tD = irTypeName(memD)
      ir += "  %\(p)D_trunc = fptrunc float %\(p)D_scaled to \(tD)\n"
      ir += "  %\(p)D_typed = bitcast i8 addrspace(1)* %\(p)D_ptr to \(tD) addrspace(1)*\n"
      ir += "  store \(tD) %\(p)D_trunc, \(tD) addrspace(1)* %\(p)D_typed\n"
    }
    ir += "  br label %\(p)skip_D\n\n"

    ir += "\(p)skip_D:\n"
    ir += "  br label %exit\n"

    return ir
  }

  // MARK: - Backward Key-Value Cleanup

  func generateBackwardKeyValueCleanup(
    prefix p: String,
    dkCount: Int, dvCount: Int,
    blockP: UInt16, blockH: UInt16,
    D: UInt32, paddedD: UInt32, headEdge: UInt32,
    headLoopFloor: UInt32,
    parallelDim: UInt32,
    regdK: GEMMOperandPrecision, memdK: GEMMOperandPrecision,
    regdV: GEMMOperandPrecision, memdV: GEMMOperandPrecision,
    leadingDimdK: UInt32, leadingBlockDimdK: UInt32,
    leadingDimdV: UInt32, leadingBlockDimdV: UInt32,
    transposeddK: Bool, transposeddV: Bool,
    cacheddK: Bool, cacheddV: Bool
  ) -> String {
    var ir = ""

    ir += """

    bkv_cleanup:
      ; === Backward Key-Value cleanup: store dK, store dV ===

    """

    // Store dV via cache store
    ir += generateCacheStore(
      operand: .dV,
      prefix: "\(p)v",
      regCount: dvCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regPrec: regdV, memPrec: memdV,
      leadingDim: leadingDimdV, leadingBlockDim: leadingBlockDimdV,
      transposed: transposeddV,
      phiPrefix: "dv_phi"
    )

    // Store dK via cache store
    ir += generateCacheStore(
      operand: .dK,
      prefix: "\(p)k",
      regCount: dkCount,
      blockP: blockP, blockH: blockH,
      D: D, paddedD: paddedD, headEdge: headEdge,
      headLoopFloor: headLoopFloor,
      parallelDim: parallelDim,
      regPrec: regdK, memPrec: memdK,
      leadingDim: leadingDimdK, leadingBlockDim: leadingBlockDimdK,
      transposed: transposeddK,
      phiPrefix: "dk_phi"
    )

    ir += "  br label %exit\n"

    return ir
  }
}
