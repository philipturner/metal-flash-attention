//
//  IRTemplate.swift
//  FlashAttention
//
//  Parameterized helpers for generating LLVM IR text for monolithic
//  Metal compute kernels assembled via MetalASM.
//
//  These templates replace the shell+visible-function reverse-linking
//  pattern with a single monolithic kernel containing dispatch loop,
//  async copy, and compute body inlined together.
//

// MARK: - Module Structure

/// Generate the LLVM IR module header: datalayout, triple, event_t type,
/// and threadgroup buffer global.
///
/// - Parameter tgBufferFloats: Size of threadgroup buffer in 32-bit words.
func irModuleHeader(tgBufferFloats: Int = 8192) -> String {
  _ = tgBufferFloats
  return """
  source_filename = "monolithic_gemm"
  target datalayout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32"
  target triple = "air64_v28-apple-macosx26.0.0"

  %event_t = type opaque
  """
}

/// Generate intrinsic declarations for async copy, barrier, wait, and lifetime.
func irIntrinsicDeclarations() -> String {
  """

  declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(i64, i64, i8 addrspace(3)*, i64, i64, <2 x i64>, i8 addrspace(1)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
  declare %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(i64, i64, i8 addrspace(1)*, i64, i64, <2 x i64>, i8 addrspace(3)*, i64, i64, <2 x i64>, <2 x i64>, i32) #1
  declare void @air.wait_simdgroup_events(i32, %event_t addrspace(3)**) #1
  declare void @air.wg.barrier(i32, i32) #1
  declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #2
  declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #2
  """
}

// MARK: - Multiply-Accumulate Intrinsics

/// The LLVM IR type name for a precision.
func irTypeName(_ precision: GEMMOperandPrecision) -> String {
  switch precision {
  case .FP32: return "float"
  case .FP16: return "half"
  case .BF16: return "bfloat"
  }
}

/// The LLVM IR vector type for a 64-element simdgroup matrix.
func irVecType(_ precision: GEMMOperandPrecision) -> String {
  return "<64 x \(irTypeName(precision))>"
}

/// The mangled suffix for a simdgroup matrix vector type.
func irVecSuffix(_ precision: GEMMOperandPrecision) -> String {
  switch precision {
  case .FP32: return "v64f32"
  case .FP16: return "v64f16"
  case .BF16: return "v64bf16"
  }
}

/// Generate the multiply_accumulate intrinsic declaration for given precisions.
///
/// - Parameters:
///   - A: Precision of left operand
///   - B: Precision of right operand
///   - C: Precision of accumulator (also return type)
func irMultiplyAccumulateDeclaration(
  A: GEMMOperandPrecision,
  B: GEMMOperandPrecision,
  C: GEMMOperandPrecision
) -> String {
  let retSuffix = irVecSuffix(C)
  let aSuffix = irVecSuffix(A)
  let bSuffix = irVecSuffix(B)
  let cSuffix = irVecSuffix(C)
  let name = "@air.simdgroup_matrix_8x8_multiply_accumulate.\(retSuffix).\(aSuffix).\(bSuffix).\(cSuffix)"
  return "declare \(irVecType(C)) \(name)(\(irVecType(A)), \(irVecType(B)), \(irVecType(C))) local_unnamed_addr #1"
}

/// Generate a call to the multiply_accumulate intrinsic.
///
/// Returns the SSA name of the result.
func irMultiplyAccumulateCall(
  result: String,
  A: (name: String, precision: GEMMOperandPrecision),
  B: (name: String, precision: GEMMOperandPrecision),
  C: (name: String, precision: GEMMOperandPrecision)
) -> String {
  let retSuffix = irVecSuffix(C.precision)
  let aSuffix = irVecSuffix(A.precision)
  let bSuffix = irVecSuffix(B.precision)
  let cSuffix = irVecSuffix(C.precision)
  let intrinsicName = "@air.simdgroup_matrix_8x8_multiply_accumulate.\(retSuffix).\(aSuffix).\(bSuffix).\(cSuffix)"
  return "  \(result) = tail call fast \(irVecType(C.precision)) \(intrinsicName)(\(irVecType(A.precision)) \(A.name), \(irVecType(B.precision)) \(B.name), \(irVecType(C.precision)) \(C.name)) #3"
}

// MARK: - Vector Element Transfer Patterns

/// Generate IR to place 2 elements from a <2 x T> into a <64 x T> vector.
/// Uses insertelement (avoids shufflevector with vector constant masks
/// that MetalASM's parser doesn't support).
///
/// Produces 3 lines of IR: extract e0, extract e1, insert e0 at 0, insert e1 at 1.
/// The result name gets `_e0`, `_e1`, `_v0` suffixes for intermediates.
func irShuffleToVec64(result: String, src: String, type: GEMMOperandPrecision) -> String {
  let t = irTypeName(type)
  let v64 = "<64 x \(t)>"
  var ir = ""
  ir += "  \(result)_e0 = extractelement <2 x \(t)> \(src), i32 0\n"
  ir += "  \(result)_e1 = extractelement <2 x \(t)> \(src), i32 1\n"
  ir += "  \(result)_v0 = insertelement \(v64) undef, \(t) \(result)_e0, i32 0\n"
  ir += "  \(result) = insertelement \(v64) \(result)_v0, \(t) \(result)_e1, i32 1"
  return ir
}

/// Generate IR to extract first 2 elements from a <64 x T> into a <2 x T>.
/// Uses extractelement + insertelement.
func irShuffleFromVec64(result: String, src: String, type: GEMMOperandPrecision) -> String {
  let t = irTypeName(type)
  var ir = ""
  ir += "  \(result)_e0 = extractelement <64 x \(t)> \(src), i32 0\n"
  ir += "  \(result)_e1 = extractelement <64 x \(t)> \(src), i32 1\n"
  ir += "  \(result)_v0 = insertelement <2 x \(t)> undef, \(t) \(result)_e0, i32 0\n"
  ir += "  \(result) = insertelement <2 x \(t)> \(result)_v0, \(t) \(result)_e1, i32 1"
  return ir
}

/// Generate IR to create a named zero <64 x T> vector via bitcast.
/// Returns a full instruction line (no leading spaces, no trailing newline).
/// Usage: `ir += "  " + irZeroVec64(result: "%c_sram_0", precision: .FP32) + "\n"`
func irZeroVec64(result: String, precision: GEMMOperandPrecision) -> String {
  let v = irVecType(precision)
  return "\(result) = bitcast \(v) zeroinitializer to \(v)"
}

// MARK: - Address Space Helpers

/// Address spaces in Metal AIR.
enum IRAddressSpace: Int {
  case thread_ = 0
  case device = 1
  case constant = 2
  case threadgroup = 3
}

/// Generate a GEP (getelementptr) for an i32 index into threadgroup cmd area.
func irCmdLoad(result: String, cmdBase: String, index: Int) -> String {
  """
    \(result)_ptr = getelementptr i32, i32 addrspace(3)* \(cmdBase), i64 \(index)
    \(result) = load i32, i32 addrspace(3)* \(result)_ptr
  """
}

// MARK: - Dual Async Copy Block

/// Generate the dual async copy block (A+B from device to threadgroup).
/// This reads copy parameters from the threadgroup command area and executes
/// two `air.simdgroup_async_copy_2d` calls.
///
/// - Parameters:
///   - bufferA: SSA name of buffer A (i8 addrspace(1)*)
///   - bufferB: SSA name of buffer B (i8 addrspace(1)*)
///   - evAlloca: SSA name of the event array alloca
///   - tgBase: SSA name of threadgroup base pointer (i8 addrspace(3)*)
///   - tgCmd: SSA name of threadgroup command pointer (i32 addrspace(3)*)
///   - resumeResult: SSA name for the next resume point value
///   - resumeInput: SSA name of the current resume point
func irDualCopyBlock(
  label: String = "do_dual_copy",
  bufferA: String = "%A",
  bufferB: String = "%B",
  evAlloca: String = "%ev",
  tgBase: String = "%tg_base",
  tgCmd: String = "%tg_cmd",
  resumeResult: String = "%nr_d",
  resumeInput: String = "%resume_point"
) -> String {
  """
  \(label):
    ; ── A copy params ──
    %a1 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 1
    %a_dst_s32 = load i32, i32 addrspace(3)* %a1
    %a_dst_s = zext i32 %a_dst_s32 to i64
    %a2 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 2
    %a_src_s32 = load i32, i32 addrspace(3)* %a2
    %a_src_s = zext i32 %a_src_s32 to i64
    %a3 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 3
    %a_stw32 = load i32, i32 addrspace(3)* %a3
    %a_stw = zext i32 %a_stw32 to i64
    %a4 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 4
    %a_sth32 = load i32, i32 addrspace(3)* %a4
    %a_sth = zext i32 %a_sth32 to i64
    %a5 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 5
    %a_off_lo = load i32, i32 addrspace(3)* %a5
    %a6 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 6
    %a_off_hi = load i32, i32 addrspace(3)* %a6
    %a_off_lo64 = zext i32 %a_off_lo to i64
    %a_off_hi64 = zext i32 %a_off_hi to i64
    %a_off_hi_s = shl i64 %a_off_hi64, 32
    %a_off = or i64 %a_off_lo64, %a_off_hi_s
    %a7 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 7
    %a_tg_off32 = load i32, i32 addrspace(3)* %a7
    %a_tg_off = zext i32 %a_tg_off32 to i64
    %a21 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 21
    %a_dtw32 = load i32, i32 addrspace(3)* %a21
    %a_dtw = zext i32 %a_dtw32 to i64
    %a22 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 22
    %a_dth32 = load i32, i32 addrspace(3)* %a22
    %a_dth = zext i32 %a_dth32 to i64
    %a_sv0 = insertelement <2 x i64> zeroinitializer, i64 %a_stw, i32 0
    %a_stile = insertelement <2 x i64> %a_sv0, i64 %a_sth, i32 1
    %a_dv0 = insertelement <2 x i64> zeroinitializer, i64 %a_dtw, i32 0
    %a_dtile = insertelement <2 x i64> %a_dv0, i64 %a_dth, i32 1
    %a_src_p = getelementptr i8, i8 addrspace(1)* \(bufferA), i64 %a_off
    %a_dst_p = getelementptr i8, i8 addrspace(3)* \(tgBase), i64 %a_tg_off
    %ev0p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* \(evAlloca), i64 0, i64 0
    %a_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1,
      i8 addrspace(3)* %a_dst_p, i64 %a_dst_s, i64 1, <2 x i64> %a_dtile,
      i8 addrspace(1)* %a_src_p, i64 %a_src_s, i64 1, <2 x i64> %a_stile,
      <2 x i64> zeroinitializer, i32 0
    )
    store %event_t addrspace(3)* %a_ev, %event_t addrspace(3)** %ev0p
    ; ── B copy ──
    %b11 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 11
    %b_dst_s32 = load i32, i32 addrspace(3)* %b11
    %b_dst_s = zext i32 %b_dst_s32 to i64
    %b12 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 12
    %b_src_s32 = load i32, i32 addrspace(3)* %b12
    %b_src_s = zext i32 %b_src_s32 to i64
    %b13 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 13
    %b_stw32 = load i32, i32 addrspace(3)* %b13
    %b_stw = zext i32 %b_stw32 to i64
    %b14 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 14
    %b_sth32 = load i32, i32 addrspace(3)* %b14
    %b_sth = zext i32 %b_sth32 to i64
    %b15 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 15
    %b_off_lo = load i32, i32 addrspace(3)* %b15
    %b16 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 16
    %b_off_hi = load i32, i32 addrspace(3)* %b16
    %b_off_lo64 = zext i32 %b_off_lo to i64
    %b_off_hi64 = zext i32 %b_off_hi to i64
    %b_off_hi_s = shl i64 %b_off_hi64, 32
    %b_off = or i64 %b_off_lo64, %b_off_hi_s
    %b17 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 17
    %b_tg_off32 = load i32, i32 addrspace(3)* %b17
    %b_tg_off = zext i32 %b_tg_off32 to i64
    %b23 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 23
    %b_dtw32 = load i32, i32 addrspace(3)* %b23
    %b_dtw = zext i32 %b_dtw32 to i64
    %b24 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 24
    %b_dth32 = load i32, i32 addrspace(3)* %b24
    %b_dth = zext i32 %b_dth32 to i64
    %b_sv0 = insertelement <2 x i64> zeroinitializer, i64 %b_stw, i32 0
    %b_stile = insertelement <2 x i64> %b_sv0, i64 %b_sth, i32 1
    %b_dv0 = insertelement <2 x i64> zeroinitializer, i64 %b_dtw, i32 0
    %b_dtile = insertelement <2 x i64> %b_dv0, i64 %b_dth, i32 1
    %b_src_p = getelementptr i8, i8 addrspace(1)* \(bufferB), i64 %b_off
    %b_dst_p = getelementptr i8, i8 addrspace(3)* \(tgBase), i64 %b_tg_off
    %ev1p = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* \(evAlloca), i64 0, i64 1
    %b_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1,
      i8 addrspace(3)* %b_dst_p, i64 %b_dst_s, i64 1, <2 x i64> %b_dtile,
      i8 addrspace(1)* %b_src_p, i64 %b_src_s, i64 1, <2 x i64> %b_stile,
      <2 x i64> zeroinitializer, i32 0
    )
    store %event_t addrspace(3)* %b_ev, %event_t addrspace(3)** %ev1p
    ; Wait for both
    call void @air.wait_simdgroup_events(i32 2, %event_t addrspace(3)** %ev0p)
    call void @air.wg.barrier(i32 2, i32 1)
    \(resumeResult) = add i32 \(resumeInput), 1
    br label %after_async
  """
}

// MARK: - Single Load Block

/// Generate the single async copy (device → threadgroup) block.
func irSingleLoadBlock(
  label: String = "do_single_load",
  buffer: String = "%C",
  evAlloca: String = "%ev",
  tgBase: String = "%tg_base",
  tgCmd: String = "%tg_cmd",
  resumeResult: String = "%nr_l",
  resumeInput: String = "%resume_point"
) -> String {
  """
  \(label):
    %l1 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 1
    %l_dst_s32 = load i32, i32 addrspace(3)* %l1
    %l_dst_s = zext i32 %l_dst_s32 to i64
    %l2 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 2
    %l_src_s32 = load i32, i32 addrspace(3)* %l2
    %l_src_s = zext i32 %l_src_s32 to i64
    %l3 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 3
    %l_tw32 = load i32, i32 addrspace(3)* %l3
    %l_tw = zext i32 %l_tw32 to i64
    %l4 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 4
    %l_th32 = load i32, i32 addrspace(3)* %l4
    %l_th = zext i32 %l_th32 to i64
    %l5 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 5
    %l_off_lo = load i32, i32 addrspace(3)* %l5
    %l6 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 6
    %l_off_hi = load i32, i32 addrspace(3)* %l6
    %l_off_lo64 = zext i32 %l_off_lo to i64
    %l_off_hi64 = zext i32 %l_off_hi to i64
    %l_off_hi_s = shl i64 %l_off_hi64, 32
    %l_off = or i64 %l_off_lo64, %l_off_hi_s
    %l7 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 7
    %l_tg_off32 = load i32, i32 addrspace(3)* %l7
    %l_tg_off = zext i32 %l_tg_off32 to i64
    %l_src_p = getelementptr i8, i8 addrspace(1)* \(buffer), i64 %l_off
    %l_dst_p = getelementptr i8, i8 addrspace(3)* \(tgBase), i64 %l_tg_off
    %l_v0 = insertelement <2 x i64> zeroinitializer, i64 %l_tw, i32 0
    %l_tile = insertelement <2 x i64> %l_v0, i64 %l_th, i32 1
    %l_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* \(evAlloca), i64 0, i64 0
    %l_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p3i8.p1i8(
      i64 1, i64 1,
      i8 addrspace(3)* %l_dst_p, i64 %l_dst_s, i64 1, <2 x i64> %l_tile,
      i8 addrspace(1)* %l_src_p, i64 %l_src_s, i64 1, <2 x i64> %l_tile,
      <2 x i64> zeroinitializer, i32 0
    )
    store %event_t addrspace(3)* %l_ev, %event_t addrspace(3)** %l_evp
    call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %l_evp)
    call void @air.wg.barrier(i32 2, i32 1)
    \(resumeResult) = add i32 \(resumeInput), 1
    br label %after_async
  """
}

// MARK: - Single Store Block

/// Generate the single async copy (threadgroup → device) block.
func irSingleStoreBlock(
  label: String = "do_single_store",
  buffer: String = "%C",
  evAlloca: String = "%ev",
  tgBase: String = "%tg_base",
  tgCmd: String = "%tg_cmd",
  resumeResult: String = "%nr_s",
  resumeInput: String = "%resume_point"
) -> String {
  """
  \(label):
    %s1 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 1
    %s_dst_s32 = load i32, i32 addrspace(3)* %s1
    %s_dst_s = zext i32 %s_dst_s32 to i64
    %s2 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 2
    %s_src_s32 = load i32, i32 addrspace(3)* %s2
    %s_src_s = zext i32 %s_src_s32 to i64
    %s3 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 3
    %s_tw32 = load i32, i32 addrspace(3)* %s3
    %s_tw = zext i32 %s_tw32 to i64
    %s4 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 4
    %s_th32 = load i32, i32 addrspace(3)* %s4
    %s_th = zext i32 %s_th32 to i64
    %s5 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 5
    %s_off_lo = load i32, i32 addrspace(3)* %s5
    %s6 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 6
    %s_off_hi = load i32, i32 addrspace(3)* %s6
    %s_off_lo64 = zext i32 %s_off_lo to i64
    %s_off_hi64 = zext i32 %s_off_hi to i64
    %s_off_hi_s = shl i64 %s_off_hi64, 32
    %s_off = or i64 %s_off_lo64, %s_off_hi_s
    %s7 = getelementptr i32, i32 addrspace(3)* \(tgCmd), i64 7
    %s_tg_off32 = load i32, i32 addrspace(3)* %s7
    %s_tg_off = zext i32 %s_tg_off32 to i64
    %s_dst_p = getelementptr i8, i8 addrspace(1)* \(buffer), i64 %s_off
    %s_src_p = getelementptr i8, i8 addrspace(3)* \(tgBase), i64 %s_tg_off
    %s_v0 = insertelement <2 x i64> zeroinitializer, i64 %s_tw, i32 0
    %s_tile = insertelement <2 x i64> %s_v0, i64 %s_th, i32 1
    %s_evp = getelementptr [2 x %event_t addrspace(3)*], [2 x %event_t addrspace(3)*]* \(evAlloca), i64 0, i64 0
    %s_ev = call %event_t addrspace(3)* @air.simdgroup_async_copy_2d.p1i8.p3i8(
      i64 1, i64 1,
      i8 addrspace(1)* %s_dst_p, i64 %s_dst_s, i64 1, <2 x i64> %s_tile,
      i8 addrspace(3)* %s_src_p, i64 %s_src_s, i64 1, <2 x i64> %s_tile,
      <2 x i64> zeroinitializer, i32 0
    )
    store %event_t addrspace(3)* %s_ev, %event_t addrspace(3)** %s_evp
    call void @air.wait_simdgroup_events(i32 1, %event_t addrspace(3)** %s_evp)
    call void @air.wg.barrier(i32 2, i32 1)
    \(resumeResult) = add i32 \(resumeInput), 1
    br label %after_async
  """
}

// MARK: - Dispatch Loop Skeleton

/// Generate the dispatch loop that sits between compute blocks and
/// async copy blocks. Reads cmd[0] and branches to the appropriate
/// copy block or exit.
///
/// The compute body blocks should branch to `dispatch_loop` when they
/// need an async copy, after writing params to the cmd area.
func irDispatchLoopAndAfterAsync() -> String {
  """

  after_async:
    %next_resume = phi i32 [%nr_d, %do_dual_copy], [%nr_l, %do_single_load], [%nr_s, %do_single_store]
    br label %dispatch_loop
  """
}

// MARK: - Kernel Metadata

/// Generate metadata for a GEMM kernel with 3 device buffers (A, B, C)
/// plus system values (gid, sidx, lane_id).
///
/// - Parameters:
///   - kernelName: The kernel function name
///   - functionRef: The LLVM IR function signature type string
func irGEMMKernelMetadata(kernelName: String = "gemm") -> String {
  """

  attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="96" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
  attributes #1 = { convergent mustprogress nounwind willreturn }
  attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
  attributes #3 = { convergent nounwind willreturn }
  attributes #4 = { nounwind }

  !air.kernel = !{!0}
  !llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
  !air.compile_options = !{!15, !16, !17}
  !llvm.ident = !{!19}
  !air.version = !{!20}
  !air.language_version = !{!21}
  !air.source_file_name = !{!22}

  !0 = !{void (i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(1)*, i8 addrspace(3)*, <3 x i32>, i16, i16)* @\(kernelName), !1, !2}
  !1 = !{}
  !2 = !{!3, !4, !5, !31, !30, !6, !7}
  !3 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"A"}
  !4 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"B"}
  !5 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"C"}
  !31 = !{i32 3, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"tg_mem"}
  !30 = !{i32 4, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
  !6 = !{i32 5, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
  !7 = !{i32 6, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}

  !8 = !{i32 1, !"wchar_size", i32 4}
  !9 = !{i32 7, !"air.max_device_buffers", i32 31}
  !10 = !{i32 7, !"air.max_constant_buffers", i32 31}
  !11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
  !12 = !{i32 7, !"air.max_textures", i32 128}
  !13 = !{i32 7, !"air.max_read_write_textures", i32 8}
  !14 = !{i32 7, !"air.max_samplers", i32 16}
  !15 = !{!"air.compile.denorms_disable"}
  !16 = !{!"air.compile.fast_math_enable"}
  !17 = !{!"air.compile.framebuffer_fetch_enable"}
  !19 = !{!"MetalASM (monolithic GEMM)"}
  !20 = !{i32 2, i32 8, i32 0}
  !21 = !{!"Metal", i32 4, i32 0, i32 0}
  !22 = !{!"monolithic_gemm.ll"}
  """
}

// MARK: - Attention Intrinsics

/// Additional intrinsic declarations needed for attention kernels:
/// simd_shuffle_xor, exp2, log2.
func irAttentionIntrinsicDeclarations() -> String {
  """

  declare float @air.simd_shuffle_xor.f32(float, i32) #1
  declare float @llvm.exp2.f32(float) #1
  declare float @llvm.log2.f32(float) #1
  """
}

/// Generate a simd_shuffle_xor call.
func irShuffleXorCall(
  result: String, value: String, mask: Int
) -> String {
  "  \(result) = call float @air.simd_shuffle_xor.f32(float \(value), i32 \(mask))"
}

/// Generate a fast exp2 call.
func irExp2Call(result: String, value: String) -> String {
  "  \(result) = call fast float @llvm.exp2.f32(float \(value))"
}

/// Generate a fast log2 call.
func irLog2Call(result: String, value: String) -> String {
  "  \(result) = call fast float @llvm.log2.f32(float \(value))"
}

// MARK: - Attention Kernel Metadata

/// Generate metadata for an attention kernel with 10 device buffers
/// (Q, K, V, O, L, D, dO, dV, dK, dQ) plus 1 TG buffer + system values.
///
/// Buffer bindings:
///   0: Q (read), 1: K (read), 2: V (read), 3: O (read_write),
///   4: L (read_write), 5: D (read_write),
///   6: dO (read), 7: dV (read_write), 8: dK (read_write), 9: dQ (read_write)
func irAttentionKernelMetadata(kernelName: String = "attention") -> String {
  // Build the function type signature: 10 device buffers + 1 TG + gid + sidx + lane_id
  let ptrArgs = (0..<10).map { _ in "i8 addrspace(1)*" }.joined(separator: ", ")
  let fnType = "void (\(ptrArgs), i8 addrspace(3)*, <3 x i32>, i16, i16)"

  return """

  attributes #0 = { convergent mustprogress nounwind willreturn "frame-pointer"="none" "min-legal-vector-width"="96" "no-builtins" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
  attributes #1 = { convergent mustprogress nounwind willreturn }
  attributes #2 = { argmemonly mustprogress nocallback nofree nosync nounwind willreturn }
  attributes #3 = { convergent nounwind willreturn }
  attributes #4 = { nounwind }

  !air.kernel = !{!0}
  !llvm.module.flags = !{!8, !9, !10, !11, !12, !13, !14}
  !air.compile_options = !{!15, !16, !17}
  !llvm.ident = !{!19}
  !air.version = !{!20}
  !air.language_version = !{!21}
  !air.source_file_name = !{!22}

  !0 = !{\(fnType)* @\(kernelName), !1, !2}
  !1 = !{}
  !2 = !{!30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43}
  !30 = !{i32 0, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"Q"}
  !31 = !{i32 1, !"air.buffer", !"air.location_index", i32 1, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"K"}
  !32 = !{i32 2, !"air.buffer", !"air.location_index", i32 2, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"V"}
  !33 = !{i32 3, !"air.buffer", !"air.location_index", i32 3, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"O"}
  !34 = !{i32 4, !"air.buffer", !"air.location_index", i32 4, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"L"}
  !35 = !{i32 5, !"air.buffer", !"air.location_index", i32 5, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"D_buf"}
  !36 = !{i32 6, !"air.buffer", !"air.location_index", i32 6, i32 1, !"air.read", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dO"}
  !37 = !{i32 7, !"air.buffer", !"air.location_index", i32 7, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dV"}
  !38 = !{i32 8, !"air.buffer", !"air.location_index", i32 8, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dK"}
  !39 = !{i32 9, !"air.buffer", !"air.location_index", i32 9, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"dQ"}
  !40 = !{i32 10, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 3, !"air.arg_type_size", i32 1, !"air.arg_type_align_size", i32 1, !"air.arg_type_name", !"uchar", !"air.arg_name", !"tg_mem"}
  !41 = !{i32 11, !"air.threadgroup_position_in_grid", !"air.arg_type_name", !"uint3", !"air.arg_name", !"gid"}
  !42 = !{i32 12, !"air.simdgroup_index_in_threadgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"sidx"}
  !43 = !{i32 13, !"air.thread_index_in_simdgroup", !"air.arg_type_name", !"ushort", !"air.arg_name", !"lane_id"}

  !8 = !{i32 1, !"wchar_size", i32 4}
  !9 = !{i32 7, !"air.max_device_buffers", i32 31}
  !10 = !{i32 7, !"air.max_constant_buffers", i32 31}
  !11 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
  !12 = !{i32 7, !"air.max_textures", i32 128}
  !13 = !{i32 7, !"air.max_read_write_textures", i32 8}
  !14 = !{i32 7, !"air.max_samplers", i32 16}
  !15 = !{!"air.compile.denorms_disable"}
  !16 = !{!"air.compile.fast_math_enable"}
  !17 = !{!"air.compile.framebuffer_fetch_enable"}
  !19 = !{!"MetalASM (monolithic attention)"}
  !20 = !{i32 2, i32 8, i32 0}
  !21 = !{!"Metal", i32 4, i32 0, i32 0}
  !22 = !{!"monolithic_attention.ll"}
  """
}
