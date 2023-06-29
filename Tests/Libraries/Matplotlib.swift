//
//  Matplotlib.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Foundation
import PythonKit

// Example Python tensors to ensure the graph works correctly.
func MPL_showExamples(
  withGEMM: Bool
) {
  let M = 100
  let N = 100
  let K = 100
  let matrices: [PythonObject] = (0..<3).map { _ in
    // Configured with Accelerate and using AMX-accelerated FP32.
    let np = PythonContext.global.np
    let rng = np.random.default_rng()
    if withGEMM {
      return np.matmul(
        rng.random(PythonObject(tupleOf: M, K), dtype: np.float32),
        rng.random(PythonObject(tupleOf: K, N), dtype: np.float32))
    } else {
      return rng.random(PythonObject(tupleOf: M, K), dtype: np.float32)
    }
  }
  
  let parameters = EuclideanDistanceParameters(matrixK: K)
  MPL_showGraphs(
    primary: matrices[0],
    secondary: matrices[1],
    ternary: matrices[2],
    parameters: parameters,
    isComparison: false)
}

func MPL_showBackends<T: TensorElement>(
  mfa: Tensor<T>,
  mps: Tensor<T>,
  numpy: Tensor<T>,
  parameters: EuclideanDistanceParameters
) {
  precondition(mfa.buffer.backend == .mfa)
  precondition(mps.buffer.backend == .mps)
  precondition(numpy.buffer.backend == .numpy)
  
  let primary = mfa.numpy()
  let secondary = mps.numpy()
  let ternary = numpy.numpy()
  MPL_showGraphs(
    primary: primary,
    secondary: secondary,
    ternary: ternary,
    parameters: parameters,
    isComparison: false)
}

func MPL_showComparison<T: TensorElement>(
  actual: Tensor<T>,
  expected: Tensor<T>,
  parameters: EuclideanDistanceParameters
) {
  let actualBackend = actual.buffer.backend
  let expectedBackend = expected.buffer.backend
  precondition(actualBackend != expectedBackend)
  
  let primary = actual.numpy()
  let secondary = expected.numpy()
  let ternary = primary - secondary
  MPL_showGraphs(
    primary: primary,
    secondary: secondary,
    ternary: ternary,
    parameters: parameters,
    isComparison: true)
}

fileprivate func MPL_showGraphs(
  primary: PythonObject,
  secondary: PythonObject,
  ternary: PythonObject,
  parameters: EuclideanDistanceParameters,
  isComparison: Bool
) {
  let np = PythonContext.global.np
  let plt = PythonContext.global.plt
  let mpl = PythonContext.global.mpl
  
  // https://matplotlib.org/2.0.2/examples/api/colorbar_only.html
  // Make a figure and axes with dimensions as desired.
  let fig = plt.figure(figsize: PythonObject(tupleOf: 10, 4))
  
  let delta: Double = 0.02
  let shift1: Double = 0.01
  let shift2: Double = 0.02
  var ax1: PythonObject
  var ax2: PythonObject?
  
  if isComparison {
    ax1 = fig.add_axes([
      0.10 - delta, 0.10 + shift1, 0.50 + 2 * delta, 0.05])
    ax2 = fig.add_axes([
      0.70 - delta, 0.10 + shift1, 0.20 + 2 * delta, 0.05])
  } else {
    ax1 = fig.add_axes([
      0.10 - delta, 0.10 + shift1, 0.80 + 2 * delta, 0.05])
  }
  let ax3 = fig.add_axes([
    0.10 - delta, 0.25 - shift2, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])
  let ax4 = fig.add_axes([
    0.40 - delta, 0.25 - shift2, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])
  let ax5 = fig.add_axes([
    0.70 - delta, 0.25 - shift2, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])
  
  // Set the colormap and norm to correspond to the data for which
  // the colorbar will be used.
  let top = mpl.cm.get_cmap("Oranges_r", 128)
  let bottom = mpl.cm.get_cmap("Blues", 128)
  let newcolors = np.vstack([
    top(np.linspace(0, 1, 128)),
    bottom(np.linspace(0, 1, 128))
  ])
  let cmap = mpl.colors.ListedColormap(newcolors, name: "OrangeBlue")
  
  var cb1: PythonObject
  var cb2: PythonObject?
  var norm1: PythonObject
  var norm2: PythonObject?
  
  let padding: Double = 0.00
  let vmin1 = Double(parameters.averageMagnitude) * (0.00 - padding)
  let vmax1 = Double(parameters.averageMagnitude) * (1.00 + padding)
  norm1 = mpl.colors.Normalize(vmin: vmin1, vmax: vmax1)
  cb1 = mpl.colorbar.ColorbarBase(ax1, cmap: cmap,
                                  norm: norm1,
                                  orientation: "horizontal")
  
  if isComparison {
    let sqrK = 0.5 * Double(parameters.averageDeviation)
    let vmin2 = -sqrK
    let vmax2 = +sqrK
    norm2 = mpl.colors.Normalize(vmin: vmin2, vmax: vmax2)
    cb2 = mpl.colorbar.ColorbarBase(ax2!, cmap: cmap,
                                    norm: norm2!,
                                    orientation: "horizontal")
  }
  
  // Suppress compiler warnings.
  _ = cb1
  _ = cb2
  
  struct Plot {
    var matrix: PythonObject
    var axis: PythonObject
    var norm: PythonObject
    var title: String
    
    init(
      _ matrix: PythonObject,
      axis: PythonObject,
      norm: PythonObject,
      title: String
    ) {
      self.matrix = matrix
      self.axis = axis
      self.norm = norm
      self.title = title
    }
  }
  var plots: [Plot]
  if isComparison {
    plots = [
      Plot(primary, axis: ax3, norm: norm1, title: "Actual"),
      Plot(secondary, axis: ax4, norm: norm1, title: "Expected"),
      Plot(ternary, axis: ax5, norm: norm2!, title: "Difference"),
    ]
  } else {
    plots = [
      Plot(primary, axis: ax3, norm: norm1, title: "MFA"),
      Plot(secondary, axis: ax4, norm: norm1, title: "MPS"),
      Plot(ternary, axis: ax5, norm: norm1, title: "NumPy"),
    ]
  }
  for plot in plots {
    let (matrix, ax) = (plot.matrix, plot.axis)
    ax.title.set_text(plot.title)
    ax.imshow(matrix, cmap: cmap, norm: plot.norm)
    
    func tickStep(dimension: Int) -> Int {
      func section(base: Int) -> Int? {
        if dimension < 10 * base { return 1 * base }
        if dimension < 20 * base { return 2 * base }
        if dimension < 50 * base { return 5 * base }
        return nil
      }
      
      if dimension < 5 { return 1 }
      for tenPower in 0..<10 {
        var base = 1
        for _ in 0..<tenPower {
          base *= 10
        }
        if let output = section(base: base) {
          return output
        }
      }
      fatalError("Number too large: \(dimension)")
    }
    
//    let ndim = matrix.ndim
//    let N = Int(matrix.shape[ndim - 1])!
//    let M = Int(matrix.shape[ndim - 2])!
//    ax.set_yticks(np.arange(0, M, tickStep(dimension: M)))
//    ax.set_xticks(np.arange(0, N, tickStep(dimension: N)))
    ax.xaxis.set_ticks_position("top")
  }
  plt.show()
}
