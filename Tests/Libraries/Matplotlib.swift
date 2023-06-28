//
//  Matplotlib.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/27/23.
//

import Foundation
import PythonKit

func testImports() {
  let np = PythonContext.global.np
  let plt = PythonContext.global.plt
  let mpl = PythonContext.global.mpl
  
  let M = 100
  let N = 100
  let K = 100
  let matrices: [PythonObject] = (0..<3).map { _ in
    // Configured with Accelerate and using AMX-accelerated FP32.
    let rng = np.random.default_rng()
    return np.matmul(
      rng.random(PythonObject(tupleOf: M, K), dtype: np.float32),
      rng.random(PythonObject(tupleOf: K, N), dtype: np.float32))
  }
  
  // https://matplotlib.org/2.0.2/examples/api/colorbar_only.html
  // Make a figure and axes with dimensions as desired.
  let fig = plt.figure(figsize: PythonObject(tupleOf: 10, 4))
  
  let delta: Double = 0.02
  let ax1 = fig.add_axes([
    0.10 - delta, 0.10, 0.50 + 2 * delta, 0.05])
  let ax2 = fig.add_axes([
    0.70 - delta, 0.10, 0.20 + 2 * delta, 0.05])
  
  let ax3 = fig.add_axes([
    0.10 - delta, 0.25, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])
  let ax4 = fig.add_axes([
    0.40 - delta, 0.25, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])
  let ax5 = fig.add_axes([
    0.70 - delta, 0.25, 0.20 + 2 * delta, 0.50 + 2 * delta * 10/4])

  // Set the colormap and norm to correspond to the data for which
  // the colorbar will be used.
  let top = mpl.cm.get_cmap("Oranges_r", 128)
  let bottom = mpl.cm.get_cmap("Blues", 128)
  let newcolors = np.vstack([top(np.linspace(0, 1, 128)),
                         bottom(np.linspace(0, 1, 128))])
  let cmap = mpl.colors.ListedColormap(newcolors, name: "OrangeBlue")

  // ColorbarBase derives from ScalarMappable and puts a colorbar
  // in a specified axes, so it has everything needed for a
  // standalone colorbar.  There are many more kwargs, but the
  // following gives a basic continuous colorbar with ticks
  // and labels.
  let padding: Double = 0.00
  let norm1 = mpl.colors.Normalize(
    vmin: Double(K) * -(0.00 + padding), vmax: Double(K) * (0.50 + padding))
  let cb1 = mpl.colorbar.ColorbarBase(ax1, cmap: cmap,
                                      norm: norm1,
                                      orientation: "horizontal")
  
  let sqrK = 0.5 * Double(K) // sqrt(Double(K))
  let norm2 = mpl.colors.Normalize(
    vmin: -sqrK, vmax: sqrK)
  let cb2 = mpl.colorbar.ColorbarBase(ax2, cmap: cmap,
                                      norm: norm2,
                                      orientation: "horizontal")
  
  struct Plot {
    var matrix: PythonObject
    var axis: PythonObject
    var norm: PythonObject
  }
  let plots: [Plot] = [
    Plot(matrix: matrices[0], axis: ax3, norm: norm1),
    Plot(matrix: matrices[1], axis: ax4, norm: norm1),
    Plot(matrix: matrices[2], axis: ax5, norm: norm2),
  ]
  for plot in plots {
    let ax = plot.axis
    ax.imshow(plot.matrix, cmap: cmap, norm: plot.norm)
    
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
        print("tenPower: \(tenPower), base: \(base)")
        if let output = section(base: base) {
          return output
        }
      }
      fatalError("Number too large: \(dimension)")
    }
    ax.set_yticks(np.arange(0, M, tickStep(dimension: M)))
    ax.set_xticks(np.arange(0, N, tickStep(dimension: N)))
    ax.xaxis.set_ticks_position("top")
  }
  
  plt.draw()
  plt.pause(2)
}
