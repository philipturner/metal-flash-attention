# FlashAttention (Metal Port)

This repository ports the official implementation of [FlashAttention](https://github.com/Dao-AILab/flash-attention) to Apple silicon. It is a minimal, maintainable set of source files that reproduces the FlashAttention algorithm.

## Documentation

Single-headed attention only, to focus on the core bottlenecks of different attention algorithms (register pressure, parallelism). With the basic algorithm done correctly, it should be comparatively trivial to add customizations like block sparsity.

Everything is JIT compiled at runtime. This constrasts with the previous implementation, which relied on an executable embedded in Xcode 14.2.

The backward pass uses less memory than [Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention). The official implementation allocates scratch space for atomics and partial sums. Apple hardware lacks native FP32 atomics (`metal::atomic<float>` is emulated). While attempting to circumvent the lack of hardware support, bandwidth and parallelization bottlenecks in the FlashAttention-2 backward kernel were revealed. An alternative backward pass was designed with higher compute cost (7 GEMMs instead of 5 GEMMs). It achieves 100% parallelization efficiency across both the row and column dimensions of the attention matrix. Most importantly, it is easier to code and maintain.

A lot of crazy stuff was done to overcome register pressure bottlenecks. At large head dimensions (e.g. 256), none of the matrix blocks can fit into registers. Not even the accumulator can. Therefore, intentional register spilling is done, but in a more optimized way. A third block dimension was added to the attention algorithm, which blocks along `D`. The aspect ratio of attention matrix blocks was warped heavily, to minimize the bandwidth cost of register spilling. For example, 16-32 along the parallelization dimension and 80-128 along the traversal dimension. There is a large parameter file that takes the `D` dimension, and determines which operands can fit into registers. It then assigns a block size that balances many competing bottlenecks.

The end result is a consistent 4500 gigainstructions per second on M1 Max (80% ALU utilization), at infinite sequence length and infinite head dimension.

![M1_Max_Image.png](./Documentation/M1_Max_Image.png)

![M4_Image.png](./Documentation/M4_Image.png)

Raw Data: https://docs.google.com/spreadsheets/d/1Xf4jrJ7e19I32J1IWIekGE9uMFTeZKoOpQ6hlUoh-xY/edit?usp=sharing

## Quantifying Performance

TODO: Explain roofline model

## Usage

### Setting Up Workflow

On macOS, download the Swift package and compile with `-Xswiftc -Ounchecked`. This compiler option is needed for performance-sensitive CPU code. Release mode cannot be used because it forces the entire codebase to be recompiled from scratch, every time there is a single change. Navigate to the Git repo in Finder and double-click `Package.swift`. An Xcode window should pop up. On the left, there should be a hierarchy of files. If you cannot unravel the hierarchy, something went wrong.

```
git clone https://github.com/philipturner/metal-flash-attention
swift build -Xswiftc -Ounchecked # Does is even compile?
swift test -Xswiftc -Ounchecked # Does the test suite finish in ~10 seconds?
```

Alternatively, create a new Xcode project with the SwiftUI template. Override the `"Hello, world!"` string with a call to a function that returns a `String`. This function will execute the script of your choosing, then call `exit(0)`, so the app crashes before rendering anything to the screen. You will use the output in the Xcode console as feedback about your code. This workflow is compatible with both macOS and iOS.

Specify `-Xswiftc -Ounchecked` through <b>Project</b> > your project's name > <b>Build Settings</b> > <b>Swift Compiler - Code Generation</b> > <b>Optimization Level</b>. The second column of the table lists your project's name. Click <b>Other</b> in the dropdown and type `-Ounchecked` in the panel that appears. Next, add this repository as a Swift package dependency. Look through some of the tests under `Tests/FlashAttention`. Copy the raw source code for one of these tests into your project. Invoke the test from the function in the previous paragraph. Examine what it displays on the console.

To modify the kernel generation code (e.g. add multi-head or mask support), copy the raw source code into your Xcode project. Either use `git clone` in a separate folder, or download the raw files on GitHub as a ZIP. There is also a way to link to your fork of `metal-flash-attention` and autosave your changes to the cloud, but this is more difficult to set up. Remove the Swift package dependency from the previous paragraph. Re-run the test of your choosing. Does it compile and display something in the console?

### Editing Source Code

Locate of the multi-line string literals in either of these folders:

```
Sources/FlashAttention/Attention/AttentionKernel
Sources/FlashAttention/GEMM/GEMMKernel
```

Add random text to one of them. Compile and run the project again. Something should go terribly wrong. For example, the Metal compiler may throw an error. If this does not happen, try messing up a different line of code somewher else. If the test still passes, Xcode is not registering your changes.

Proceed with coding up [block sparsity](https://pytorch.org/blog/flexattention/) or something. Get feedback about whether the code works at all, whether it works fast, and whether it works fast for any problem size. Then integrate the raw source code into your app, or translate it to another language.
