//
//  build.swift
//  MetalFlashAttention
//
//  Created by Philip Turner on 6/26/23.
//

import Foundation
import QuartzCore

// MARK: - Documentation

// Color scheme for anything from the command-line:
// - cyan / all caps:
//     warning messages
//     section headers
// - green:
//     options
//     script parameters
//     command-line arguments
// - yellow:
//     performance data

let title = """
  Script for compiling 'libMetalFlashAttention.metallib' and configuring the \
  test suite.
  """

func makeGreen<T: StringProtocol>(_ string: T) -> String {
  "\u{1b}[0;32m\(string)\u{1b}[0m"
}
func makeYellow<T: StringProtocol>(_ string: T) -> String {
  "\u{1b}[0;33m\(string)\u{1b}[0m"
}
func makeCyan<T: StringProtocol>(_ string: T) -> String {
  "\u{1b}[0;36m\(string)\u{1b}[0m"
}

let argumentReprs: [String: String] = [
  "--external-metallib-path=PATH": "Use a pre-compiled metallib for tests. Mutually exclusive with \(makeGreen("--xcode-path")).",
  "--help | -h": "Show documentation for command-line arguments.",
  "--platform=(iOS|macOS)": "Which platform to build the metallib for (required).",
  "--verbose": "Whether to show compiler warnings that are not errors.",
  "--xcode-path=PATH": "Xcode application to compile the AIR files with (uses Xcode 14.2 by default).",
]

func makeUsage(help: Bool = false) -> String {
  var output: String
  if help {
    output = """
    \(makeCyan("HELP:"))
      \(title)
    """
  } else {
    output = """
    \(makeCyan("USAGE:"))
      swift build.swift
    """
  }
  
  let maxPadding = argumentReprs.keys.reduce(0, { max($0, $1.count) })
  for key in argumentReprs.keys.sorted() {
    output += "\n    "
    let greenKey = makeGreen(key)
    if help {
      let padding = maxPadding - key.count
      output += greenKey + String(repeating: " ", count: padding)
      output += " : \(argumentReprs[key]!)"
    } else {
      output += "[" + greenKey + "]"
    }
  }
  return output
}

if CommandLine.arguments.contains(where: {
  $0 == "--help" || $0 == "-h"
}) {
  print(makeUsage(help: true))
  exit(0)
}

func error(message: String) -> Never {
  print(makeCyan("ERROR:"), message)
  print(makeUsage())
  exit(1)
}

func logLatency(startTime: Double, prefix: String, suffix: String) {
  let endTime = CACurrentMediaTime()
  let latencyRepr = "\(String(format: "%.1f", Double(endTime - startTime))) s"
  
  var output = prefix
  output += makeYellow(latencyRepr)
  output += suffix
  print(output)
}

// MARK: - Parse Arguments

struct BuildSettings {
  var externalMetallibPath: String? = nil
  var platform: Platform? = nil
  var verbose: Bool = false
  var xcodePath: String = "/Applications/Xcode 14.2.app"
  
  enum Platform {
    case iOS
    case macOS
    
    var metalToolsPath: String {
      switch self {
      case .iOS:
        return "ios"
      case .macOS:
        return "macos"
      }
    }
    
    var deploymentVersionArgument: String {
      switch self {
      case .iOS:
        return "-mios-version-min=16.0.0"
      case .macOS:
        return "-mmacosx-version-min=13.0.0"
      }
    }
    
    var xcrunSDK: String {
      switch self {
      case .iOS:
        return "iphoneos"
      case .macOS:
        return "macosx"
      }
    }
  }
  
  func metalToolPath(executable: String) -> String {
    guard let metalToolsPath = platform?.metalToolsPath else {
      error(message: "Must specify platform before locating Metal tools.")
    }
    var output = xcodePath
    output += "/Contents/Developer/Toolchains/XcodeDefault.xctoolchain"
    output += "/usr/metal/\(metalToolsPath)/bin/\(executable)"
    return output
  }
  
  func xcrunMetalArguments(executable: String) -> [String] {
    guard let xcrunSDK = platform?.xcrunSDK else {
      error(message: "Must specify platform before locating Metal tools.")
    }
    return ["-sdk", xcrunSDK, executable]
  }
}

var settings = BuildSettings()
var i: Int = 1
while i < CommandLine.arguments.count {
  defer { i += 1 }
  let argument = CommandLine.arguments[i]
  
  func extractSecondArgument(flag: String) -> String {
    if argument.starts(with: flag + "=") {
      return String(argument.dropFirst(flag.count + 1))
    } else {
      i += 1
      guard i < CommandLine.arguments.count else {
        error(message: "No arguments were found after '\(flag)'.")
      }
      return CommandLine.arguments[i]
    }
  }
  if argument.starts(with: "--external-metallib-path") {
    let path = extractSecondArgument(flag: "--external-metallib-path")
    settings.externalMetallibPath = path
    guard fileExists(url: URL(fileURLWithPath: path)) else {
      error(message: "Invalid external metallib: '\(path)'")
    }
    continue
  }
  if argument.starts(with: "--platform") {
    let platform = extractSecondArgument(flag: "--platform")
    switch platform {
    case "iOS":
      settings.platform = .iOS
    case "macOS":
      settings.platform = .macOS
    default:
      error(message: "Unrecognized platform: '\(platform)'")
    }
    continue
  }
  if argument.starts(with: "--xcode-path") {
    let path = extractSecondArgument(flag: "--xcode-path")
    settings.xcodePath = path
    guard directoryExists(url: URL(fileURLWithPath: path)) else {
      error(message: "Invalid Xcode path: '\(path)'")
    }
    continue
  }
  
  switch argument {
  case "--verbose":
    // This is recommended when calling the script from Xcode's build system.
    // Although, being able to suppress the warnings is sometimes nice.
    settings.verbose = true
  default:
    error(message: "Unrecognized argument: '\(argument)'")
  }
}
  
// A platform must always be specified.
if settings.platform == nil {
  error(message: "Did not specify a platform.")
}

// If a metallib isn't already specified, ensure you can build one from scratch.
if settings.externalMetallibPath == nil {
  for executable in ["metal", "metal-dsymutil"] {
    let path = settings.metalToolPath(executable: executable)
    assertFileExists(url: URL(fileURLWithPath: path))
  }
}

// MARK: - Prepare File Directories

func directoryExists(url: URL) -> Bool {
  var isDirectory: ObjCBool = false
  let succeeded = FileManager.default.fileExists(
    atPath: url.path, isDirectory: &isDirectory)
  return succeeded && isDirectory.boolValue
}

func fileExists(url: URL) -> Bool {
  var isDirectory: ObjCBool = false
  let succeeded = FileManager.default.fileExists(
    atPath: url.path, isDirectory: &isDirectory)
  return succeeded && !isDirectory.boolValue
}

func assertDirectoryExists(url: URL, line: UInt = #line) {
  guard directoryExists(url: url) else {
    error(message: """
      Line \(line):
      Directory not found at '\(url.path)'.
      """)
  }
}

func assertFileExists(url: URL, line: UInt = #line) {
  guard fileExists(url: url) else {
    error(message: """
      Line \(line):
      File not found at '\(url.path)'.
      """)
  }
}

func touchDirectory(url: URL) {
  if !directoryExists(url: url) {
    try! FileManager.default.createDirectory(
      at: url, withIntermediateDirectories: false)
  }
  assertDirectoryExists(url: url)
}

let workDir = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
assertDirectoryExists(url: workDir)

let sourcesDir = workDir.appending(component: "Sources")
assertDirectoryExists(url: sourcesDir)

let buildDir = workDir.appending(component: "build")
let libDir = buildDir.appending(component: "lib")
let srcDir = buildDir.appending(component: "src")
touchDirectory(url: buildDir)
touchDirectory(url: libDir)
touchDirectory(url: srcDir)

// Always copy the license.
do {
  let srcDir = workDir.appending(component: "LICENSE")
  let dstDir = buildDir.appending(component: "LICENSE")
  try? FileManager.default.removeItem(at: dstDir)
  try! FileManager.default.copyItem(at: srcDir, to: dstDir)
}

// Always copy the API.
do {
  let srcDir = workDir.appending(components: "Documentation", "API.md")
  let dstDir = buildDir.appending(component: "API.md")
  try? FileManager.default.removeItem(at: dstDir)
  try! FileManager.default.copyItem(at: srcDir, to: dstDir)
}

// Allocate the temporary directory.
let tmpDir = buildDir.appending(component: "tmp")
try? FileManager.default.removeItem(at: tmpDir)
try! FileManager.default.createDirectory(
  at: tmpDir, withIntermediateDirectories: false)

var performanceCores: Int = 0
var sizeOfInt: Int = 8
sysctlbyname(
  "hw.perflevel0.physicalcpu_max", &performanceCores, &sizeOfInt, nil, 0)
let subpaths = FileManager.default.subpaths(atPath: sourcesDir.path)!
let numThreads = min(performanceCores, subpaths.count)

// Data for multithreading.
let queue = DispatchQueue(
  label: "com.philipturner.metal-flash-attention.compile")
var compiledFiles: Int = numThreads
var airPaths: [String] = []

// MARK: - Compile AIR Files (Multiple Cores)

let skipCompilation = (settings.externalMetallibPath != nil)
let compileAIRStart = CACurrentMediaTime()
DispatchQueue.concurrentPerform(iterations: numThreads) { z in
  // Arguments to invoke the Metal compiler with.
  var arguments: [String] = []
  arguments.append(settings.platform!.deploymentVersionArgument)
  arguments.append("-c")
  arguments.append("-frecord-sources")
  
  // Suppress compiler warnings unless the user enters '--verbose'.
  if settings.verbose {
    arguments.append("-Wno-unused-function")
    arguments.append("-Wno-unused-variable")
  } else {
    arguments.append("-w")
  }
  
  var i = z
  while i < subpaths.count {
    defer {
      queue.sync {
        if compiledFiles >= subpaths.count {
          i = subpaths.count
        } else {
          i = compiledFiles
          compiledFiles += 1
        }
      }
    }
    
    // First, copy the file into the sources directory.
    let subpath = subpaths[i]
    do {
      let srcURL = sourcesDir.appending(path: subpath)
      let dstURL = srcDir.appending(path: subpath)
      try? FileManager.default.removeItem(at: dstURL)
      try! FileManager.default.copyItem(at: srcURL, to: dstURL)
    }
    if skipCompilation {
      continue
    }
    
    // Then, pass the original file location into the Metal compiler.
    var metalURL: URL
    var airPath = tmpDir.appending(path: subpath).path
    if subpath.hasSuffix(".metal") {
      metalURL = sourcesDir.appending(path: subpath)
      airPath.removeLast(".metal".count)
    } else {
      // There isn't any other way to include the header files in the debug
      // symbols. This workaround requires that no Metal file has the exact same
      // name as any header.
      let srcURL = sourcesDir.appending(path: subpath)
      let dstURL = tmpDir.appending(path: subpath + ".metal")
      try? FileManager.default.removeItem(at: dstURL)
      try! FileManager.default.copyItem(at: srcURL, to: dstURL)
      metalURL = dstURL
    }
    airPath.append(".air")
    try? FileManager.default.removeItem(atPath: airPath)
    guard FileManager.default.createFile(
      atPath: airPath, contents: nil) else {
      fatalError("Could not create destination path '\(airPath)'.")
    }
    queue.sync {
      airPaths.append(airPath)
    }
    
    let process = Process()
    let toolPath = settings.metalToolPath(executable: "metal")
    process.executableURL = URL(filePath: toolPath)
    process.arguments = arguments + [metalURL.path, "-o", airPath]
    
    // Encapsulate the compiler output into pipes, otherwise it will be jumbled
    // because multiple threads are writing simultaneously (like a data race).
    let stdoutPipe = Pipe()
    let stderrPipe = Pipe()
    process.standardOutput = stdoutPipe
    process.standardError = stderrPipe
    try! process.run()
    process.waitUntilExit()
    
    var outputs: [String] = []
    for pipe in [stdoutPipe, stderrPipe] {
      let data = pipe.fileHandleForReading.readDataToEndOfFile()
      outputs.append(String(data: data, encoding: .utf8)!)
    }
    queue.sync {
      for output in outputs where output.count > 0 {
        print(output)
      }
      if process.terminationStatus != 0 {
        compiledFiles = subpaths.count + 1
        print("Could not compile source '\(subpath)'.")
      }
    }
    guard process.terminationStatus == 0 else {
      return
    }
  }
}

// An error is signaled by invalidating the number of compiled files.
if compiledFiles == subpaths.count + 1 {
  exit(2)
}

// MARK: - Compile Metal Library (Single Core)

let metallibName = "libMetalFlashAttention.metallib"
let metallibsymName = "libMetalFlashAttention.metallibsym"
if skipCompilation {
  let metallibURL = URL(filePath: settings.externalMetallibPath!)
  guard metallibURL.path.hasSuffix(metallibName) else {
    let green = makeGreen(metallibName)
    error(message: "External metallib was not named \(green).")
  }
  guard fileExists(url: metallibURL) else {
    let green = makeGreen(metallibName)
    error(message: "Could not locate \(green).")
  }
  
  let metallibsymURL = metallibURL.deletingLastPathComponent()
    .appending(component: metallibsymName)
  guard fileExists(url: metallibsymURL) else {
    let green = makeGreen(metallibsymName)
    var message = "Could not locate \(green) in the same directory as "
    message += makeGreen(metallibName) + "."
    error(message: message)
  }
  
  // Copy the metallib and debug symbols.
  let targets = [
    metallibURL: metallibName,
    metallibsymURL: metallibsymName
  ]
  for (key, value) in targets {
    let srcURL = key
    let dstURL = libDir.appending(path: value)
    try? FileManager.default.removeItem(at: dstURL)
    try! FileManager.default.copyItem(at: srcURL, to: dstURL)
  }
  logLatency(
    startTime: compileAIRStart,
    prefix: "Packaged the Metal library in: ",
    suffix: "")
} else {
  logLatency(
    startTime: compileAIRStart,
    prefix: "Compiled the AIR files in: ",
    suffix: "")
  
  let compileMetallibStart = CACurrentMediaTime()
  var arguments: [String] = []
  
  // Package the metallib using the up-to-date Xcode version.
  func runProcess() {
    let process = try! Process.run(
      URL(fileURLWithPath: "/usr/bin/xcrun"),
      arguments: arguments)
    process.waitUntilExit()
  }
  
  arguments = []
  arguments += settings.xcrunMetalArguments(executable: "metal")
  arguments.append(settings.platform!.deploymentVersionArgument)
  arguments.append("-frecord-sources")
  arguments.append("-o")
  arguments.append(libDir.appending(component: metallibName).path)
  arguments += airPaths
  runProcess()
  
  arguments = []
  arguments += settings.xcrunMetalArguments(executable: "metal-dsymutil")
  arguments.append("-flat")
  arguments.append("-remove-source")
  arguments.append(libDir.appending(component: metallibName).path)
  arguments.append("-o")
  arguments.append(libDir.appending(component: metallibsymName).path)
  runProcess()
  
  logLatency(
    startTime: compileMetallibStart,
    prefix: "Compiled the metallib in: ",
    suffix: "")
}

// MARK: - Clean Up

// This shouldn't trigger if you encounter a fatal error. That way, you can
// look inside the directory and examine the culprit.
try! FileManager.default.removeItem(at: tmpDir)
