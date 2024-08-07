#ifndef CoreCount_hpp
#define CoreCount_hpp

#include <stdint.h>

/// Finds the core count on macOS devices, using IORegistry.
///
/// Source: [AppleGPUInfo](https://github.com/philipturner/applegpuinfo)
///
/// This code was generated by GPT-4 a few days after launch (early 2023).
/// Since then, it has undergone extensive human review and real-world testing.
/// It proved that proto-AGI could be a practically useful tool, in this case
/// assisting with code creation.
int64_t findCoreCount();

#endif /* CoreCount_hpp */
