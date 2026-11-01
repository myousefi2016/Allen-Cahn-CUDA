// Explicit template instantiation for common types.
#include "cuda/DeviceField.cuh"

namespace ac::cuda {

// Ensure the linker has symbols for double and float fields.
template class DeviceField<double>;
template class DeviceField<float>;
template class DeviceField<int>;

} // namespace ac::cuda
