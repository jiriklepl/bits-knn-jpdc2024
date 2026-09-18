#include <cstddef>
#include <cstdint>

#include <distance/distance-inl.cuh>

// cuVS declares this specialization extern in distance-ext.cuh. Instantiate it here
// because the fused wrapper uses cuVS headers without linking the full cuVS library.
// Keep the definition in a separate translation unit: distance-inl.cuh and
// distance-ext.cuh both specify default template arguments and cannot be combined.
template std::size_t
cuvs::distance::getWorkspaceSize<cuvs::distance::DistanceType::L2Expanded, float, float, float,
                                 std::int32_t>(const float*, const float*, std::int32_t,
                                               std::int32_t, std::int32_t);
