//
// Copyright (c) Microsoft. All rights reserved.
// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma warning(disable : 4267) // conversion from size_t to int or other types

#include "Basics.h"
#include "MPIWrapper.h"
#include "Matrix.h"
#include "SimpleDistGradAggregatorHelper.h"
#include "DistGradHeader.h"
#include "IDistGradAggregator.h"
#include "SimpleDistGradAggregator.h"
#include "V2SimpleDistGradAggregator.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
std::shared_ptr<IDistGradAggregator<ElemType>> GetSimpleDistGradAggregator(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes)
{
    if (Globals::UseV2Aggregator())
        return std::make_shared<V2SimpleDistGradAggregator<ElemType>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            ::CNTK::MPICommunicator(packThresholdSizeInBytes));
    else
        return std::make_shared<SimpleDistGradAggregator<ElemType>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            packThresholdSizeInBytes);
}

template <>
std::shared_ptr<IDistGradAggregator<half>> GetSimpleDistGradAggregator<half>(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes)
{
    if (Globals::UseV2Aggregator())
    {
        fprintf(stderr, "Currently we do not support V2 aggregator for half.");
        NOT_IMPLEMENTED;
    }
    else
        return std::make_shared<SimpleDistGradAggregator<half>>(
            mpi,
            useAsyncAggregation,
            deviceId,
            syncStatsTrace,
            packThresholdSizeInBytes);
}

template std::shared_ptr<IDistGradAggregator<float>> GetSimpleDistGradAggregator(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes);

template std::shared_ptr<IDistGradAggregator<double>> GetSimpleDistGradAggregator(
    const MPIWrapperPtr& mpi,
    bool useAsyncAggregation,
    int deviceId,
    int syncStatsTrace,
    size_t packThresholdSizeInBytes);
} } }