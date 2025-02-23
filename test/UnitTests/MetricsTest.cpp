/******************************************************************************
 * FIRESTARTER - A Processor Stress Test Utility
 * Copyright (C) 2024 TU Dresden, Center for Information Services and High
 * Performance Computing
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/\>.
 *
 * Contact: daniel.hackenberg@tu-dresden.de
 *****************************************************************************/

#include "firestarter/Measurement/Metric.hpp"
#include "firestarter/Measurement/MetricInterface.h"

#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::testing;

namespace {

void callback() {};

template <int32_t ReturnValue> auto init() -> int32_t { return ReturnValue; };

auto getReading(double* /*Value*/) -> int32_t { return 0; };

auto getError() -> const char* { return ""; };

const MetricType InsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/1, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

const MetricType NoInsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

template <int32_t InitReturnValue, const MetricType& Type> struct MetricMock {
  inline static const char* Name = "Name";
  inline static const char* Unit = "Unit";
  inline static uint64_t CallbackTime = 0U;

  MOCK_METHOD(int32_t, registerInsertCallbackMock, (void (*Ptr1)(void*, const char*, int64_t, double), void* Ptr2));

  MOCK_METHOD(int32_t, finiMock, ());

  MetricInterface AvailableMetric = {
      /*Name=*/Name,
      /*Type=*/Type,
      /*Unit=*/Unit,
      /*CallbackTime=*/CallbackTime,
      /*Callback=*/callback,
      /*Init=*/init<InitReturnValue>,
      /*Fini=*/fini,
      /*GetReading=*/getReading,
      /*GetError=*/getError,
      /*RegisterInsertCallback=*/registerInsertCallback,
  };

  static auto instance() -> MetricMock& {
    static MetricMock Instance;
    return Instance;
  }

  MetricMock(MetricMock const&) = delete;
  void operator=(MetricMock const&) = delete;

private:
  MetricMock() = default;

  static auto registerInsertCallback(void (*FunctionPtr)(void*, const char*, int64_t, double), void* Cls) -> int32_t {
    return instance().registerInsertCallbackMock(FunctionPtr, Cls);
  }

  static auto fini() -> int32_t { return instance().finiMock(); }
};

}; // namespace

TEST(MetricsTest, CheckAvailableMetricFromCInterface) {
  auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
  EXPECT_CALL(AvailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  // Metric is created, it is not yet initialized.
  EXPECT_EQ(AvailableRoot->Name, AvailableMetricMock.Name);
  EXPECT_TRUE(AvailableRoot->Values.empty());
  EXPECT_EQ(AvailableRoot->MetricPtr, &AvailableMetricMock.AvailableMetric);
  EXPECT_TRUE(AvailableRoot->Submetrics.empty());
  EXPECT_FALSE(AvailableRoot->Dylib);
  EXPECT_FALSE(AvailableRoot->Stdin);
  EXPECT_FALSE(AvailableRoot->Initialized);

  EXPECT_TRUE(AvailableRoot->Available);

  // Check if the metric inititializes
  EXPECT_TRUE(AvailableRoot->initialize());
}

TEST(MetricsTest, CheckUnavailableMetricFromCInterface) {
  auto& UnavailableMetricMock = MetricMock</*InitReturnValue=*/1, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(UnavailableMetricMock, finiMock()).Times(1);
  EXPECT_CALL(UnavailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto UnavailableRoot = firestarter::measurement::RootMetric::fromCInterface(UnavailableMetricMock.AvailableMetric);

  EXPECT_FALSE(UnavailableRoot->Available);

  // Check if the metric does not inititalize
  EXPECT_FALSE(UnavailableRoot->initialize());
}

TEST(MetricsTest, CheckInsertCallback) {
  // Check that the insert callback is provided during initialization
  {
    auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, InsertCallbackMetricType>::instance();
    EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);

    auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

    // Check if the metric inititializes
    EXPECT_CALL(AvailableMetricMock,
                registerInsertCallbackMock(firestarter::measurement::insertCallback, AvailableRoot.get()))
        .Times(1);

    EXPECT_TRUE(AvailableRoot->initialize());
  }

  {
    auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
    EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);

    auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

    // Check if the metric inititializes
    EXPECT_CALL(AvailableMetricMock,
                registerInsertCallbackMock(firestarter::measurement::insertCallback, AvailableRoot.get()))
        .Times(0);
    EXPECT_TRUE(AvailableRoot->initialize());
  }
}