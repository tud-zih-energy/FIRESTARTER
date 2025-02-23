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

#include <chrono>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::testing;

namespace {

template <int32_t ReturnValue> auto init() -> int32_t { return ReturnValue; };

auto getReading(double* /*Value*/) -> int32_t { return 0; };

auto getError() -> const char* { return ""; };

const MetricType InsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/1, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

const MetricType NoInsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

template <int32_t InitReturnValue, const MetricType& Type, uint64_t CallbackTime = 0U> struct MetricMock {
  inline static const char* Name = "Name";
  inline static const char* Unit = "Unit";

  MOCK_METHOD(void, callbackMock, ());

  MOCK_METHOD(int32_t, finiMock, ());

  MOCK_METHOD(int32_t, registerInsertCallbackMock, (void (*Ptr1)(void*, const char*, int64_t, double), void* Ptr2));

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

  static void callback() { return instance().callbackMock(); }

  static auto fini() -> int32_t { return instance().finiMock(); }

  static auto registerInsertCallback(void (*FunctionPtr)(void*, const char*, int64_t, double), void* Cls) -> int32_t {
    return instance().registerInsertCallbackMock(FunctionPtr, Cls);
  }
};

}; // namespace

TEST(MetricsTest, CheckAvailableFromCInterface) {
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

TEST(MetricsTest, CheckUnavailableFromCInterface) {
  auto& UnavailableMetricMock = MetricMock</*InitReturnValue=*/1, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(UnavailableMetricMock, finiMock()).Times(1);
  EXPECT_CALL(UnavailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto UnavailableRoot = firestarter::measurement::RootMetric::fromCInterface(UnavailableMetricMock.AvailableMetric);

  EXPECT_FALSE(UnavailableRoot->Available);

  // Check if the metric does not inititalize
  EXPECT_FALSE(UnavailableRoot->initialize());
}

TEST(MetricsTest, CheckRegisterInsertCallbackFromCInterface) {
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

TEST(MetricsTest, CheckNoTimedCallbackFromCInterface) {
  // Callback is not enabled (Callbacktime = 0)

  auto& AvailableMetricMock =
      MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType, /*CallbackTime=*/0U>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::insertCallback, AvailableRoot.get()))
      .Times(0);

  {
    auto Callback = AvailableRoot->getTimedCallback();
    EXPECT_FALSE(Callback.has_value());
  }

  EXPECT_TRUE(AvailableRoot->initialize());

  {
    auto Callback = AvailableRoot->getTimedCallback();
    EXPECT_FALSE(Callback.has_value());
  }
}

TEST(MetricsTest, CheckTimedCallbackFromCInterface) {
  constexpr auto CallbackTime = 11U;
  auto& AvailableMetricMock =
      MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType, /*CallbackTime=*/CallbackTime>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::insertCallback, AvailableRoot.get()))
      .Times(0);

  {
    auto Callback = AvailableRoot->getTimedCallback();
    EXPECT_TRUE(Callback.has_value());

    // no initialized, callback should not call the metric callback

    // Call the callback
    EXPECT_CALL(AvailableMetricMock, callbackMock()).Times(0);
    std::get<0>(Callback.value())();

    // Check the correct time is returned
    EXPECT_EQ(std::get<1>(Callback.value()), std::chrono::microseconds(CallbackTime));
  }

  EXPECT_TRUE(AvailableRoot->initialize());

  {
    auto Callback = AvailableRoot->getTimedCallback();
    EXPECT_TRUE(Callback.has_value());

    // initialized, callback should  call the metric callback

    // Call the callback
    EXPECT_CALL(AvailableMetricMock, callbackMock()).Times(1);
    std::get<0>(Callback.value())();

    // Check the correct time is returned
    EXPECT_EQ(std::get<1>(Callback.value()), std::chrono::microseconds(CallbackTime));
  }
}