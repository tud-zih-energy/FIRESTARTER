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

#include "firestarter/Measurement/MeasurementWorker.hpp"
#include "firestarter/Measurement/Metric.hpp"
#include "firestarter/Measurement/MetricInterface.h"
#include "firestarter/Measurement/TimeValue.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <thread>

// NOLINTNEXTLINE(google-build-using-namespace)
using namespace ::testing;

namespace {

template <int32_t ReturnValue> auto init() -> int32_t { return ReturnValue; };

auto getError() -> const char* { return ""; };

using InsertFunction = void (*)(void*, uint64_t, int64_t, double);
InsertFunction FrequencyInsert = nullptr;
void* FrequencyContext = nullptr;
InsertFunction IpcInsert = nullptr;
void* IpcContext = nullptr;

auto registerFrequency(InsertFunction Insert, void* Context) -> int32_t {
  FrequencyInsert = Insert;
  FrequencyContext = Context;
  return 0;
}

auto registerIpc(InsertFunction Insert, void* Context) -> int32_t {
  IpcInsert = Insert;
  IpcContext = Context;
  return 0;
}

const MetricType InsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/1, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

const MetricType NoInsertCallbackMetricType{
    /*Absolute=*/0,       /*Accumalative=*/0,         /*DivideByThreadCount=*/0,
    /*InsertCallback=*/0, /*IgnoreStartStopDelta=*/0, /*Reserved=*/0};

template <int32_t InitReturnValue, const MetricType& Type, uint64_t CallbackTime = 0U, bool UseSubmetrics = false>
struct MetricMock {
  inline static const char* FakeName = "Name";
  inline static const char* FakeUnit = "Unit";
  inline static double FakeValue = 1.0;

  MOCK_METHOD(void, callbackMock, ());

  MOCK_METHOD(int32_t, finiMock, ());

  MOCK_METHOD(const char**, getSubmetricNamesMock, ());

  MOCK_METHOD(int32_t, getReadingMock, (double*, uint64_t));

  MOCK_METHOD(int32_t, registerInsertCallbackMock, (void (*Ptr1)(void*, uint64_t, int64_t, double), void* Ptr2));

  MetricInterface AvailableMetric = {
      /*Name=*/FakeName,
      /*Type=*/Type,
      /*Unit=*/FakeUnit,
      /*CallbackTime=*/CallbackTime,
      /*Callback=*/callback,
      /*Init=*/init<InitReturnValue>,
      /*Fini=*/fini,
      /*GetSubmetricNames=*/
      nullptr,
      /*GetReading=*/getReading,
      /*GetError=*/getError,
      /*RegisterInsertCallback=*/registerInsertCallback,
  };

  static auto instance() -> MetricMock& {
    static MetricMock Instance;
    if constexpr (UseSubmetrics) {
      Instance.AvailableMetric.GetSubmetricNames = getSubmetricNames;
    }
    return Instance;
  }

  MetricMock(MetricMock const&) = delete;
  void operator=(MetricMock const&) = delete;

private:
  MetricMock() = default;

  static void callback() { return instance().callbackMock(); }

  static auto fini() -> int32_t { return instance().finiMock(); }

  static auto getSubmetricNames() -> const char** { return instance().getSubmetricNamesMock(); };

  static auto getReading(double* Value, uint64_t NumElems) -> int32_t {
    *Value = FakeValue;
    return instance().getReadingMock(Value, NumElems);
  }

  static auto registerInsertCallback(void (*FunctionPtr)(void*, uint64_t, int64_t, double), void* Cls) -> int32_t {
    return instance().registerInsertCallbackMock(FunctionPtr, Cls);
  }
};

}; // namespace

TEST(MetricsTest, CheckAvailableFromCInterface) {
  auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);
  EXPECT_CALL(AvailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  // Metric is created, it is not yet initialized.
  EXPECT_EQ(AvailableRoot->Name, AvailableMetricMock.FakeName);
  EXPECT_TRUE(AvailableRoot->Values.empty());
  EXPECT_EQ(AvailableRoot->MetricPtr, &AvailableMetricMock.AvailableMetric);
  EXPECT_TRUE(AvailableRoot->Submetrics.empty());
  EXPECT_FALSE(AvailableRoot->Dylib);
  EXPECT_FALSE(AvailableRoot->Stdin);
  EXPECT_FALSE(AvailableRoot->Initialized);

  EXPECT_TRUE(AvailableRoot->Available);

  // Check if the metric inititializes
  EXPECT_TRUE(AvailableRoot->initialize());

  EXPECT_TRUE(AvailableRoot->Submetrics.empty());
  EXPECT_TRUE(AvailableRoot->Initialized);
}

TEST(MetricsTest, CheckUnavailableFromCInterface) {
  auto& UnavailableMetricMock = MetricMock</*InitReturnValue=*/1, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(UnavailableMetricMock, finiMock()).Times(1);
  EXPECT_CALL(UnavailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(UnavailableMetricMock, getReadingMock(_, _)).Times(0);
  EXPECT_CALL(UnavailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto UnavailableRoot = firestarter::measurement::RootMetric::fromCInterface(UnavailableMetricMock.AvailableMetric);

  EXPECT_FALSE(UnavailableRoot->Available);
  EXPECT_FALSE(UnavailableRoot->Initialized);

  // Check if the metric does not inititalize
  EXPECT_FALSE(UnavailableRoot->initialize());

  EXPECT_FALSE(UnavailableRoot->Initialized);
}

TEST(MetricsTest, CheckRegisterInsertCallbackFromCInterface) {
  // Check that the insert callback is provided during initialization
  {
    auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, InsertCallbackMetricType>::instance();
    EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
    EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
    EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

    auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

    // Check if the metric inititializes
    EXPECT_CALL(AvailableMetricMock,
                registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
        .Times(1);

    EXPECT_TRUE(AvailableRoot->initialize());
  }

  {
    auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
    EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
    EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
    EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

    auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

    // Check if the metric inititializes
    EXPECT_CALL(AvailableMetricMock,
                registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
        .Times(0);
    EXPECT_TRUE(AvailableRoot->initialize());
  }
}

TEST(MetricsTest, CheckNoTimedCallbackFromCInterface) {
  // Callback is not enabled (Callbacktime = 0)

  auto& AvailableMetricMock =
      MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType, /*CallbackTime=*/0U>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
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
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
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

TEST(MetricsTest, CheckInsertCallbackFromCInterface) {
  // Check that the insert callback is returned for a pulling metric
  {
    auto& PullingMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
    EXPECT_CALL(PullingMetricMock, finiMock()).Times(2);
    EXPECT_CALL(PullingMetricMock, getSubmetricNamesMock()).Times(0);

    auto PullingRoot = firestarter::measurement::RootMetric::fromCInterface(PullingMetricMock.AvailableMetric);

    // Check if the metric inititializes
    EXPECT_CALL(PullingMetricMock,
                registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, PullingRoot.get()))
        .Times(0);

    {
      auto Callback = PullingRoot->getInsertCallback();
      EXPECT_TRUE(Callback.has_value());

      EXPECT_CALL(PullingMetricMock, getReadingMock(_, _)).Times(0);
      Callback.value()();
    }

    EXPECT_TRUE(PullingRoot->initialize());

    {
      auto Callback = PullingRoot->getInsertCallback();
      EXPECT_TRUE(Callback.has_value());

      // mock the get reading function
      EXPECT_CALL(PullingMetricMock, getReadingMock(_, _)).Times(1).WillOnce(Return(EXIT_SUCCESS));

      auto TimeBefore = std::chrono::high_resolution_clock::now();

      Callback.value()();

      auto TimeAfter = std::chrono::high_resolution_clock::now();

      // check that the value has been added to the Value vector.
      EXPECT_FALSE(PullingRoot->Values.empty());

      const auto& [Time, Value] = PullingRoot->Values.back();

      EXPECT_LE(TimeBefore, Time);
      EXPECT_LE(Time, TimeAfter);

      EXPECT_EQ(Value, PullingMetricMock.FakeValue);
    }
  }

  // No callback from a pushing metric
  {
    auto& PushingMetricMock = MetricMock</*InitReturnValue=*/0, InsertCallbackMetricType>::instance();
    EXPECT_CALL(PushingMetricMock, finiMock()).Times(1);
    EXPECT_CALL(PushingMetricMock, getSubmetricNamesMock()).Times(0);
    EXPECT_CALL(PushingMetricMock, getReadingMock(_, _)).Times(0);
    EXPECT_CALL(PushingMetricMock, registerInsertCallbackMock(_, _)).Times(0);

    auto PushingRoot = firestarter::measurement::RootMetric::fromCInterface(PushingMetricMock.AvailableMetric);

    // No callback
    auto Callback = PushingRoot->getInsertCallback();
    EXPECT_FALSE(Callback.has_value());
  }
}

TEST(MetricsTest, CheckAvailableFromStdin) {
  auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);
  EXPECT_CALL(AvailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto AvailableRoot =
      firestarter::measurement::RootMetric::fromStdin(MetricMock<0, NoInsertCallbackMetricType, 0>::FakeName);

  // Metric is created, it is not yet initialized.
  EXPECT_EQ(AvailableRoot->Name, AvailableMetricMock.FakeName);
  EXPECT_TRUE(AvailableRoot->Values.empty());
  EXPECT_EQ(AvailableRoot->MetricPtr, nullptr);
  EXPECT_TRUE(AvailableRoot->Submetrics.empty());
  EXPECT_FALSE(AvailableRoot->Dylib);
  EXPECT_TRUE(AvailableRoot->Stdin);
  EXPECT_TRUE(AvailableRoot->Initialized);

  EXPECT_TRUE(AvailableRoot->Available);

  // Check if the metric inititializes
  EXPECT_TRUE(AvailableRoot->initialize());

  EXPECT_TRUE(AvailableRoot->Initialized);
}

TEST(MetricsTest, CheckNoTimedCallbackFromStdin) {
  auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

  auto AvailableRoot =
      firestarter::measurement::RootMetric::fromStdin(MetricMock<0, NoInsertCallbackMetricType, 0>::FakeName);

  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
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

TEST(MetricsTest, CheckNoInsertCallbackFromStdin) {
  auto& AvailableMetricMock = MetricMock</*InitReturnValue=*/0, InsertCallbackMetricType>::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(0);
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);

  auto AvailableRoot =
      firestarter::measurement::RootMetric::fromStdin(MetricMock<0, InsertCallbackMetricType, 0>::FakeName);

  // Check if the metric inititializes
  EXPECT_CALL(AvailableMetricMock,
              registerInsertCallbackMock(firestarter::measurement::RootMetric::insertCallback, AvailableRoot.get()))
      .Times(0);

  {
    auto Callback = AvailableRoot->getInsertCallback();
    EXPECT_FALSE(Callback.has_value());
  }

  EXPECT_TRUE(AvailableRoot->initialize());

  {
    auto Callback = AvailableRoot->getInsertCallback();
    EXPECT_FALSE(Callback.has_value());
  }
}

TEST(MetricsTest, CheckSubmetricsFromCInterface) {
  using Mock =
      MetricMock</*InitReturnValue=*/0, NoInsertCallbackMetricType, /*CallbackTime=*/0U, /*UseSubmetrics=*/true>;

  std::array<const char*, 3> Submetrics = {
      "submetric-0",
      "submetric-1",
      nullptr,
  };

  auto& AvailableMetricMock = Mock::instance();
  EXPECT_CALL(AvailableMetricMock, finiMock()).Times(2);
  EXPECT_CALL(AvailableMetricMock, getSubmetricNamesMock()).Times(1).WillOnce(Return(Submetrics.data()));
  EXPECT_CALL(AvailableMetricMock, getReadingMock(_, _)).Times(0);
  EXPECT_CALL(AvailableMetricMock, registerInsertCallbackMock(_, _)).Times(0);

  auto AvailableRoot = firestarter::measurement::RootMetric::fromCInterface(AvailableMetricMock.AvailableMetric);

  // Metric is created, it is not yet initialized.
  EXPECT_EQ(AvailableRoot->Name, AvailableMetricMock.FakeName);
  EXPECT_TRUE(AvailableRoot->Values.empty());
  EXPECT_EQ(AvailableRoot->MetricPtr, &AvailableMetricMock.AvailableMetric);
  EXPECT_FALSE(AvailableRoot->Dylib);
  EXPECT_FALSE(AvailableRoot->Stdin);
  EXPECT_FALSE(AvailableRoot->Initialized);
  EXPECT_TRUE(AvailableRoot->Available);

  // Check that the submetrics are registered
  EXPECT_EQ(AvailableRoot->Submetrics.size(), 2);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Name, std::string(Submetrics[0]));
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Name, std::string(Submetrics[1]));

  // Check if the metric inititializes
  EXPECT_TRUE(AvailableRoot->initialize());

  EXPECT_TRUE(AvailableRoot->Initialized);

  // Check that submetrics are inserted
  const auto TV0 = firestarter::measurement::TimeValue(std::chrono::high_resolution_clock::now(), 0.0);
  const auto TV1 = firestarter::measurement::TimeValue(std::chrono::high_resolution_clock::now(), 0.0);
  const auto TV2 = firestarter::measurement::TimeValue(std::chrono::high_resolution_clock::now(), 0.0);
  const auto TV3 = firestarter::measurement::TimeValue(std::chrono::high_resolution_clock::now(), 0.0);

  // Insert into root
  AvailableRoot->insert(ROOT_METRIC_INDEX, TV0.Time, TV0.Value);
  EXPECT_EQ(AvailableRoot->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values.size(), 0);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values.size(), 0);
  EXPECT_EQ(AvailableRoot->Values[0], TV0);

  // Insert into the first submetric
  AvailableRoot->insert(1, TV1.Time, TV1.Value);
  EXPECT_EQ(AvailableRoot->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values.size(), 0);
  EXPECT_EQ(AvailableRoot->Values[0], TV0);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values[0], TV1);

  // Insert into the second submetric
  AvailableRoot->insert(2, TV2.Time, TV2.Value);
  EXPECT_EQ(AvailableRoot->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Values[0], TV0);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values[0], TV1);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values[0], TV2);

  // Inserting in a invalid metric does not change anything
  AvailableRoot->insert(3, TV3.Time, TV3.Value);
  EXPECT_EQ(AvailableRoot->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values.size(), 1);
  EXPECT_EQ(AvailableRoot->Values[0], TV0);
  EXPECT_EQ(AvailableRoot->Submetrics[0]->Values[0], TV1);
  EXPECT_EQ(AvailableRoot->Submetrics[1]->Values[0], TV2);
}

TEST(MetricsTest, MeasurementWorkerUsesUpdatedThreadCount) {
  const auto OriginalFrequency = PerfMetric::PerfFreqMetric;
  const auto OriginalIpc = PerfMetric::PerfIpcMetric;
  struct RestoreMetrics {
    MetricInterface Frequency;
    MetricInterface Ipc;
    ~RestoreMetrics() {
      PerfMetric::PerfFreqMetric = Frequency;
      PerfMetric::PerfIpcMetric = Ipc;
    }
  } Restore{OriginalFrequency, OriginalIpc};

  PerfMetric::PerfFreqMetric = {"perf-freq", {0, 1, 1, 1, 0, 0}, "GHz", 0, nullptr, init<0>, init<0>, nullptr, nullptr,
                                getError,    registerFrequency};
  PerfMetric::PerfIpcMetric = {"perf-ipc", {1, 0, 0, 1, 0, 0}, "IPC",      0, nullptr, init<0>, init<0>, nullptr,
                               nullptr,    getError,           registerIpc};

  firestarter::measurement::MeasurementWorker Worker(std::chrono::milliseconds(1), 4, {}, {});
  const auto Frequency = firestarter::MetricName::fromString("perf-freq");
  const auto Ipc = firestarter::MetricName::fromString("perf-ipc");
  Worker.initMetrics({Frequency, Ipc});
  ASSERT_NE(FrequencyInsert, nullptr);
  ASSERT_NE(IpcInsert, nullptr);

  for (const auto ThreadCount : {4U, 2U, 8U}) {
    if (ThreadCount != 4U) {
      Worker.setNumThreads(ThreadCount);
    }
    Worker.startMeasurement();
    const auto InsertValues = [](double FrequencyValue) {
      const auto Time = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::high_resolution_clock::now().time_since_epoch())
                            .count();
      FrequencyInsert(FrequencyContext, ROOT_METRIC_INDEX, Time, FrequencyValue);
      IpcInsert(IpcContext, ROOT_METRIC_INDEX, Time, 3.0);
    };
    InsertValues(0.0);
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    InsertValues(100.0);

    const auto Values = Worker.getValues();
    ASSERT_EQ(Values.at(Frequency).NumTimepoints, 1);
    EXPECT_GT(Values.at(Frequency).Average, 0.0);
    EXPECT_DOUBLE_EQ(Values.at(Ipc).Average, 3.0);

    Worker.setNumThreads(1);
    const auto Unnormalized = Worker.getValues();
    EXPECT_NEAR(Values.at(Frequency).Average * ThreadCount, Unnormalized.at(Frequency).Average,
                Unnormalized.at(Frequency).Average * 1e-12);
    EXPECT_DOUBLE_EQ(Unnormalized.at(Ipc).Average, 3.0);
  }
}