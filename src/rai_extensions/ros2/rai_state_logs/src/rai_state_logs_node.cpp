// Copyright (C) 2024 Robotec.AI
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//         http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <rai_interfaces/srv/string_list.hpp>
#include <rcl_interfaces/msg/log.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rcutils/time.h>

class LogDigestNode : public rclcpp::Node
{
public:
  using Content = std::tuple<std::string, size_t, int64_t>;

  LogDigestNode()
  : Node("rai_state_logs_node")
  {
    constexpr uint16_t default_limit = 512;
    this->declare_parameter("filters", std::vector<std::string>());
    this->declare_parameter("max_lines", default_limit);
    this->declare_parameter("include_meta", true);
    this->declare_parameter("clear_on_retrieval", true);
    filters_ = get_parameter("filters").as_string_array();
    max_lines_ = static_cast<uint16_t>(get_parameter("max_lines").as_int());
    include_meta_ = get_parameter("include_meta").as_bool();
    clear_on_retrieval_ = get_parameter("clear_on_retrieval").as_bool();

    // Hack to overcome https://github.com/ros2/rclcpp/issues/1955
    if (filters_.size() == 1 && filters_[0].empty()) {
      filters_.clear();
    }

    RCLCPP_INFO(get_logger(), "filters: %s", std::to_string(filters_.size()).c_str());

    if (max_lines_ < 1) {
      RCLCPP_WARN(get_logger(), "Invalid value of max_lines_ parameter, reverting to default");
      max_lines_ = default_limit;
    }

    const size_t history_depth = 100;
    log_subscription_ = this->create_subscription<rcl_interfaces::msg::Log>(
      "rosout", history_depth,
      std::bind(&LogDigestNode::log_callback, this, std::placeholders::_1));

    log_digest_srv_ = create_service<rai_interfaces::srv::StringList>(
      "get_log_digest", std::bind(
        &LogDigestNode::digest_request_callback, this, std::placeholders::_1,
        std::placeholders::_2));
  }

private:
  void digest_request_callback(
    [[maybe_unused]] const std::shared_ptr<rai_interfaces::srv::StringList::Request> request,
    const std::shared_ptr<rai_interfaces::srv::StringList::Response> response)
  {
    std::vector<std::string> responseLogs;
    for (const auto & [k, v] : timestamps_) {
      if (auto logElement = logs_.find(v); logElement != logs_.end()) {
        auto msg_string = std::get<0>(logElement->second);
        auto repetitions = std::get<1>(logElement->second);
        if (repetitions > 1) {
          msg_string += " [" + std::to_string(repetitions) + " occurrences]";
        }
        responseLogs.push_back(msg_string);
      }
    }

    response->success = true;
    response->string_list = responseLogs;
    if (clear_on_retrieval_) {
      timestamps_.clear();
      logs_.clear();
    }
  }

  std::string formatSource(const rcl_interfaces::msg::Log & msg) const
  {
    // Surprisingly, log to string isn't exposed from ROS.
    static const std::unordered_map<uint8_t, std::string> levelMapping = {
      {rcl_interfaces::msg::Log::DEBUG, "DEBUG"}, {rcl_interfaces::msg::Log::INFO, "INFO"},
      {rcl_interfaces::msg::Log::WARN, "WARN"}, {rcl_interfaces::msg::Log::ERROR, "ERROR"},
      {rcl_interfaces::msg::Log::FATAL, "FATAL"},
    };
    auto levelMapElement = levelMapping.find(msg.level);
    std::string levelString("INVALID_LEVEL");
    if (levelMapElement != levelMapping.end()) {
      levelString = levelMapElement->second;
    }

    // C++20 std::format
    // [Node][Level][Source Code Origin]: Message
    const std::string filename = std::filesystem::path(msg.file).filename();
    return std::string("[") + msg.name + "] [" + levelString + "] [" + filename + "." +
           msg.function + "(" + std::to_string(msg.line) + ")]";
  }

  std::string formatMessage(const rcl_interfaces::msg::Log & msg) const
  {
    return formatSource(msg) + ": " + msg.msg;
  }

  bool passesFilters(const std::string & text) const
  {
    return std::any_of(
      filters_.cbegin(), filters_.cend(), [&text](const std::string & f) {
        return text.find(f) != std::string::npos;
      });
  }

  void log_callback(const rcl_interfaces::msg::Log & msg)
  {
    // This can be exposed to configuration if needed.
    const auto threshold_level = rcl_interfaces::msg::Log::WARN;
    const auto jointText = formatMessage(msg);
    const bool isImportant = msg.level >= threshold_level;
    const bool isInteresting = passesFilters(jointText);
    if (!isImportant && !isInteresting) {
      return;
    }

    const int64_t nanoTimestamp =
      RCUTILS_S_TO_NS(static_cast<int64_t>(msg.stamp.sec)) + msg.stamp.nanosec;

    std::string source = formatSource(msg);
    if (auto sameSource = logs_.find(source); sameSource != logs_.end()) {
      auto & content = sameSource->second;
      std::get<1>(content)++;
      auto oldTimestamp = std::get<2>(content);
      if (auto found = timestamps_.find(oldTimestamp); found != timestamps_.end()) {
        timestamps_.erase(found);
      }
      std::get<2>(content) = nanoTimestamp;
    } else {
      auto storageText = include_meta_ ? jointText : msg.msg;
      logs_.insert(std::make_pair(source, std::make_tuple(storageText, 1u, nanoTimestamp)));
    }

    timestamps_[nanoTimestamp] = source;

    // Extensions: consider priorities, never cutting off errors, etc.
    // Also, this is not very efficient structure-wise to keep a rolling window.
    if (timestamps_.size() > max_lines_) {
      if (auto lastElement = std::prev(timestamps_.end()); lastElement != timestamps_.end()) {
        auto sourceToDelete = lastElement->second;
        if (auto logToDelete = logs_.find(sourceToDelete); logToDelete != logs_.end()) {
          logs_.erase(logToDelete);
        }
        timestamps_.erase(lastElement);
      }
    }
  }

  rclcpp::Subscription<rcl_interfaces::msg::Log>::SharedPtr log_subscription_;
  rclcpp::Service<rai_interfaces::srv::StringList>::SharedPtr log_digest_srv_;

  bool include_meta_;
  bool clear_on_retrieval_;
  uint16_t max_lines_;
  std::vector<std::string> filters_;
  std::unordered_map<std::string, Content> logs_;
  std::map<int64_t, std::string, std::greater<int64_t>> timestamps_;
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<LogDigestNode>());
  rclcpp::shutdown();
  return 0;
}
