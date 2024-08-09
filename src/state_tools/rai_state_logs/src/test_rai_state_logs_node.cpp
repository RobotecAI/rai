#include <chrono>
#include <thread>

#include <rclcpp/rclcpp.hpp>

// Move to tests etc.
int main(int argc, char ** argv)
{
  using namespace std::chrono_literals;
  rclcpp::init(argc, argv);
  auto node = std::make_shared<rclcpp::Node>("test_rai_state_logs_node");
  int i = 0;
  while (rclcpp::ok()) {
    RCLCPP_INFO(node->get_logger(), "test info constant");
    RCLCPP_INFO(node->get_logger(), "test info with numbers: '%d'", i);
    RCLCPP_ERROR(node->get_logger(), "test error");
    RCLCPP_ERROR(node->get_logger(), "test error with numbers: '%d'", i);
    RCLCPP_WARN(node->get_logger(), "test warn with numbers: '%d'", i);
    i++;
    std::this_thread::sleep_for(100ms);
  }
  rclcpp::shutdown();
  return 0;
}
