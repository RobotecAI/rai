from collections import defaultdict
from typing import Dict

import pytest
from _pytest.terminal import TerminalReporter
from tabulate import tabulate


@pytest.fixture
def chat_openai_multimodal():
    from langchain_openai.chat_models import ChatOpenAI

    return ChatOpenAI(model="gpt-4o")


@pytest.fixture
def chat_openai_text():
    from langchain_openai.chat_models import ChatOpenAI

    return ChatOpenAI(model="gpt-3.5")


@pytest.fixture
def chat_bedrock_multimodal():
    from langchain_aws.chat_models import ChatBedrock

    return ChatBedrock(  # type: ignore
        model_id="anthropic.claude-3-haiku-20240307-v1:0", region_name="us-west-2"
    )


class UsageTracker:
    def __init__(self):
        self.vendor_cost: Dict[str, float] = defaultdict(lambda: 0.0)
        self.vendor_total_tokens: Dict[str, int] = defaultdict(lambda: 0)
        self.vendor_input_tokens: Dict[str, int] = defaultdict(lambda: 0)
        self.vendor_output_tokens: Dict[str, int] = defaultdict(lambda: 0)
        self.total_cost = 0

    def add_usage(
        self,
        vendor: str,
        cost: float,
        total_tokens: int = 0,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        self.vendor_cost[vendor] += cost
        self.vendor_total_tokens[vendor] += total_tokens
        self.vendor_input_tokens[vendor] += input_tokens
        self.vendor_output_tokens[vendor] += output_tokens

    def get_vendor_cost(self, vendor: str) -> float:
        return self.vendor_cost[vendor]

    def get_total_cost(self) -> float:
        return sum(self.vendor_cost.values())

    def get_vendor_total_tokens(self, vendor: str) -> int:
        return self.vendor_total_tokens[vendor]

    def get_total_tokens(self) -> int:
        return sum(self.vendor_total_tokens.values())

    def get_vendor_input_tokens(self, vendor: str) -> int:
        return self.vendor_input_tokens[vendor]

    def get_total_input_tokens(self) -> int:
        return sum(self.vendor_input_tokens.values())

    def get_vendor_output_tokens(self, vendor: str) -> int:
        return self.vendor_output_tokens[vendor]

    def get_total_output_tokens(self) -> int:
        return sum(self.vendor_output_tokens.values())


tracker = UsageTracker()


@pytest.fixture
def usage_tracker():
    return tracker


def pytest_terminal_summary(
    terminalreporter: TerminalReporter, exitstatus: int, config: pytest.Config
) -> None:
    terminalreporter.section("Usage Summary")
    table = [
        [vendor, f"{cost:.4f}", total_tokens, input_tokens, output_tokens]
        for vendor, cost, total_tokens, input_tokens, output_tokens in zip(
            tracker.vendor_cost.keys(),
            tracker.vendor_cost.values(),
            tracker.vendor_total_tokens.values(),
            tracker.vendor_input_tokens.values(),
            tracker.vendor_output_tokens.values(),
        )
    ]
    table.append(
        [
            "Total",
            f"{tracker.get_total_cost():.4f}",
            tracker.get_total_tokens(),
            tracker.get_total_input_tokens(),
            tracker.get_total_output_tokens(),
        ]
    )
    terminalreporter.write(
        f"{tabulate(table, headers=['Vendor', 'Cost $', 'Total Tokens', 'Input Tokens', 'Output Tokens'], tablefmt='grid')}\n"
    )


def pytest_configure(config: pytest.Config) -> None:
    config.pluginmanager.register(tracker, "usage_tracker")
