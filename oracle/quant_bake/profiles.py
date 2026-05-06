from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QuantProfile:
    name: str
    layout: str
    runtime: str
    experimental: bool = False


NATIVE_INT4_PROFILES: dict[str, QuantProfile] = {
    "int4-gptq": QuantProfile("int4-gptq", "Int4Quantized", "native-int4"),
    "int4-awq": QuantProfile("int4-awq", "Int4Quantized", "native-int4"),
    "int4-autoround": QuantProfile("int4-autoround", "Int4Quantized", "native-int4"),
    "int4-hqq": QuantProfile("int4-hqq", "Int4Quantized", "native-int4"),
}

RUNTIME_BACKED_PROFILES: dict[str, QuantProfile] = {
    "higgs4": QuantProfile("higgs4", "HiggsGridQuantized", "higgs4"),
    "quip-e8": QuantProfile("quip-e8", "QuipE8Quantized", "quip-e8"),
    "qtip-trellis2": QuantProfile(
        "qtip-trellis2",
        "QtipTrellisQuantized",
        "qtip-trellis2",
        experimental=True,
    ),
}

ALIASES = {
    "gptq": "int4-gptq",
    "awq": "int4-awq",
    "autoround": "int4-autoround",
    "signround": "int4-autoround",
    "hqq": "int4-hqq",
    "higgs-4": "higgs4",
    "quipe8": "quip-e8",
    "qtip": "qtip-trellis2",
}


def parse_profile(raw: str | None) -> QuantProfile:
    name = ALIASES.get(raw or "int4-gptq", raw or "int4-gptq")
    if name in NATIVE_INT4_PROFILES:
        return NATIVE_INT4_PROFILES[name]
    if name in RUNTIME_BACKED_PROFILES:
        return RUNTIME_BACKED_PROFILES[name]
    valid = ", ".join(sorted([*NATIVE_INT4_PROFILES, *RUNTIME_BACKED_PROFILES]))
    raise ValueError(f"unknown quant profile {raw!r}; expected one of: {valid}")
