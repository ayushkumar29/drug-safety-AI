"""
LLM Configuration Test Script
Verifies environment variable loading for LM Studio, OpenAI, and fallback configurations without making live API calls or exposing secrets.
"""

import os
try:
    from llm.config import LLMConfig
except ImportError:
    from config import LLMConfig



def test_lm_studio_config():
    print("================================================================================")
    print("TESTING LM STUDIO CONFIGURATION LOADING")
    print("================================================================================")

    # Set LM Studio environment variables in memory
    os.environ["LLM_PROVIDER"] = "lmstudio"
    os.environ["LLM_MODEL"] = "google/gemma-4-e4b"
    os.environ["LLM_BASE_URL"] = "http://10.174.159.49:1234/v1"
    os.environ["LLM_API_KEY"] = "lm-studio"

    config = LLMConfig.from_env()

    assert config.provider == "lmstudio", f"Provider mismatch: {config.provider}"
    assert config.model == "google/gemma-4-e4b", f"Model mismatch: {config.model}"
    assert config.base_url == "http://10.174.159.49:1234/v1", f"Base URL mismatch: {config.base_url}"
    assert config.api_key is not None and len(config.api_key) > 0, "API key missing!"

    print("  [OK] Provider loaded: ", config.provider)
    print("  [OK] Model loaded:    ", config.model)
    print("  [OK] Base URL loaded: ", config.base_url)
    print("  [OK] API Key exists:   YES (length =", len(config.api_key), "chars - secret hidden)")
    print("LM Studio Configuration Test: PASS [OK]")


def test_openai_config():
    print("\n================================================================================")
    print("TESTING OPENAI CONFIGURATION LOADING")
    print("================================================================================")

    os.environ["LLM_PROVIDER"] = "openai"
    os.environ["LLM_MODEL"] = "gpt-4o-mini"
    os.environ["LLM_BASE_URL"] = "https://api.openai.com/v1"
    os.environ["LLM_API_KEY"] = "test-key-placeholder"

    config = LLMConfig.from_env()

    assert config.provider == "openai", f"Provider mismatch: {config.provider}"
    assert config.model == "gpt-4o-mini", f"Model mismatch: {config.model}"
    assert config.base_url == "https://api.openai.com/v1", f"Base URL mismatch: {config.base_url}"
    assert config.api_key is not None, "API key missing!"

    print("  [OK] Provider loaded: ", config.provider)
    print("  [OK] Model loaded:    ", config.model)
    print("  [OK] Base URL loaded: ", config.base_url)
    print("  [OK] API Key exists:   YES (secret hidden)")
    print("OpenAI Configuration Test: PASS [OK]")


def test_fallback_config():
    print("\n================================================================================")
    print("TESTING FALLBACK CONFIGURATION LOADING")
    print("================================================================================")

    # Clear override vars
    for k in ["LLM_PROVIDER", "LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"]:
        os.environ.pop(k, None)

    config = LLMConfig.from_env()

    assert config.provider == "fallback", f"Provider mismatch: {config.provider}"
    assert config.model == "deterministic-fallback", f"Model mismatch: {config.model}"
    print("  [OK] Fallback Provider loaded: ", config.provider)
    print("  [OK] Fallback Model loaded:    ", config.model)
    print("Fallback Configuration Test: PASS [OK]")


if __name__ == "__main__":
    test_lm_studio_config()
    test_openai_config()
    test_fallback_config()
    print("\n================================================================================")
    print("ALL CONFIGURATION TESTS COMPLETED SUCCESSFULLY!")
    print("================================================================================")
