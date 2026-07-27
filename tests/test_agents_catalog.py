from app.catalog.agents import load_agent_title_labels


def test_load_agent_title_labels_without_platform_url() -> None:
    assert load_agent_title_labels("", 5) == ()
