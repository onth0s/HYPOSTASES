from hypostases.utils.merge_spec import strip_yaml_frontmatter


def test_strip_yaml_frontmatter_present():
    content = "---\ntitle: Spec Part I\nauthor: Antigravity\n---\n# Main spec title\nSpec content goes here."
    stripped = strip_yaml_frontmatter(content)
    assert stripped == "# Main spec title\nSpec content goes here."


def test_strip_yaml_frontmatter_absent():
    content = "# Main spec title\nSpec content goes here."
    stripped = strip_yaml_frontmatter(content)
    assert stripped == content


def test_strip_yaml_frontmatter_empty():
    assert strip_yaml_frontmatter("") == ""


def test_strip_yaml_frontmatter_with_leading_whitespace():
    content = "\n\n  \n---\ntitle: Spec Part I\n---\n# Title"
    stripped = strip_yaml_frontmatter(content)
    assert stripped == "# Title"


def test_strip_yaml_frontmatter_windows_line_endings():
    content = "---\r\ntitle: Spec Part I\r\nauthor: Antigravity\r\n---\r\n# Main spec title\r\nSpec content goes here."
    stripped = strip_yaml_frontmatter(content)
    assert stripped == "# Main spec title\nSpec content goes here."
