import re


def load_outlines_from_file(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as f:
        return load_outlines(f.read())


def load_outlines(text: str) -> dict:
    lines = text.strip().split('\n')
    result = {}
    current_section = None

    for line in lines:
        if line.startswith("#"):
            continue

        is_section_title = line.startswith('- ')

        # Remove leading dash and space if present
        line = re.sub(r'^- ', '', line)
        line = line.strip().strip('-').strip()
        if not line:
            continue

        # Remove chapter numbers
        line = re.sub(r'^第[一二三四五六七八九十]+章\s*', '', line)

        if is_section_title:
            current_section = line
            if current_section not in result:
                result[current_section] = []
            else:
                raise ValueError(f"Duplicate section title: {current_section}")
        else:
            if current_section:
                result[current_section].append(line.strip())
            else:
                raise ValueError(f"Content outside of section: {line}")

    return result


if __name__ == '__main__':
    # Example usage
    markdown_text = """
## 测试
- 第一章 有理数
  - 正数和负数
  - 有理数的概念
  - 有理数的加法
- 第二章 整式的加减
  - 整式的概念
  - 单项式
  - 多项式
"""

    parsed_dict = load_outlines(markdown_text)
    print(parsed_dict)
