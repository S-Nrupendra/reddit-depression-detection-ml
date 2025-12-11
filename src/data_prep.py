import re
import html

def remove_urls(text):
    return re.sub(r"https\S+|www\.\S+", "", text)

def remove_html_tags(text):
    return re.sub(r"<.*?>", "", text)

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags
        "\U00002702-\U000027B0"  # misc
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r"",text)

def remove_special_chars(text):
    return re.sub(r"[^A-Za-z0-9\s]", " ", text)

def normalize_spaces(text):
    return re.sub(r"\s+", " ", text).strip()

def basic_cleaning(text):
    if not isinstance(text, str):
        return ""
    
    text = html.unescape(text)
    text = text.lower()
    text = remove_urls(text)
    text = remove_html_tags(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    text = normalize_spaces(text)
    
    return text