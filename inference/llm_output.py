def lengthOfLongestSubstring(s: str) -> int:
    max_len = 0
    char_index = {}
    
    for i, char in enumerate(s):
        if char in char_index and char_index[char] >= i - max_len:
            max_len = i - char_index[char]
        else:
            max_len = max(max_len, i - char_index.get(char, -1))
        char_index[char] = i
    
    return max_len