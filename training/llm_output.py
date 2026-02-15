def remove_duplicates(input_list):
    if not isinstance(input_list, list):
        raise ValueError("Input must be a list")
    if len(input_list) == 0:
        return []
    seen = set()
    result = []
    for element in input_list:
        if element not in seen:
            seen.add(element)
            result.append(element)
    return result

def improve_remove_duplicates(input_list=None):
    try:
        if input_list is None:
            raise ValueError("Input must be a list")
        # Check for empty list
        if len(input_list) == 0:
            return []
        # Handle non-list inputs
        if not isinstance(input_list, list):
            raise ValueError("Input must be a list")
        seen = set()
        result = []
        for element in input_list:
            if element not in seen:
                seen.add(element)
                result.append(element)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        raise