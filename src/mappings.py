# each list must have the same length
def create_string_mappings(before: list[str], after: list[str], delimiter=" -> "):
    return [("".join([pre, delimiter, post])) for (pre, post) in zip(before, after)]


def create_mappings_dict(before: list[str], after: list[str]):
    return {pre: post for (pre, post) in zip(before, after)}

#create_mappings_dict(["a", "b", "c"], ["x", "y", "z"])