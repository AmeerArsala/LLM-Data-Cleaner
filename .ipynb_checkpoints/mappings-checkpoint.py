# each list must have the same length
def create_mappings(before: list[str], after: list[str], delimiter=" -> "):
    return [(pre + delimiter + post) for (pre, post) in zip(before, after)]


def create_mappings_as_dict(before: list[str], after: list[str]):
    return {pre: post for (pre, post) in zip(before, after)}

#create_mappings_as_dict(["a", "b", "c"], ["x", "y", "z"])
