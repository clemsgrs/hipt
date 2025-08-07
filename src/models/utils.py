def update_state_dict(
    *,
    model_dict: dict,
    state_dict: dict,
):
    """
    Matches weights between `model_dict` and `state_dict`, accounting for:
    - Key mismatches (missing in model_dict)
    - Shape mismatches (tensor size differences)

    Args:
        model_dict (dict): model state dictionary (expected keys and shapes)
        state_dict (dict): checkpoint state dictionary (loaded keys and values)

    Returns:
        updated_state_dict (dict): Weights mapped correctly to `model_dict`
        msg (str): Log message summarizing the result
    """
    success = 0
    shape_mismatch = 0
    missing_keys = 0
    updated_state_dict = {}
    shape_mismatch_list = []
    missing_keys_list = []
    used_keys = set()
    for model_key, model_val in model_dict.items():
        matched_key = False
        for state_key, state_val in state_dict.items():
            if state_key in used_keys:
                continue
            if model_key == state_key:
                if model_val.size() == state_val.size():
                    updated_state_dict[model_key] = state_val
                    used_keys.add(state_key)
                    success += 1
                    matched_key = True  # key is successfully matched
                    break
                else:
                    shape_mismatch += 1
                    shape_mismatch_list.append(model_key)
                    matched_key = True  # key is matched, but weight cannot be loaded
                    break
        if not matched_key:
            # key not found in state_dict
            updated_state_dict[model_key] = model_val  # keep original weights
            missing_keys += 1
            missing_keys_list.append(model_key)
    # log summary
    msg = f"{success}/{len(model_dict)} weight(s) loaded successfully"
    if shape_mismatch > 0:
        msg += f"\n{shape_mismatch} weight(s) not loaded due to mismatching shapes: {shape_mismatch_list}"
    if missing_keys > 0:
        msg += f"\n{missing_keys} key(s) from checkpoint not found in model: {missing_keys_list}"
    return updated_state_dict, msg