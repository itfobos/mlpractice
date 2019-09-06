MISS = 'Miss. '
DR = 'Dr. '
NOT_MISS = '. '


def to_female_name(value):
    if MISS in value:
        result = _extract_miss_name(value)
    elif DR in value:
        result = _extract_doctor_name(value)
    else:
        result = _extract_not_miss_name(value)

    return result.strip()


def _extract_not_miss_name(value):
    parts = value.split('. ')
    if len(parts) < 2:
        raise ValueError('Wrong not Miss name: ' + value)

    after_delimiter_part = parts[-1]

    if '(' in after_delimiter_part:
        return _extract_brackets_mrs_name(after_delimiter_part)

    return _extract_no_brackets_mrs_name(after_delimiter_part)


def _extract_doctor_name(value):
    after_dr_part = value.split(DR)[-1]
    return after_dr_part.split()[0]


def _extract_brackets_mrs_name(value):
    left_bracket_index = value.find('(')
    if left_bracket_index < 0:
        raise ValueError('Mrs. name has no left round bracket: ' + value)

    right_bracket_index = value.find(')')
    if right_bracket_index < 0:
        raise ValueError('Mrs. name has no right round bracket ' + value)

    return value[left_bracket_index + 1: right_bracket_index].split()[0]


def _extract_no_brackets_mrs_name(after_delimiter_part):
    return after_delimiter_part.split()[0]


def _extract_miss_name(value):
    after_miss_part = value.split(MISS)[-1]
    return after_miss_part.split()[0]
