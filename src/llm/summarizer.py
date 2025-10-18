from typing import List

def simple_protocol_classifier(log_lines: List[str]) -> List[int]:
    """
    Very naive heuristic classifier for demonstration.
    Maps log lines to pseudo protocol categories.
    0 = general, 1 = tcp, 2 = udp, 3 = suspicious
    """
    labels = []
    for line in log_lines:
        l = line.lower()
        if "tcp" in l:
            labels.append(1)
        elif "udp" in l:
            labels.append(2)
        elif "denied" in l or "alert" in l or "attack" in l:
            labels.append(3)
        else:
            labels.append(0)
    return labels