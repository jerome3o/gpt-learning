import torch


def sizeof_fmt(
    num, 
    suffix="B"
):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


def mem_size(t: torch.Tensor):
    return sizeof_fmt(t.element_size() * t.numel())


def model_mem_size(t: torch.nn.Module):
    total_bytes = 0

    for p in t.parameters():
        total_bytes += p.element_size() * p.numel()

    print(sizeof_fmt(total_bytes))


def mem_summary():
    for d in range(torch.cuda.device_count()):
        avail, total = torch.cuda.mem_get_info(d)
        used = total - avail
        print(f"cuda:{d}: {sizeof_fmt(used)} ({(used / total)*100:0.2f}%)")