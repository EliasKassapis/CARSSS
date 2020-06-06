def all_file_paths(path):
    return sorted([p for p in path.iterdir() if not p.is_dir()])
