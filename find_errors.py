import re

log_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\563f741e-f2d5-4d96-8abb-583b92b85534\.system_generated\tasks\task-134.log"
out_path = "output/log/task_errors.txt"

with open(log_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

found_lines = []
for idx, line in enumerate(lines):
    # Search for common error patterns
    if any(p in line for p in ["Traceback", "Exception", "Error", "failed", "Failed", "raise"]):
        found_lines.append((idx, line.strip()))

with open(out_path, 'w', encoding='utf-8') as f_out:
    f_out.write(f"Total lines in log: {len(lines)}\n")
    f_out.write(f"Found {len(found_lines)} potential error lines:\n\n")
    for idx, line in found_lines:
        f_out.write(f"Line {idx}: {line}\n")
        # Write context around the line
        f_out.write("Context:\n")
        for k in range(max(0, idx-3), min(len(lines), idx+8)):
            f_out.write(f"  {k}: {lines[k]}")
        f_out.write("-" * 40 + "\n")

print(f"Analysis saved to {out_path}")
